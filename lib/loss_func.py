from contextlib import nullcontext

import math
import torch
import torch.nn as nn
import numpy as np
import umap
from pytorch_metric_learning.utils.loss_and_miner_utils import get_matches_and_diffs, \
    get_all_triplets_indices_vectorized_method, get_all_triplets_indices_loop_method

from lib.dataLoader import distributed_sinkhorn
from lib.semi import interleave
from kmeans_pytorch import kmeans

import torch.nn.functional as F


def scMatch_ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def scMatch_contrast_loss_clust(logits_w, logits_s, feats_ulb_w, n_clusts, device, temperature=1.0, clust_cutoff=0.8):
    logits_softmax = torch.softmax(logits_w.detach(), dim=-1)

    # feats_ulb_w = feats_ulb_w.detach().cpu().numpy()

    n_clusts = int(n_clusts)
    #     print(logits_softmax.shape, feats_ulb_w.shape, feats_ulb_w.dtype, n_clusts)

    # 1) get only selected logits and feats
    max_probs, max_idx = torch.max(logits_softmax, dim=-1)
    mask_bool = max_probs.ge(clust_cutoff)  # .cpu().numpy()
    del max_probs, max_idx

    # 2) 1. kmeans
    new_pseudo, _ = kmeans(X=feats_ulb_w, num_clusters=n_clusts)
    pseudo_onehot = torch.nn.functional.one_hot(new_pseudo, num_classes=n_clusts).float()
    pseudo_onehot = pseudo_onehot.to(device)

    # 3) get distribution for each cluster
    clust_dist = []
    for i in range(n_clusts):
        mask = new_pseudo==i
        tmp_dist_clust = logits_softmax[mask]
        if tmp_dist_clust.numel()==0:
            return torch.tensor(0.0),new_pseudo
        clust_dist.append(tmp_dist_clust.mean(0))
    dists = torch.stack(clust_dist)

    sim = torch.mm(torch.softmax(logits_s, dim=-1), dists.t() / temperature)  # B*N, K*N --> B *K
    sim_probs = sim / sim.sum(1, keepdim=True)

    loss_c = - ((torch.log(sim_probs + 1e-6) * pseudo_onehot)).sum(1)
    loss_c = loss_c * mask_bool.float()

    loss_c = loss_c.mean()

    return loss_c,new_pseudo



def scMatch_consistency_loss(logits_w, logits_s, y_ulb, name='ce', T=1.0, p_cutoff=0.0,
                     use_hard_labels=True, use_sharpen=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask_bool = max_probs.ge(p_cutoff)
        mask = mask_bool.float()

        ulb_acc_num = (max_idx == y_ulb).float() * mask

        with torch.no_grad():
            dist_ulb = pseudo_label.detach().cpu().mean(0).tolist()
            if mask.sum() >= 1:
                dist_ulb_high = pseudo_label[mask_bool].detach().cpu().mean(0).tolist()
            else:
                dist_ulb_high = None

        if use_hard_labels:
            masked_loss = scMatch_ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            if use_sharpen:
                # pseudo_label = torch.softmax(logits_w/T, dim=-1)
                pseudo_label = torch.softmax(pseudo_label / T, dim=-1)
            masked_loss = scMatch_ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), mask.sum(), ulb_acc_num.sum(), dist_ulb, dist_ulb_high

    else:
        assert Exception('Not Implemented consistency_loss')


class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u


def trap_loss(embedding, predict_label, anchor):
    threshold = 0.1
    if len(embedding) == 0:
        return torch.tensor(0).to(embedding.device)
    cos_dist = 1 - F.softmax(torch.mm(embedding, anchor.T), dim=1)
    loss_predict_target = torch.stack([torch.sign(dist[label]-threshold) * dist[label] for dist, label in zip(cos_dist, predict_label)])

    # loss_predict_non_target = []
    # for dist, label in zip(cos_dist, predict_label):
    #     for i in range(len(anchor)):
    #         if i != label:
    #             loss_predict_non_target.append(dist[label])
    # loss_predict_non_target = torch.stack(loss_predict_non_target)

    return torch.mean(loss_predict_target) # - torch.mean(loss_predict_non_target)


class SWAVLoss(nn.Module):
    """Modified from https://github.com/facebookresearch/swav"""

    def __init__(self, number_aug=8, temperature=0.1):
        super(SWAVLoss, self).__init__()
        self.number_aug = number_aug
        self.temperature = temperature

    def forward(self, embedding, anchor, bs, no_grad=True, query_dist=None):
        feature_x_C = torch.mm(embedding, anchor.T)
        # # do softmax
        # feature_x_C = F.softmax(feature_x_C, dim=1)
        aver_dist = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
        aver_q = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
        mean_embedding = torch.mean(torch.stack([embedding[bs * i: bs * (i + 1)] for i in range(self.number_aug)]), dim=0)
        for i, q_id in enumerate(range(self.number_aug)):
            if no_grad == True:
                with torch.no_grad():
                    out = feature_x_C[bs * q_id: bs * (q_id + 1)].detach()
                    if query_dist != None:
                        out = query_dist
                    aver_dist += out
                    # aver_var.append(torch.diag(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1))
                    # get assignments
                    q = distributed_sinkhorn(out) #[-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
                    # pred_labels.append(q)
                    aver_q += q
            else:
                out = feature_x_C[bs * q_id: bs * (q_id + 1)]
                if query_dist != None:
                    out = query_dist
                aver_dist += out
                # aver_var.append(torch.diag(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1))
                # get assignments
                q = distributed_sinkhorn(out)  # [-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
                # pred_labels.append(q)
                aver_q += q

        #     subloss = 0
        #     for v in np.delete(np.arange(np.sum(self.number_aug)), q_id):
        #         x = feature_x_C[bs * v: bs * (v + 1)] / self.temperature
        #         subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
        #     loss += subloss / (np.sum(self.number_aug) - 1)
        # loss /= self.number_q
        # aver_label = sum(pred_labels[:self.number_aug]) / self.number_aug
        aver_dist /= self.number_aug
        # aver_var = torch.stack(aver_var, dim=0).var(dim=0)
        aver_q /= self.number_aug

        return aver_dist, aver_q


def pca_torch(X, n_components):
    """
    Perform PCA on input tensor X and reduce to n_components dimensions.

    Args:
    - X: Input data, a PyTorch tensor of shape [n_samples, n_features].
    - n_components: Number of principal components to keep.

    Returns:
    - X_pca: The input data projected to n_components dimensions.
    """
    # Step 1: Center the data (subtract the mean)
    X_mean = X.mean(dim=0)  # Compute the mean of each feature
    X_centered = X - X_mean  # Subtract the mean to center the data

    # Step 2: Compute the covariance matrix (not divided by n-1 for simplicity)
    covariance_matrix = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    # Step 3: Perform SVD on the covariance matrix
    U, S, V = torch.svd(covariance_matrix)

    # Step 4: Select the top n_components principal components (eigenvectors)
    principal_components = V[:, :n_components]

    # Step 5: Project the centered data onto the principal components
    X_pca = X_centered @ principal_components

    return X_pca


# class SWAVGMMLoss(nn.Module):
#     """Modified from https://github.com/facebookresearch/swav"""
#
#     def __init__(self, number_aug=8, temperature=0.1):
#         super(SWAVGMMLoss, self).__init__()
#         self.number_aug = number_aug
#         self.temperature = temperature
#
#     #
#     # def GMM_dist(self, embedding, embedding_labels, anchors, labels):
#     #     pdf_all_class = []
#     #     num_a = int(len(embedding)/len(embedding_labels))
#     #     embedding_labels_fulll = torch.cat([embedding_labels for i in range(num_a)])
#     #     for i in torch.unique(labels):
#     #         # Select the anchors corresponding to the current class
#     #         anchor_class_i = anchors[labels == i]  # Adjust to select anchors for the current class
#     #         pdf_class_i = []
#     #         # Step 1: Center the data (subtract the mean of each feature)
#     #         embedding_i = embedding # [embedding_labels_fulll==i]
#     #         mean_embedding = embedding_i.mean(dim=0, keepdim=True)  # Mean of each feature across samples
#     #         centered_embedding = embedding_i - mean_embedding  # Center the data
#     #
#     #         # Step 2: Calculate the covariance matrix
#     #         covariance_matrix = (centered_embedding.T @ centered_embedding) / (embedding.shape[0] - 1)
#     #         torch.cholesky(covariance_matrix)
#     #         cholesky = cholesky_dict[i]
#     #
#     #         for anchor in anchor_class_i:
#     #             mean = anchor  # Use the current anchor as the mean
#     #             # covariance_matrix = torch.eye(len(anchor)).to(anchor.device)  # Identity matrix as covariance
#     #             mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
#     #
#     #             # Calculate the log probability and exponentiate to get the PDF
#     #             pdf = mvn.log_prob(embedding).exp()  # PDF values
#     #             pdf_class_i.append(pdf)
#     #
#     #         # Take the maximum PDF across all anchors for the current class
#     #         pdf_class_i = torch.max(torch.cat(pdf_class_i, dim=1), dim=1)[0]  # Max PDF values for class i
#     #         pdf_all_class.append(pdf_class_i)
#     #
#     #     # Concatenate the PDFs across all classes
#     #     pdf_all_class = torch.cat(pdf_all_class, dim=1)
#     #     #
#     #     # # Prepare to calculate the difference
#     #     # diff_temp = []
#     #     # for anchor in anchors:
#     #     #     diff_i = embedding - anchor  # Shape [n*bs, 128]
#     #     #     diff_temp.append(diff_i)
#     #     #
#     #     # # Stack differences for all anchors
#     #     # diff_temp = torch.cat(diff_temp, dim=1)  # Shape [n*bs, num_anchor * 128]
#     #     #
#     #     # # Compute the differences using broadcasting
#     #     # diff = embedding.unsqueeze(1) - anchors.unsqueeze(0)  # Shape [n*bs, num_anchors, 128]
#     #     #
#     #     # # Check that both diff methods yield the same result
#     #     # print(torch.equal(diff, diff_temp))
#     #     #
#     #     # # Solve for the linear system: L^T * y = diff
#     #     # # Ensure 'self.cholesky' is the Cholesky factor of the covariance matrix.
#     #     # y = torch.cholesky_solve(diff.view(-1, diff.size(-1)).transpose(0, 1), cholesky).transpose(0, 1)  # Shape [n*bs, num_anchors, 128]
#     #     #
#     #     # # Calculate the log determinant of the covariance matrix
#     #     # log_det = torch.logdet(cholesky)
#     #     #
#     #     # # Compute the log probability
#     #     # # Use einsum for the dot product, ensuring correct dimensionality
#     #     # log_prob = -0.5 * (self.dim * torch.log(2 * torch.pi) + log_det + torch.einsum('ijk,ikl->ij', diff, y))
#     #     #
#     #     # # Ensure that log_prob and pdf_all_class are compatible
#     #     # # pdf_all_class should match the shape of log_prob (adjust accordingly if necessary)
#     #     # # Since pdf_all_class is in PDF space, we want to make sure the following is true:
#     #     # pdf_from_log_prob = log_prob.exp()  # Convert log_prob back to PDF space
#     #     #
#     #     # # Check equality
#     #     # is_equal = torch.allclose(pdf_all_class, pdf_from_log_prob,
#     #     #                           atol=1e-6)  # Allow some tolerance due to numerical errors
#     #     # print("Are pdf_all_class and exp(log_prob) equal?:", is_equal)
#     #
#     #     return pdf_all_class
#
#     def forward(self, embedding, anchors, labels, bs, no_grad=True):
#         # X = torch.cat([anchors, embedding])
#         # X_ = pca_torch(X, n_components=10)
#         # anchors = X_[:len(anchors)]
#         # embedding = X_[len(anchors):]
#         X = torch.cat([anchors, embedding], dim=0)
#
#         # Step 2: Convert PyTorch tensor to NumPy array for UMAP
#         X_np = X.cpu().detach().numpy()
#
#         # Step 3: Apply UMAP to reduce to 10 dimensions
#         umap_reducer = umap.UMAP(n_components=2, random_state=42)  # Reduce to 10 dimensions
#         X_reduced = umap_reducer.fit_transform(X_np)
#         X_reduced_tensor = torch.tensor(X_reduced)
#
#         # Step 5: Split back into anchors and embeddings
#         anchors = X_reduced_tensor[:len(anchors)].to(anchors.device)
#         embedding = X_reduced_tensor[len(anchors):].to(anchors.device)
#
#         feature_x_C = []
#         for class_i in torch.unique(labels):
#             anchors_i = anchors[labels==class_i]
#             feature_x_C.append(torch.max(torch.mm(embedding, anchors_i.T), dim=1).values)
#
#         feature_x_C = torch.stack(feature_x_C, dim=1)
#         # feature_x_C = self.GMM_dist(embedding, embedding_labels, anchors, labels)
#         # # do softmax
#         # feature_x_C = F.softmax(feature_x_C, dim=1)
#         aver_dist = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
#         aver_q = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
#         mean_embedding = torch.mean(torch.stack([embedding[bs * i: bs * (i + 1)] for i in range(self.number_aug)]), dim=0)
#         for i, q_id in enumerate(range(self.number_aug)):
#             if no_grad == True:
#                 with torch.no_grad():
#                     out = feature_x_C[bs * q_id: bs * (q_id + 1)].detach()
#                     aver_dist += out
#                     # aver_var.append(torch.diag(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1))
#                     # get assignments
#                     q = distributed_sinkhorn(out) #[-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
#                     # pred_labels.append(q)
#                     aver_q += q
#             else:
#                 out = feature_x_C[bs * q_id: bs * (q_id + 1)]
#                 aver_dist += out
#                 # aver_var.append(torch.diag(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1))
#                 # get assignments
#                 q = distributed_sinkhorn(out)  # [-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
#                 # pred_labels.append(q)
#                 aver_q += q
#
#         #     subloss = 0
#         #     for v in np.delete(np.arange(np.sum(self.number_aug)), q_id):
#         #         x = feature_x_C[bs * v: bs * (v + 1)] / self.temperature
#         #         subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
#         #     loss += subloss / (np.sum(self.number_aug) - 1)
#         # loss /= self.number_q
#         # aver_label = sum(pred_labels[:self.number_aug]) / self.number_aug
#         aver_dist /= self.number_aug
#         # aver_var = torch.stack(aver_var, dim=0).var(dim=0)
#         aver_q /= self.number_aug
#
#         return aver_dist, aver_q
#

# class SWAVLoss(nn.Module):
#     """Modified from https://github.com/facebookresearch/swav"""
#
#     def __init__(self, number_aug=8, temperature=0.1):
#         super(SWAVLoss, self).__init__()
#         self.number_aug = number_aug
#         self.temperature = temperature
#
#     def forward(self, embedding, anchor, bs, no_grad=True):
#         feature_x_C = torch.mm(embedding, anchor.T)
#         # # # do softmax
#         # feature_x_C = F.softmax(feature_x_C, dim=1)
#         aver_dist = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
#         aver_q = torch.zeros([bs, feature_x_C.shape[1]]).to(feature_x_C.device)
#         mean_embedding = torch.mean(torch.stack([embedding[bs * i: bs * (i + 1)] for i in range(self.number_aug)]), dim=0)
#         # aver_var = []
#         for i, q_id in enumerate(range(self.number_aug)):
#             if no_grad == True:
#                 with torch.no_grad():
#                     out = feature_x_C[bs * q_id: bs * (q_id + 1)].detach()
#                     aver_dist += out
#                     # aver_var.append(torch.diag(F.softmax(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1)))
#                     # get assignments
#                     q = distributed_sinkhorn(out) #[-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
#                     # pred_labels.append(q)
#                     aver_q += q
#             else:
#                 out = feature_x_C[bs * q_id: bs * (q_id + 1)]
#                 aver_dist += out
#                 # aver_var.append(torch.diag(F.softmax(torch.mm(embedding[bs * q_id: bs * (q_id + 1)], mean_embedding.T), dim=1)))
#                 # get assignments
#                 q = distributed_sinkhorn(out)  # [-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
#                 # pred_labels.append(q)
#                 aver_q += q
#
#         #     subloss = 0
#         #     for v in np.delete(np.arange(np.sum(self.number_aug)), q_id):
#         #         x = feature_x_C[bs * v: bs * (v + 1)] / self.temperature
#         #         subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
#         #     loss += subloss / (np.sum(self.number_aug) - 1)
#         # loss /= self.number_q
#         # aver_label = sum(pred_labels[:self.number_aug]) / self.number_aug
#         aver_dist /= self.number_aug
#         # aver_var = torch.stack(aver_var, dim=0).var(dim=0)
#         aver_q /= self.number_aug
#
#         return aver_dist, aver_q

#
# class SWAVLoss(nn.Module):
#     """Modified from https://github.com/facebookresearch/swav"""
#
#     def __init__(self, number_aug=8, temperature=0.1):
#         super(SWAVLoss, self).__init__()
#         self.number_aug = number_aug
#         self.temperature = temperature
#
#     def forward(self, embedding, anchor, bs, no_grad=False):
#         # Calculate the feature similarity
#         feature_x_C = torch.mm(embedding, anchor.T)
#         # Apply softmax
#         feature_x_C = F.softmax(feature_x_C, dim=1)
#
#         # Pre-allocate tensors
#         aver_dist = torch.zeros(bs, feature_x_C.size(1), device=feature_x_C.device)
#         aver_q = torch.zeros(bs, feature_x_C.size(1), device=feature_x_C.device)
#
#         # Use context manager for gradient control
#         with (torch.no_grad() if no_grad else nullcontext()):
#             for q_id in range(self.number_aug):
#                 # Slice the feature similarity tensor
#                 out = feature_x_C[bs * q_id: bs * (q_id + 1)]
#
#                 # Update average distance
#                 aver_dist += out
#
#                 # Compute assignments using distributed Sinkhorn function
#                 q = distributed_sinkhorn(out)
#                 aver_q += q
#
#         # Average the distances and assignments
#         aver_dist /= self.number_aug
#         aver_q /= self.number_aug
#
#         return aver_dist, aver_q


class SimCLRLoss(nn.Module):
    """Modified from https://github.com/wvangansbeke/Unsupervised-Classification."""

    def __init__(self, temperature, reduction="mean"):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.size()
        assert n == 2
        mask = torch.eye(b, dtype=torch.float32).to(features.device)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        if self.reduction == "mean":
            loss = -((mask * log_prob).sum(1) / mask.sum(1)).mean()
        elif self.reduction == "none":
            loss = -((mask * log_prob).sum(1) / mask.sum(1))
        else:
            raise ValueError("The reduction must be mean or none!")

        return loss



class ContrastiveLoss(nn.Module):
    """Modified from https://github.com/wvangansbeke/Unsupervised-Classification."""

    def __init__(self, temperature, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features, feature_x_C, bs):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        q = distributed_sinkhorn(feature_x_C)
        margin_p, margin_n = 0.8, 0.8
        # labels = torch.cat([labels, labels])
        # max_values, max_indices = torch.ones_like(labels.unsqueeze(1)).float(), labels.unsqueeze(1)
        max_values, max_indices = torch.max(q, dim=1, keepdim=True)

        match_matrix = (max_indices == max_indices.T).float()
        match_matrix.fill_diagonal_(0)
        match_matrix = match_matrix[:bs].to(features.device)
        right_part = match_matrix[:,bs:]
        right_part.fill_diagonal_(1)
        match_matrix[:,bs:] = right_part

        unmatch_matrix = (max_indices != max_indices.T).float()
        unmatch_matrix = unmatch_matrix[:bs]
        right_part = unmatch_matrix[:,bs:]
        right_part.fill_diagonal_(0)
        unmatch_matrix[:,bs:] = right_part
        # filter_self_mask = torch.ones((features.shape[0], features.shape[0]))
        # filter_self_mask.fill_diagonal_(0)
        # filter_self_mask = filter_self_mask[:bs].to(features.device)
        #
        weight = torch.mm(max_values, max_values.T)[:bs].to(features.device)
        right_part = weight[:,bs:]
        right_part.fill_diagonal_(1)
        weight[:,bs:] = right_part

        # anchor = features[:bs]

        # Dot product
        dot_product = torch.matmul(features, features.T)[:bs]
        loss_p, loss_n = (weight * match_matrix * (torch.relu(margin_p - dot_product))).mean(), (weight * unmatch_matrix * (torch.relu(dot_product - margin_n))).mean()

        # dot_product = torch.cdist(features, features)[:bs] # / self.temperature
        # loss_p, loss_n = torch.relu(match_matrix * dot_product - margin_p).mean(), torch.relu(margin_n - unmatch_matrix * dot_product).mean()
        print('loss_p {} vs. loss_n {}'.format(loss_p.item(), loss_n.item()))
        loss = loss_p + loss_n
        # Log-sum trick for numerical stability
        # logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        # logits = dot_product - logits_max.detach()

        # Improved Deep Metric Learning with Multi-class N-pair Loss Objective
        # N-pair Loss
        # exp_logits = torch.exp(logits) * unmatch_matrix
        # log_prob = logits - torch.log(torch.exp(logits)+exp_logits.sum(1, keepdim=True))

        # if self.reduction == "mean":
        #     loss = -((match_matrix * log_prob).sum(1) / match_matrix.sum(1)).mean()
        # elif self.reduction == "none":
        #     loss = -((match_matrix * log_prob).sum(1) / match_matrix.sum(1))
        # else:
        #     raise ValueError("The reduction must be mean or none!")

        return loss


def tripletPair(labels):
    all_matches, all_diffs = get_matches_and_diffs(labels, labels)

    if (
            all_matches.shape[0] * all_matches.shape[1] * all_matches.shape[1]
            < torch.iinfo(torch.int32).max
    ):
        # torch.nonzero is not supported for tensors with more than INT_MAX elements
        return get_all_triplets_indices_vectorized_method(all_matches, all_diffs)

    return get_all_triplets_indices_loop_method(labels, all_matches, all_diffs)


class TripletLoss(nn.Module):
    """Modified from https://github.com/wvangansbeke/Unsupervised-Classification."""

    def __init__(self, temperature=1, reduction="mean"):
        super(TripletLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features, labels):
        margin = 0.2
        anchor, positive, negative = tripletPair(labels)
        sim = torch.matmul(features, features.T)
        sim_ap, sim_an = sim[anchor, positive], sim[anchor, negative]
        loss = torch.relu(sim_ap - sim_an + margin).mean()

        return loss

def get_loss_fun(method, parameter_dict):
    if method == 'swav':
        number_aug, temperature = 8, 0.1
        if 'number_aug' in parameter_dict.keys():
            number_aug = parameter_dict['number_aug']
        if 'temperature' in parameter_dict.keys():
            temperature = parameter_dict['temperature']
        lossfunc = SWAVLoss(number_aug, temperature)
    # elif method == 'swav_GMM':
    #     number_aug, temperature = 8, 0.1
    #     if 'number_aug' in parameter_dict.keys():
    #         number_aug = parameter_dict['number_aug']
    #     if 'temperature' in parameter_dict.keys():
    #         temperature = parameter_dict['temperature']
    #     lossfunc = SWAVGMMLoss(number_aug, temperature)
    elif method =='simclr':
        temperature, reduction = parameter_dict['temperature'], "mean"
        if 'reduction' in parameter_dict.keys():
            reduction = parameter_dict['reduction']
        lossfunc = SimCLRLoss(temperature, reduction)
    elif method == 'constrastice':
        temperature, reduction = parameter_dict['temperature'], "mean"
        if 'reduction' in parameter_dict.keys():
            reduction = parameter_dict['reduction']
        lossfunc = ContrastiveLoss(temperature, reduction)
    elif method == 'mixmatch':
        rampup_length, lambda_u = parameter_dict['rampup_length'], parameter_dict['lambda_u']
        lossfunc = MixMatchLoss(rampup_length, lambda_u)
    else:
        return None
    return lossfunc



def triplet_loss(anchors, high_confdience_data, high_confdience_labels):
    margin = 0.5
    dist_anchors_data = torch.mm(anchors, high_confdience_data.T)
    dist_triple = dist_anchors_data.unsqueeze(1) - dist_anchors_data.unsqueeze(2)
    labels_anchors = torch.tensor([i for i in range(len(anchors))]).to(anchors.device)
    label_match = (labels_anchors.unsqueeze(1) == high_confdience_labels.unsqueeze(0)).float()
    label_triple = label_match.unsqueeze(1) - label_match.unsqueeze(2)
    # for cosine similairty, negative pari - positive pair i.e. == -1
    indices_negative_positive = label_triple == -1
    if torch.sum(indices_negative_positive) == 0:
        return torch.tensor(0).to(anchors.device)
    loss = torch.relu(dist_triple[indices_negative_positive]+margin).mean()
    return loss


def discrimination_loss(anchors):
    dist_anchors = torch.mm(anchors, anchors.T)
    # Set the diagonal elements to zero
    dist_anchors.fill_diagonal_(0)
    loss = torch.max(dist_anchors)

    return loss


def mixMatchLoss(mixmatch_criterion, model, low_confidence_data, low_confdience_softmax, high_confdience_data, high_confidence_pseudolabels, num_classes, num_aug, device, batch_id):
    # Following the original code
    # for high confidence data take the one set of augmentation
    # for low confidence data taking the two set of augmentation
    # # TODO: replace the labels data with val data
    batch_x, batch_u = len(high_confdience_data[0]), len(low_confidence_data[0])
    batch_size = min(batch_x, batch_u) # TODO: replace the batch_x

    if len(low_confidence_data) >= 2:
        xinput, xtarget, uinput1, uinput2 = high_confdience_data[0][:batch_size], \
            high_confidence_pseudolabels[0][:batch_size].to(device), low_confidence_data[0][:batch_size], \
            low_confidence_data[1][:batch_size]
    else:
        xinput, xtarget, uinput1, uinput2 = high_confdience_data[0][:batch_size], \
            high_confidence_pseudolabels[0][:batch_size].to(device), low_confidence_data[0][:batch_size], \
            low_confidence_data[0][:batch_size]
    # if batch_x < batch_u:
    #     batch_size = batch_u
    #     repeat_num = int(batch_size/batch_x) + 1
    #     xinput, xtarget = high_confdience_data[0].repeat([repeat_num] + [1 for _ in range(len(high_confdience_data[0].shape) - 1)])[:batch_size], \
    #         high_confidence_pseudolabels[0].repeat([repeat_num] + [1 for _ in range(len(high_confidence_pseudolabels[0].shape) - 1)])[:batch_size]
    #     uinput1, uinput2 = low_confidence_data[0], low_confidence_data[1]
    # else:
    #     batch_size = batch_x
    #     repeat_num = int(batch_size / batch_u) + 1
    #     xinput, xtarget = high_confdience_data[0], \
    #         high_confidence_pseudolabels[0]
    #     uinput1, uinput2 = low_confidence_data[0].repeat([repeat_num] + [1 for _ in range(len(low_confidence_data[0].shape) - 1)])[:batch_size], \
    #         low_confidence_data[1].repeat([repeat_num] + [1 for _ in range(len(low_confidence_data[1].shape) - 1)])[:batch_size]

    xtarget_raw = torch.zeros(batch_size, num_classes).to(device)
    xtarget = xtarget_raw.scatter_(
        1, xtarget.view(-1, 1).long(), 1
    )
    xinput = xinput.to(device)
    xtarget = xtarget.to(device)
    uinput1 = uinput1.to(device)
    uinput2 = uinput2.to(device)

    with torch.no_grad():
        # compute guessed labels of unlabel samples
        uoutput1 = model(uinput1)
        uoutput2 = model(uinput2)
        # TODO: use the emebedding directly from previous steps
        # uoutput = model(torch.cat([uinput1, uinput2]))
        # uoutput1, uoutput2 = uoutput[:uinput1.shape[0]], uoutput[uinput1.shape[0]:]
        p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2

        # aver_softmax = torch.zeros((batch_size, num_classes)).to(device)
        # for i in range(num_aug):
        #     aver_softmax += torch.softmax(low_confdience_softmax[i][:batch_size].to(device), dim=1)
        #
        # p = aver_softmax / num_aug

        temperature = 0.5
        pt = p ** (1 / temperature)
        utarget = pt / pt.sum(dim=1, keepdim=True)
        utarget = utarget.detach()

    # mixup
    all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
    all_target = torch.cat([xtarget, utarget, utarget], dim=0)
    l = np.random.beta(0.75, 0.75)
    l = max(l, 1 - l)
    idx = torch.randperm(all_input.size(0))
    input_a, input_b = all_input, all_input[idx]
    target_a, target_b = all_target, all_target[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)

    # logit = [model(mixed_input[0])]
    # for input in mixed_input[1:]:
    #     logit.append(model(input))

    concatenated_input = torch.cat(mixed_input, dim=0)
    concatenated_logit = model(concatenated_input)
    logit = torch.chunk(concatenated_logit, len(mixed_input), dim=0)

    # put interleaved samples back
    logit = interleave(logit, batch_size)
    xlogit = logit[0]
    ulogit = torch.cat(logit[1:], dim=0)

    Lx, Lu, lambda_u = mixmatch_criterion(
        xlogit,
        mixed_target[:batch_size],
        ulogit,
        mixed_target[batch_size:],
        batch_id,
    )
    loss = Lx + lambda_u * Lu

    return loss
#
# def mixMatchLoss(mixmatch_criterion, model, low_confidence_data, low_confdience_embedding, high_confdience_data, high_confidence_pseudolabels, num_classes, num_aug, device, batch_id):
#     # Following the original code
#     # for high confidence data take the one set of augmentation
#     # for low confidence data taking the two set of augmentation
#     # TODO: replace the labels data with val data
#     batch_x, batch_u = len(high_confdience_data), len(low_confidence_data[0])
#
#     if batch_x < batch_u:
#         batch_size = batch_u
#         repeat_num = int(batch_u/batch_x) + 1
#         xinput, xtarget = high_confdience_data.repeat([repeat_num] + [1 for _ in range(len(high_confdience_data.shape) - 1)])[:batch_size], \
#             high_confidence_pseudolabels.repeat([repeat_num] + [1 for _ in range(len(high_confidence_pseudolabels.shape) - 1)])[:batch_size]
#         uinput1, uinput2 = low_confidence_data[0], low_confidence_data[1]
#
#     else:
#         batch_size = min(batch_x, batch_u)  # TODO: replace the batch_x
#         xinput, xtarget, uinput1, uinput2 = high_confdience_data[:batch_size], \
#             high_confidence_pseudolabels[:batch_size].to(device), low_confidence_data[0][:batch_size], \
#             low_confidence_data[1][:batch_size]
#
#
#     xtarget_raw = torch.zeros(batch_size, num_classes).to(device)
#     xtarget = xtarget_raw.scatter_(
#         1, xtarget.view(-1, 1).long(), 1
#     )
#     xinput = xinput.to(device)
#     xtarget = xtarget.to(device)
#     uinput1 = uinput1.to(device)
#     uinput2 = uinput2.to(device)
#
#     with torch.no_grad():
#         # compute guessed labels of unlabel samples
#         # uoutput1 = model(uinput1)
#         # uoutput2 = model(uinput2)
#         # TODO: use the emebedding directly from previous steps
#         aver_softmax = torch.zeros((batch_size, num_classes)).to(device)
#         for i in range(num_aug):
#             aver_softmax += torch.softmax(model.fc(low_confdience_embedding[i][:batch_size]).to(device), dim=1)
#
#         p = aver_softmax / num_aug
#
#         temperature = 0.5
#         pt = p ** (1 / temperature)
#         utarget = pt / pt.sum(dim=1, keepdim=True)
#         utarget = utarget.detach()
#
#     # mixup
#     all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
#     all_target = torch.cat([xtarget, utarget, utarget], dim=0)
#     l = np.random.beta(0.75, 0.75)
#     l = max(l, 1 - l)
#     idx = torch.randperm(all_input.size(0))
#     input_a, input_b = all_input, all_input[idx]
#     target_a, target_b = all_target, all_target[idx]
#     mixed_input = l * input_a + (1 - l) * input_b
#     mixed_target = l * target_a + (1 - l) * target_b
#
#     # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
#     mixed_input = list(torch.split(mixed_input, batch_size))
#     mixed_input = interleave(mixed_input, batch_size)
#
#     # logit = [model(mixed_input[0])]
#     # for input in mixed_input[1:]:
#     #     logit.append(model(input))
#
#     concatenated_input = torch.cat(mixed_input, dim=0)
#     concatenated_logit = model(concatenated_input)
#     logit = torch.chunk(concatenated_logit, len(mixed_input), dim=0)
#
#     # put interleaved samples back
#     logit = interleave(logit, batch_size)
#     xlogit = logit[0]
#     ulogit = torch.cat(logit[1:], dim=0)
#
#     Lx, Lu, lambda_u = mixmatch_criterion(
#         xlogit,
#         mixed_target[:batch_size],
#         ulogit,
#         mixed_target[batch_size:],
#         batch_id,
#     )
#     loss = Lx + lambda_u * Lu
#
#     return loss


### loss from https://github.com/microsoft/Semi-supervised-learning/tree/main ###
def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum() # \bar{h}

    # modulate prob model
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'kl']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()
### loss from https://github.com/microsoft/Semi-supervised-learning/tree/main ###


if __name__ == '__main__':
    feature = nn.functional.normalize(torch.randn([8, 4]), dim=1, p=2)
    C = nn.functional.normalize(torch.randn([2, 4]), dim=1, p=2)
    feature_x_c = torch.mm(feature, C.T)
    loss = SWAVLoss(number_aug=2)
    loss_v, label = loss(feature_x_c, 4)
    # labels = torch.randint(0, 2, (4, 1))
    # # feature = torch.randn(5, 2, 10)
    # # loss = SimCLRLoss(temperature=0.5)
    # loss = ContrastiveLoss(temperature=0.5)
    # loss_v = loss(feature, feature_x_c, 4, labels)
    # embeddings, labels = torch.rand(10, 128), torch.randint(0, 9, (10,1))
    # loss = TripletMarginLoss()
    # loss_v = loss(embeddings, labels)
    # features = nn.functional.normalize(torch.randn([10, 4]), dim=1, p=2)
    # labels = torch.randint(0, 4, (10, 1))[:,0]
    # loss = TripletLoss()
    # loss_v = loss(features, labels)