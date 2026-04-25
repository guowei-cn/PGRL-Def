import os
import random
import time

import math
import torch
import torch.nn as nn
import umap
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import mixture
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from lib.loss_func import get_loss_fun, mixMatchLoss
from lib.models import gen_model
from lib.dataLoader import get_validation_data, get_batch_val, get_dataset, pseudoLabeling_ori, get_ssldata, \
    ToTargetClass, \
    source_imagenet_name_adp, calculate_tpr_fpr_no_indices, get_dataset2
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn.functional as F

from train import evaluating, poison_online

ablation_dict = {'all': (1., 1.), 'ce': (1., 0.), 'mm': (0., 1.), # 'swav': (0., 0., 1.),
                 'non': (0., 0.)}

def set_seed(seed: int):
    # Set the seed for the built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)

    # If you are using a GPU, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior for certain functions
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage:
set_seed(19920313)

def drawHist(ax, measure, label, title_prefix):
    measures_benign = measure[label == 0].flatten()
    measures_poison = measure[label == 1].flatten()

    # Calculate histograms
    bins = np.histogram(measure, bins=200)[1]

    counts_benign, bins_benign = np.histogram(measures_benign, bins=bins)
    counts_poison, bins_poison = np.histogram(measures_poison, bins=bins)
    # Find the maximum y-value for setting consistent y-axis limits
    max_y = max(counts_benign.max(), counts_poison.max())

    # Plot histograms
    if len(measures_benign) > 0:
        ax.hist(measures_benign, bins=bins_benign, alpha=0.5, label='Benign', color='blue')
    if len(measures_poison) > 0:
        ax.hist(measures_poison, bins=bins_poison, alpha=0.5, label='Poison', color='red')

    ax.set_xlabel('Measure')
    ax.set_ylabel('Frequency')
    ax.set_title(title_prefix + ' Histogram')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, max_y)  # Set y-axis limit to be consistent




def statistics_analysis(aver_dist_l, q_l, untrusted_label_l, gt_poison_l, pd_poison_l, output_file_name):
    unique_labels = sorted(set(untrusted_label_l.tolist()))
    n_classes = len(unique_labels)

    # Create a grid of subplots (n_class * 2 columns)
    # fig, axes = plt.subplots(n_classes, 5, figsize=(24, 5 * n_classes), sharex='col', sharey='row')
    fig, axes = plt.subplots(n_classes, 4, figsize=(30, 5 * n_classes))
    for idx, class_id in enumerate(unique_labels):
        class_dist = aver_dist_l[untrusted_label_l == class_id, class_id]
        class_q = q_l[untrusted_label_l == class_id, class_id]
        class_gt = gt_poison_l[untrusted_label_l == class_id]
        class_pd = pd_poison_l[untrusted_label_l == class_id]

        # Plot histograms for the average distribution
        drawHist(axes[idx, 0], class_dist, class_gt, f'Class {class_id} - Average Dist')

        # Plot histograms for the q values
        drawHist(axes[idx, 1], class_q, class_gt, f'Class {class_id} - Q Values')

        # Plot histograms for the average distribution
        drawHist(axes[idx, 2], class_dist, class_pd, f'Class {class_id} - Average Dist')

        # Plot histograms for the q values
        drawHist(axes[idx, 3], class_q, class_pd, f'Class {class_id} - Q Values')

    plt.tight_layout()
    plt.savefig(output_file_name)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory


def cal_variance_emb(ulabel_embedding, bs, num_aug):
    """
    Calculate the variance of the cosine similarity across augmentations for each sample in the batch.

    Args:
        ulabel_embedding (Tensor): The input embeddings with shape [bs * num_aug, length].
        bs (int): Batch size.
        num_aug (int): Number of augmentations.

    Returns:
        variance (Tensor): Variance of cosine similarity for each sample in the batch, shape [bs].
    """
    # Reshape the embedding into [num_aug, bs, length]
    reshaped_embeddings = ulabel_embedding.view(num_aug, bs, -1)  # shape: [num_aug, bs, length]
    reshaped_embeddings_ = torch.stack([ulabel_embedding[i*bs:(i+1)*bs] for i in range(num_aug)])
    assert torch.equal(reshaped_embeddings, reshaped_embeddings_)
    # Calculate the average embedding for each sample across augmentations
    aver_emb = reshaped_embeddings.mean(dim=0)  # shape: [bs, length]

    # Initialize a list to store cosine similarities
    cosine_similarities = []

    # For each augmentation, calculate cosine similarity to the average embedding
    for i in range(num_aug):
        aug_emb = reshaped_embeddings[i]  # shape: [bs, length]

        # Compute cosine similarity (1 - cosine distance)
        cos_sim = F.cosine_similarity(aug_emb, aver_emb, dim=-1)  # shape: [bs]

        # Store the cosine similarities
        cosine_similarities.append(cos_sim)

    # Stack cosine similarities: shape [num_aug, bs]
    cosine_similarities = torch.stack(cosine_similarities, dim=0)

    # Compute variance of the cosine similarities across augmentations (along dim=0)
    variance = cosine_similarities.var(dim=0)  # shape: [bs]

    return variance


def plot_emb(class_emb, class_anchor, class_gt, idx, axes, label):
    # Separate benign and poisoned embeddings
    benign_emb = class_emb[class_gt == 0]
    poison_emb = class_emb[class_gt == 1]
    cosine_similarity = lambda A, B: np.dot(A.flatten(), B.flatten()) / (np.linalg.norm(A.flatten()) * np.linalg.norm(B.flatten()))

    # Compute the average embeddings for benign and poisoned
    class_anchor = class_anchor.reshape(8, 16)
    avg_benign_emb = np.mean(benign_emb, axis=0).reshape(8, 16)
    avg_poison_emb = np.mean(poison_emb, axis=0).reshape(8, 16) if poison_emb.size > 0 else None

    # Select three random embeddings for benign and poisoned
    random_benign_idxs = np.random.choice(benign_emb.shape[0], 3, replace=False)
    random_benign_embs = benign_emb[random_benign_idxs].reshape(3, 8, 16)

    random_poison_embs = None
    if poison_emb.size > 0:
        random_poison_idxs = np.random.choice(poison_emb.shape[0], 3, replace=False)
        random_poison_embs = poison_emb[random_poison_idxs].reshape(3, 8, 16)

    # Plot benign embeddings
    axes[idx, 0].imshow(class_anchor, cmap='gray')
    axes[idx, 0].set_title(f"Anchor Embedding")
    cs = cosine_similarity(class_anchor, avg_benign_emb)
    axes[idx, 1].imshow(avg_benign_emb, cmap='gray')
    axes[idx, 1].set_title("Class {} - Avg Benign Embedding {:.3f}".format(label, cs))

    for i in range(3):
        cs = cosine_similarity(class_anchor, random_benign_embs[i])
        axes[idx, i + 2].imshow(random_benign_embs[i], cmap='gray')
        axes[idx, i + 2].set_title("Class {} - Benign Emb {} - Sim - {:.3f}".format(label, i + 1, cs))

    # Plot poisoned embeddings if they exist
    if avg_poison_emb is not None:
        axes[idx + 1, 0].imshow(class_anchor, cmap='gray')
        axes[idx + 1, 0].set_title(f"Anchor Embedding")
        cs = cosine_similarity(class_anchor, avg_poison_emb)
        axes[idx + 1, 1].imshow(avg_poison_emb, cmap='gray')
        axes[idx + 1, 1].set_title("Class {} - Avg Poison Embedding {:.3f}".format(label, cs))

        for i in range(3):
            cs = cosine_similarity(class_anchor, random_poison_embs[i])
            axes[idx + 1, i + 2].imshow(random_poison_embs[i], cmap='gray')
            axes[idx + 1, i + 2].set_title("Class {} - Poison Emb {} - Sim {:.3f}".format(label, i+1, cs))


def emb_vis(emb_l, anchor, untrusted_label_l, gt_poison_l, pd_poison_l, number_aug, save_name):
    untrusted_label_l = untrusted_label_l[pd_poison_l==0]
    gt_poison_l = gt_poison_l[pd_poison_l==0]
    # pd_poison_l = pd_poison_l[pd_poison_l==0]
    emb_l = emb_l[pd_poison_l==0]
    unique_labels = sorted(set(untrusted_label_l.tolist()))
    n_classes = len(unique_labels)

    # Create a grid of subplots (n_classes rows, 4 columns; one additional row for poisoned data)
    fig, axes = plt.subplots(n_classes + 1, 5, figsize=(24, 5 * (n_classes + 1)), sharex='col', sharey='row')

    idx_offset = 0
    for idx, class_id in enumerate(unique_labels):
        class_anchor = anchor[class_id]
        class_emb = emb_l[untrusted_label_l == class_id]
        class_gt = gt_poison_l[untrusted_label_l == class_id]

        plot_emb(class_emb, class_anchor, class_gt, idx + idx_offset, axes, class_id)

        # Check if the current class contains poisoned samples
        if np.any(class_gt == 1):
            idx_offset += 1  # Add an additional row for the poisoned data

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close(fig)


def anchor_vis(anchor, save_name):
    anchor = (anchor - anchor.min()) / (anchor.max() - anchor.min())
    num_class = anchor.shape[0]
    anchor_length = anchor.shape[1]

    # Check if the anchor length is divisible by 16 to match the desired image width
    if anchor_length % 16 != 0:
        raise ValueError("The length of the anchor must be divisible by 16 for proper visualization.")

    # Create subplots
    fig, axes = plt.subplots(num_class, 1, figsize=(16, num_class * 2))  # Adjust figsize as needed

    # Make sure axes is always a 2D array even if num_class is 1
    if num_class == 1:
        axes = np.array([axes])

    for i in range(num_class):
        # Extract row for the current class
        row = anchor[i, :].reshape((8, 16))  # Reshape to (8, 16)

        # Plot the sub-image
        ax = axes[i]
        ax.imshow(row, cmap='gray', vmin=np.min(anchor), vmax=np.max(anchor))
        ax.set_title(f'Class {i}')
        ax.axis('off')  # Hide the axes

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig(save_name)
    plt.close()


def draw_dist_loss_pair(aver_dist_l, untrusted_label_l, aver_loss_untrusted_l, savefile, ylabel):
    # Create a scatter plot
    aver_dist_l_target_class = [dist[label] for dist, label in zip(aver_dist_l, untrusted_label_l)]
    plt.scatter(aver_loss_untrusted_l, aver_dist_l_target_class, color='blue', label='Dist vs Loss')

    # Label the axes
    plt.xlabel('loss')
    plt.ylabel(ylabel)

    # Add a title (optional)
    plt.title('Correlation between Loss and Distance')

    # Add grid for better readability
    plt.grid(True)

    # Add a legend
    plt.legend(loc='upper right')

    # Save the figure using the savefile parameter
    plt.savefig(savefile)

    # Close the plot to free up memory
    plt.close()


def draw_dist_q_pair(aver_dist_l, q_l, untrusted_label_l, savefile):
    # Create a scatter plot
    aver_dist_l_target_class = [dist[label] for dist, label in zip(aver_dist_l, untrusted_label_l)]
    aver_q_l_target_class = [q[label] for q, label in zip(q_l, untrusted_label_l)]
    plt.scatter(aver_q_l_target_class, aver_dist_l_target_class, color='blue', label='q vs dist')

    # Label the axes
    plt.xlabel('q')
    plt.ylabel('cos sim')

    # Add a title (optional)
    plt.title('Correlation between Loss and Distance')

    # Add grid for better readability
    plt.grid(True)

    # Add a legend
    plt.legend(loc='upper right')

    # Save the figure using the savefile parameter
    plt.savefig(savefile)

    # Close the plot to free up memory
    plt.close()


def split_val(l_data, l_labels, indices_val):
    anchors_data, anchors_label, weight_data, weight_label = [], [], [], []

    for class_i in torch.unique(l_labels):
        # Filter data and labels by class
        l_data_i = l_data[l_labels == class_i]
        l_labels_i = l_labels[l_labels == class_i]

        # Take all validate data
        anchors_data.append(l_data_i)
        anchors_label.append(l_labels_i)

        # Append all data to weight_data without excluding any index
        weight_data.append(l_data_i)
        weight_label.append(l_labels_i)

    # Concatenate results into single tensors
    anchors_data = torch.cat(anchors_data)
    anchors_label = torch.cat(anchors_label)
    weight_data = torch.cat(weight_data)
    weight_label = torch.cat(weight_label)

    return anchors_data, anchors_label, weight_data, weight_label

# def split_val(l_data, l_labels, indices_val):
#     anchors_data, anchors_label, weight_data, weight_label = [], [], [], []
#     for class_i in torch.unique(l_labels):
#         l_data_i = l_data[l_labels == class_i]
#         l_labels_i = l_labels[l_labels == class_i]
#         indices_val_i = indices_val[l_labels == class_i]
#         min_indices_val_i = min(indices_val_i)
#         anchors_data.append(l_data_i[indices_val_i==min_indices_val_i])
#         anchors_label.append(l_labels_i[indices_val_i==min_indices_val_i])
#         if len(l_data_i) == 1:
#             weight_data.append(l_data_i[indices_val_i==min_indices_val_i])
#             weight_label.append(l_labels_i[indices_val_i==min_indices_val_i])
#         else:
#             weight_data.append(l_data_i[indices_val_i!=min_indices_val_i])
#             weight_label.append(l_labels_i[indices_val_i!=min_indices_val_i])
#
#     anchors_data, anchors_label, weight_data, weight_label = torch.cat(anchors_data),\
#         torch.cat(anchors_label), torch.cat(weight_data), torch.cat(weight_label)
#
#     return anchors_data, anchors_label, weight_data, weight_label


def query_weight(weights, weights_indices, high_indices, poison_flags):
    weights, weights_indices = weights.to(high_indices.device), weights_indices.to(high_indices.device)

    # high_weights = []
    # for i in high_indices:
    #     high_weights.append(weights[weights_indices == i])
    # high_weights = torch.cat(high_weights)

    mask = torch.isin(weights_indices, high_indices)
    high_weights_ = weights[mask]
    # assert torch.equal(high_weights, high_weights_)

    # calculate auc
    if torch.sum(poison_flags) == 0:
        auc_weight = torch.tensor(-1)
    else:
        auc_weight = roc_auc_score(poison_flags.cpu().detach().numpy(), -high_weights_.cpu().detach().numpy())

    return high_weights_, auc_weight


def update_weights(old_weights, old_weights_indices, untrusted_act_l, untrusted_label_l, indices_l, gt_poison_l, weight_act_l, val_weight_est_lables,
                   save_name, threshold_percent):
    num_classes = len(np.unique(untrusted_label_l))

    # Two subplots: one row for weight distribution, one row for 2D distribution (with anchors)
    fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 5, 10))

    if num_classes == 1:
        axes = [axes]  # Handle the single class case

    weight_l, weight_indices_l = [], []
    pdf_all_classes = []  # Store all pdf_class_i values across classes
    reduced_2d_all_classes = []  # Store 2D reduced samples for visualization
    weight_2d_all_classes = []  # Store the reduced 2D anchor points for visualization

    for i in np.unique(untrusted_label_l):
        weight_act_l_i = weight_act_l[val_weight_est_lables == i]  # Anchor points
        untrusted_act_l_i = untrusted_act_l[untrusted_label_l == i]  # Untrusted samples
        indices_l_i = indices_l[untrusted_label_l == i]
        gt_poison_l_i = gt_poison_l[untrusted_label_l == i]

        concatenated_data = np.concatenate((weight_act_l_i, untrusted_act_l_i), axis=0)

        # Step 1: Reduce dimensionality to 10D for pdf calculation
        umap_model_10d = umap.UMAP(n_components=10, n_jobs=-1) #, random_state=42)
        reduced_data_10d = umap_model_10d.fit_transform(concatenated_data)
        weight_act_l_i_reduced, untrusted_act_l_i_reduced = reduced_data_10d[:len(weight_act_l_i)], reduced_data_10d[
                                                                                                    len(weight_act_l_i):]

        # Step 2: Calculate the PDF values
        pdf_class_i = []
        for weight_act_i_j in weight_act_l_i_reduced:
            mvn = multivariate_normal(mean=weight_act_i_j, cov=np.eye(len(weight_act_i_j)))
            log_prob = mvn.logpdf(untrusted_act_l_i_reduced)
            pdf = np.exp(log_prob)
            pdf_class_i.append(pdf)

        pdf_class_i = np.stack(pdf_class_i, axis=1)
        pdf_class_i = np.max(pdf_class_i, axis=1)

        # Append the current class's pdf_class_i to the list of all classes
        pdf_all_classes.append(pdf_class_i)
        weight_indices_l.append(indices_l_i)

        # Step 3: Apply UMAP to reduce to 2D for visualization purposes
        umap_model_2d = umap.UMAP(n_components=2, n_jobs=-1) #, random_state=42)
        reduced_data_2d = umap_model_2d.fit_transform(np.concatenate([weight_act_l_i, untrusted_act_l_i], axis=0))
        weight_act_l_i_2d, untrusted_act_l_i_2d = reduced_data_2d[:len(weight_act_l_i)], reduced_data_2d[
                                                                                         len(weight_act_l_i):]

        reduced_2d_all_classes.append(untrusted_act_l_i_2d)  # Append only the untrusted data part
        weight_2d_all_classes.append(weight_act_l_i_2d)  # Append the anchor points for this class

    # Concatenate all pdf_class_i arrays for normalization over all classes
    pdf_all_classes = np.concatenate(pdf_all_classes)

    # Normalize across all classes
    sorted_pdf = np.sort(pdf_all_classes)[::-1]
    n = sorted_pdf.size
    index_80_percent = int(threshold_percent * n) # - 1  #  normalize pdf lower than threshold P(pdf<index_80_percent)=0.9 others set as 1
    top_80_percent_cutoff = sorted_pdf[index_80_percent]
    median_value = top_80_percent_cutoff

    # Create mask for values below the cutoff and normalize
    bottom_half_mask = pdf_all_classes <= median_value
    bottom_half_values = pdf_all_classes[bottom_half_mask]
    epsilon = 1e-10  # small constant to avoid division by zero
    bottom_half_normalized = 2 * ((bottom_half_values - np.min(bottom_half_values)) / (
            np.max(bottom_half_values) - np.min(bottom_half_values) + epsilon)) - 1

    # bottom_half_normalized[bottom_half_normalized < 1e-3] = - 0.5  # set the value close to 0 as -0.1

    # Assign 1 to the top 20% and normalize the bottom 80%
    pdf_all_classes[~bottom_half_mask] = 1
    pdf_all_classes[bottom_half_mask] = bottom_half_normalized

    # Update weights with momentum
    weight_indices_l_np_cat = np.concatenate(weight_indices_l)
    momentum_alpha = 0.5
    old_weights, old_weights_indices = old_weights.cpu().detach().numpy(), old_weights_indices.cpu().detach().numpy()
    for i, idx in enumerate(weight_indices_l_np_cat):
        assert idx in old_weights_indices, 'no idx in old weight indices'
        pdf_all_classes[i] = momentum_alpha * pdf_all_classes[i] + (1 - momentum_alpha) * old_weights[list(old_weights_indices).index(idx)]

    # Now redistribute the normalized pdf values back to their respective classes
    start_idx = 0
    for i, class_i in enumerate(np.unique(untrusted_label_l)):
        indices_l_i = weight_indices_l[i - np.min(untrusted_label_l)]  # Get the indices for this class
        num_data_points = len(indices_l_i)

        # Extract normalized values for this class
        pdf_class_i = pdf_all_classes[start_idx:start_idx + num_data_points]
        start_idx += num_data_points

        weight_l.append(pdf_class_i)

        poisoned_pdf = pdf_class_i[gt_poison_l[untrusted_label_l == class_i] == 1]
        benign_pdf = pdf_class_i[gt_poison_l[untrusted_label_l == class_i] == 0]

        # Plot histogram for normalized weights (for each class)
        ax_weights = axes[0, i]
        if len(poisoned_pdf) > 0:
            ax_weights.hist(poisoned_pdf, bins=30, range=(-1, 1), color='red', alpha=0.7, label='Poisoned')
        ax_weights.hist(benign_pdf, bins=30, range=(-1, 1), color='blue', alpha=0.7, label='Benign')
        ax_weights.set_title(f'Weight Distribution - Class {class_i}')
        ax_weights.set_xlabel('Weight')
        ax_weights.set_ylabel('Frequency')
        ax_weights.legend()

        # Plot 2D distribution of reduced samples and anchor points
        ax_2d = axes[1, i]
        reduced_2d_data_i = reduced_2d_all_classes[i]  # Get the 2D reduced samples for the current class
        reduced_2d_poison = reduced_2d_data_i[gt_poison_l[untrusted_label_l == class_i] == 1]
        reduced_2d_benign = reduced_2d_data_i[gt_poison_l[untrusted_label_l == class_i] == 0]
        weight_2d_i = weight_2d_all_classes[i]  # Get the 2D reduced anchor points for the current class

        if len(reduced_2d_poison) > 0:
            ax_2d.scatter(reduced_2d_poison[:, 0], reduced_2d_poison[:, 1], color='red', label='Poisoned', alpha=0.6)
        ax_2d.scatter(reduced_2d_benign[:, 0], reduced_2d_benign[:, 1], color='blue', label='Benign', alpha=0.6)
        ax_2d.scatter(weight_2d_i[:, 0], weight_2d_i[:, 1], color='grey', label='Anchor', marker='x', s=100, alpha=1)

        ax_2d.set_title(f'2D UMAP Distribution - Class {class_i}')
        ax_2d.set_xlabel('UMAP Dim 1')
        ax_2d.set_ylabel('UMAP Dim 2')
        ax_2d.legend()

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

    # Concatenate weights and indices and reorder based on indices
    weight_l = np.concatenate(weight_l)
    weight_indices_l = np.concatenate(weight_indices_l)
    sorted_order = np.argsort(weight_indices_l)

    sorted_weight_indices_l = weight_indices_l[sorted_order]
    sorted_weight_l = weight_l[sorted_order]
    # momuntom

    return sorted_weight_l, sorted_weight_indices_l

#
# def update_weights(untrusted_act_l, untrusted_label_l, indices_l, gt_poison_l, weight_act_l, val_weight_est_lables, save_name):
#     num_classes = len(np.unique(untrusted_label_l))
#     fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))  # One subplot per class
#
#     if num_classes == 1:
#         axes = [axes]  # Handle the single class case
#
#     weight_l, weight_indices_l = [], []
#     for i in np.unique(untrusted_label_l):
#         weight_act_l_i = weight_act_l[val_weight_est_lables==i]
#         untrusted_act_l_i = untrusted_act_l[untrusted_label_l==i]
#         indices_l_i = indices_l[untrusted_label_l==i]
#         gt_poison_l_i = gt_poison_l[untrusted_label_l==i]
#         concatenated_data = np.concatenate((weight_act_l_i, untrusted_act_l_i), axis=0)
#         umap_model = umap.UMAP(n_components=10, random_state=42)
#         reduced_data = umap_model.fit_transform(concatenated_data)
#         weight_act_l_i_reduced, untrusted_act_l_i_reduced = reduced_data[:len(weight_act_l_i)], reduced_data[len(weight_act_l_i):]
#
#         # visulaization the distrbution
#
#         pdf_class_i = []
#         for weight_act_i_j in weight_act_l_i_reduced:
#             # Step 1: Create the Multivariate Normal distribution for the current anchor
#             mvn = multivariate_normal(mean=weight_act_i_j, cov=np.eye(len(weight_act_i_j)))
#
#             # Step 2: Calculate the log probability for data_i and exponentiate to get the PDF
#             log_prob = mvn.logpdf(untrusted_act_l_i_reduced)  # Log probability
#             pdf = np.exp(log_prob)  # Exponentiate to get the PDF
#             pdf_class_i.append(pdf)  # Append the PDF for this anchor
#
#         # Step 3: Stack the PDF values and find the maximum along the second dimension (across anchors)
#         pdf_class_i = np.stack(pdf_class_i, axis=1)  # Stack PDF values along a new dimension (axis 1)
#         pdf_class_i = np.max(pdf_class_i, axis=1)
#         # normalization
#         sorted_pdf = np.sort(pdf_class_i)[::-1]
#         n = sorted_pdf.size
#         index_80_percent = int(0.8 * n) -1   # This index marks the start of the largest 80%
#
#         # Step 3: Get the cutoff value for the largest 80%
#         top_80_percent_cutoff = sorted_pdf[index_80_percent]
#
#         median_value = top_80_percent_cutoff
#         # Step 2: Create a boolean mask for the bottom half (<= median)
#         bottom_half_mask = pdf_class_i <= median_value
#         # Step 3: Normalize the bottom half between 0 and 1
#         bottom_half_values = pdf_class_i[bottom_half_mask]
#         bottom_half_normalized = (bottom_half_values - np.min(bottom_half_values)) / (
#                     np.max(bottom_half_values) - np.min(bottom_half_values))
#         # Step 4: Assign 1 to the top half
#         pdf_class_i[~bottom_half_mask] = 1
#         # Step 5: Replace the bottom half with the normalized values
#         pdf_class_i[bottom_half_mask] = bottom_half_normalized
#
#         weight_l.append(pdf_class_i)
#         weight_indices_l.append(indices_l_i)
#
#         poisoned_pdf = pdf_class_i[gt_poison_l_i==1]
#         benign_pdf = pdf_class_i[gt_poison_l_i==0]
#         # draw subfig
#         ax = axes[i]
#         if len(poisoned_pdf) > 0:
#             ax.hist(poisoned_pdf, bins=30, range=(0, 1), color='red', alpha=0.7, label='Poisoned')
#
#         ax.hist(benign_pdf, bins=30, range=(0, 1), color='blue', alpha=0.7, label='Benign')
#
#         # Add title, labels, and legend
#         ax.set_title(f'PDF Distribution - Class {i}')
#         ax.set_xlabel('PDF')
#         ax.set_ylabel('Frequency')
#         ax.legend()
#
#
#     plt.tight_layout()
#     plt.savefig(save_name)
#     plt.close()
#
#     weight_l, weight_indices_l = np.concatenate(weight_l), np.concatenate(weight_indices_l)
#     sorted_order = np.argsort(weight_indices_l)
#
#     # Step 2: Reorder both indices_l and weight_l based on the sorted order
#     sorted_weight_indices_l = weight_indices_l[sorted_order]
#     sorted_weight_l = weight_l[sorted_order]
#
#     return sorted_weight_l, sorted_weight_indices_l


# def update_weights(untrusted_act_l, untrusted_label_l, indices_l, gt_poison_l, weight_act_l, val_weight_est_lables,
#                    save_name):
#     num_classes = len(np.unique(untrusted_label_l))
#     fig, axes = plt.subplots(2, num_classes,
#                              figsize=(num_classes * 5, 10))  # Two rows: one for histograms, one for scatter
#
#     if num_classes == 1:
#         axes = [axes]  # Handle single class case
#
#     weight_l, weight_indices_l = [], []
#
#     for i in np.unique(untrusted_label_l):
#         weight_act_l_i = weight_act_l[val_weight_est_lables == i]
#         untrusted_act_l_i = untrusted_act_l[untrusted_label_l == i]
#         indices_l_i = indices_l[untrusted_label_l == i]
#         gt_poison_l_i = gt_poison_l[untrusted_label_l == i]
#
#         concatenated_data = np.concatenate((weight_act_l_i, untrusted_act_l_i), axis=0)
#         umap_model = umap.UMAP(n_components=2, random_state=42)
#         reduced_data = umap_model.fit_transform(concatenated_data)
#         weight_act_l_i_reduced, untrusted_act_l_i_reduced = reduced_data[:len(weight_act_l_i)], reduced_data[
#                                                                                                 len(weight_act_l_i):]
#
#         # Step 1: PDF Distribution Estimation
#         pdf_class_i = []
#         for weight_act_i_j in weight_act_l_i_reduced:
#             mvn = multivariate_normal(mean=weight_act_i_j, cov=np.eye(len(weight_act_i_j)))
#             log_prob = mvn.logpdf(untrusted_act_l_i_reduced)
#             pdf = np.exp(log_prob)
#             pdf_class_i.append(pdf)
#
#         pdf_class_i = np.stack(pdf_class_i, axis=1)
#         pdf_class_i = np.max(pdf_class_i, axis=1)
#
#         # Normalization (Top 80% cutoff for PDF)
#         sorted_pdf = np.sort(pdf_class_i)[::-1]
#         n = sorted_pdf.size
#         index_80_percent = int(0.1 * n) - 1
#         top_80_percent_cutoff = sorted_pdf[index_80_percent]
#         median_value = top_80_percent_cutoff
#
#         bottom_half_mask = pdf_class_i <= median_value
#         bottom_half_values = pdf_class_i[bottom_half_mask]
#         bottom_half_normalized = (bottom_half_values - np.min(bottom_half_values)) / (
#                     np.max(bottom_half_values) - np.min(bottom_half_values))
#         pdf_class_i[~bottom_half_mask] = 1
#         pdf_class_i[bottom_half_mask] = bottom_half_normalized
#
#         weight_l.append(pdf_class_i)
#         weight_indices_l.append(indices_l_i)
#
#         poisoned_pdf = pdf_class_i[gt_poison_l_i == 1]
#         benign_pdf = pdf_class_i[gt_poison_l_i == 0]
#
#         # Draw weight histogram
#         ax_hist = axes[0, i] if num_classes > 1 else axes[0]
#         if len(poisoned_pdf) > 0:
#             ax_hist.hist(poisoned_pdf, bins=30, range=(0, 1), color='red', alpha=0.7, label='Poisoned')
#         ax_hist.hist(benign_pdf, bins=30, range=(0, 1), color='blue', alpha=0.7, label='Benign')
#         ax_hist.set_title(f'PDF Distribution - Class {i}')
#         ax_hist.set_xlabel('PDF')
#         ax_hist.set_ylabel('Frequency')
#         ax_hist.legend()
#
#         # Draw reduced 2D point distribution (scatter plot)
#         ax_scatter = axes[1, i] if num_classes > 1 else axes[1]
#         data_2d_poison = untrusted_act_l_i_reduced[gt_poison_l_i == 1]
#         data_2d_benign = untrusted_act_l_i_reduced[gt_poison_l_i == 0]
#
#         if len(data_2d_poison) > 0:
#             ax_scatter.scatter(data_2d_poison[:, 0], data_2d_poison[:, 1], color='red', label='Poisoned', alpha=0.6)
#         ax_scatter.scatter(data_2d_benign[:, 0], data_2d_benign[:, 1], color='blue', label='Benign', alpha=0.6)
#         ax_scatter.scatter(weight_act_l_i_reduced[:, 0], weight_act_l_i_reduced[:, 1], color='grey', label='Anchor',
#                            alpha=1, marker='x', s=100)
#
#         ax_scatter.set_title(f'2D Distribution - Class {i}')
#         ax_scatter.set_xlabel('UMAP Dim 1')
#         ax_scatter.set_ylabel('UMAP Dim 2')
#         ax_scatter.legend()
#
#     plt.tight_layout()
#     plt.savefig(save_name)
#     plt.close()
#
#     weight_l, weight_indices_l = np.concatenate(weight_l), np.concatenate(weight_indices_l)
#     sorted_order = np.argsort(weight_indices_l)
#     sorted_weight_indices_l = weight_indices_l[sorted_order]
#     sorted_weight_l = weight_l[sorted_order]
#
#     return sorted_weight_l, sorted_weight_indices_l
#

def calculate_auc_weight_ds(weights, weights_indices, ssl_dl):
    benign_indics = ssl_dl.dataset.dataset.benign_indics
    gt_poison_l = [0 if i in benign_indics else 1 for i in weights_indices]
    # calculate auc
    auc_weight_ds = roc_auc_score(np.array(gt_poison_l), -weights)

    return auc_weight_ds


def ssl_training_weight(epoch_id, weights, weights_indices, poison_type, poison_ratio, model, ssl_dl, val_dl, criterion,
                        optimizer, batch_id, device, warmup_epoch, xloss, num_sample, threshold_percent, model_name, writer=None,
                        net_G=None, normalization=None, aug=None, vis_repeat=20):
    lambda1, lambda2 = ablation_dict[xloss]
    model.train()
    gt_poison_l, pd_poison_l, aver_dist_l, q_l, untrusted_label_l, poison_indices_l = [], [], [], [], [], []
    untrusted_act_l, indices_l = [], []

    for b_id, batch in tqdm(enumerate(ssl_dl), total=len(ssl_dl)):
        batch_d, untrusted_label, poison_flags, indices = batch[:-3], batch[-3], batch[-2], batch[-1]
        # poison if net_G != None
        if net_G != None:
            for i in range(len(batch_d)):
                batch_d[i][poison_flags==True] = poison_online(net_G, batch_d[i][poison_flags==True].to(device)).to('cpu')
                # no need to modify the label since it is a clean-label attack
        if aug != None:
            batch_d = [aug(data) for data in batch_d]
        if normalization != None:
            batch_d = [normalization(data) for data in batch_d]
        # reorder
        sorted_indices = indices.argsort()
        reordered_batch_d = [b[sorted_indices] for b in batch_d]
        reordered_untrusted_label = untrusted_label[sorted_indices]
        reordered_poison_flags = poison_flags[sorted_indices]
        reordered_indices = indices[sorted_indices]
        batch_d, untrusted_label, poison_flags, indices = reordered_batch_d, reordered_untrusted_label.to(device), \
            reordered_poison_flags.to(device), reordered_indices.to(device)

        bs = len(batch[0])
        # if bs < ssl_dl.batch_size:
        #     continue

        untrusted_label_l.append(untrusted_label)
        indices_l.append(indices)

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr_schedule[batch_id]
        optimizer.zero_grad()
        batch_val = get_batch_val(val_dl)
        l_data, l_labels, indices_val = batch_val[0], batch_val[1], batch_val[3]


        if batch_id < warmup_epoch * len(ssl_dl): # warm-up
            loss_swav, loss_ad, loss_mm, auc_weight = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            _, outputs = model(l_data.to(device), feature_flag=True)
            loss_ce = F.cross_entropy(outputs, l_labels.to(device), reduction='sum')
        else:
            val_anchor, val_anchor_labels, val_weight_est, val_weight_est_lables = split_val(l_data, l_labels, indices_val)  # split the data into two parts
            embedding, outputs = model(torch.cat([val_anchor] + [val_weight_est] + batch_d).to(device), feature_flag=True)
            act = model.get_act()
            # obtain the activation of weight_val, and untrusted data
            weight_act_l = act[len(val_anchor):len(val_anchor)+len(val_weight_est)]
            act_ = act[len(val_anchor)+len(val_weight_est):len(val_anchor)+len(val_weight_est)+bs] # weight_act_l and act_ are used to calcualte the weights later
            untrusted_act_l.append(act_)

            # low_confdience_softmax = outputs[len(l_labels):]
            # low_confdience_softmax_num_aug = [low_confdience_softmax[i * bs:(i + 1) * bs] for i in range(ssl_dl.dataset.dataset.number_aug)]

            # CE of trusted data
            loss_ce = F.cross_entropy(outputs[:len(val_anchor)+len(val_weight_est)], torch.cat([val_anchor_labels, val_weight_est_lables]).to(device), reduction='none') # ce for known data

            # # used to visiualization
            # loss_untrusted = F.cross_entropy(outputs[len(l_labels):], torch.cat([untrusted_label for _ in range(ssl_dl.dataset.dataset.number_aug)]), reduction='none')
            # aver_loss_untrusted = torch.mean(torch.stack([loss_untrusted[i * bs:(i + 1) * bs] for i in range(ssl_dl.dataset.dataset.number_aug)]), dim=0)
            # aver_loss_untrusted_l.append(aver_loss_untrusted)

            # get the anchor for each class with the order from class 0 to 9 via mean
            anchor = torch.stack([torch.mean(embedding[:len(val_anchor)][val_anchor_labels == c], dim=0)
                             for c in range(len(val_dl.dataset.dataset.classes))])
            anchor = nn.functional.normalize(anchor, dim=1, p=2)

            # swap representation
            ulabeled_embedding = embedding[len(val_anchor)+len(val_weight_est):]
            aver_dist, q = criterion(ulabeled_embedding, anchor, bs)
            aver_dist_l.append(aver_dist)
            q_l.append(q)

            # # calculate the variance
            # var_emb = cal_variance_emb(ulabeled_embedding, bs, ssl_dl.dataset.dataset.number_aug)
            # var_emb_l.append(var_emb)

            weights_query, auc_weight = query_weight(weights, weights_indices, indices, poison_flags)


            # pseduo labeling when predicted label=untrusted label
            high_indices, weights_query_high, _, _, high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, low_confidence_data, low_confdience_embedding, \
                gt_poison, pd_poison, trusted_poison_flag = pseudoLabeling_ori(indices, weights_query, None, None, None, untrusted_label, poison_flags,
                                          batch_d, ulabeled_embedding, q, None, # carfully to the loss value should be -loss as input
                                          ssl_dl, bs, device)
            # loss_l.append(batch_loss)
            gt_poison_l += gt_poison
            pd_poison_l += pd_poison

            # increase the anchor discrimination via high confidence data and triplet loss
            # loss_ad = discrimination_loss(anchor.to(device)).to(device)
            # loss_ad = triplet_loss(anchor.to(device), torch.cat(high_confdience_embedding).to(device), torch.cat(high_confdience_pseudolabels).to(device))
            # high confidence data to fine-tune the fc layer
            # losses_high = F.cross_entropy(model.fc(torch.cat(high_confdience_embedding).to(device)),
            #                 torch.cat(high_confdience_pseudolabels).to(device), reduction='none')
            # weights = (high_confdience_var_emb - high_confdience_var_emb.min()) / (high_confdience_var_emb.max() - high_confdience_var_emb.min())
            # weights = weights.repeat([ssl_dl.dataset.dataset.number_aug] + [1 for i in range(len(weights.shape) - 1)])
            # weighted_losses = losses_high * weights
            #
            # # Step 3: Compute the final weighted loss (e.g., sum or mean)
            # final_loss = weighted_losses.mean()

            high_confidence_output = model.fc(torch.cat(high_confdience_embedding).to(device))
            # only unlearn the match pair for the samples with negative weights
            un_match_flag = torch.argmax(high_confidence_output, dim=1) != torch.cat(high_confdience_pseudolabels).to(device)

            weights_query_high = torch.cat([weights_query_high for _ in range(ssl_dl.dataset.dataset.number_aug)])  # repeat num_aug

            weight_high_neg_mask = weights_query_high < 0

            # if the sample has a negative weight and its prediction is already unmatch, set weight as 0 (not unlearn it any more)
            weights_query_high[weight_high_neg_mask & un_match_flag] = 0

            losses_high = F.cross_entropy(high_confidence_output, torch.cat(high_confdience_pseudolabels).to(device),
                                          reduction='none')
            # loss_ce += lambda1 * torch.mean(losses_high * weight_high)
            loss_ce = torch.sum(torch.cat([loss_ce, losses_high * weights_query_high]))

        loss =loss_ce

        loss.backward()

        optimizer.step()
        batch_id += 1
        writer.add_scalar('tra/loss_v', loss.item(), batch_id)
        writer.add_scalar('tra/loss_ce', loss_ce.item(), batch_id)
        writer.add_scalar('tra/auc_weight', auc_weight.item(), batch_id)
        # break # TODO

    if gt_poison_l == [] and pd_poison_l == []:
        pass
    else:
        print('print TPR, FPR, ACC')
        TPR, FPR, ACC = calculate_tpr_fpr_no_indices(gt_poison_l, pd_poison_l)
        print('ACC: {:.3f}, (TPR, FPR): ({:.3f}, {:.3f})'.format(ACC, TPR, FPR))
        writer.add_scalar('tra/tpr', TPR, batch_id)
        writer.add_scalar('tra/fpr', FPR, batch_id)
        writer.add_scalar('tra/acc', ACC, batch_id)

        if epoch_id % vis_repeat == 0:
            aver_dist_l, q_l, untrusted_label_l, gt_poison_l, pd_poison_l = \
                torch.cat(aver_dist_l).cpu().numpy(), torch.cat(q_l).cpu().numpy(), \
                torch.cat(untrusted_label_l).cpu().numpy(), np.array(gt_poison_l), \
                np.array(pd_poison_l)

            start_t = time.time()
            statistics_analysis(aver_dist_l, q_l, untrusted_label_l, gt_poison_l, pd_poison_l, 'picture/{}_{}_{}_hist_{}_mn_{}.png'.format(poison_type, poison_ratio, num_sample, epoch_id, model_name))
            print('cos and q distribution take time {}'.format(time.time()-start_t))
            # update the weights
            start_t = time.time()
            untrusted_act_l, indices_l, weight_act_l, val_weight_est_lables = torch.cat(untrusted_act_l).cpu().detach().numpy(),\
                torch.cat(indices_l).cpu().detach().numpy(), weight_act_l.cpu().detach().numpy(), val_weight_est_lables.cpu().detach().numpy()

            weights, weights_indices = update_weights(weights, weights_indices, untrusted_act_l, untrusted_label_l,
                                                      indices_l, gt_poison_l, weight_act_l, val_weight_est_lables,
                                                      'picture/{}_{}_{}_hist_{}_weights_{}.png'.format(poison_type, poison_ratio, num_sample, epoch_id, model_name), threshold_percent)
            weights, weights_indices = torch.from_numpy(weights), torch.from_numpy(weights_indices)
            print('weight estimation {}s'.format(time.time()-start_t))

            # # calculate the weight auc over whole dataset
            # start_t = time.time()
            # auc_weight_ds = calculate_auc_weight_ds(weights, weights_indices, ssl_dl)
            # print('calculate the auc of weigth over dataset: {:.3f}'.format(auc_weight_ds))
            # print('auc estimation {}s'.format(time.time()-start_t))
            #
            # writer.add_scalar('tra/auc_weight_ds', auc_weight_ds, epoch_id)

    return batch_id, weights, weights_indices

#
# def ssl_training(lowest_value, largest_value, epoch_id, poison_type, poison_ratio, model, ssl_dl, val_dl, criterion, mixmatch_criterion, lr_schedule, optimizer, batch_id, device, warmup_epoch, xloss):
#     isolate_ratio = 0.05
#     lambda1, lambda2 = ablation_dict[xloss]
#     model.train()
#     gt_poison_l, pd_poison_l, aver_dist_l, aver_var_l, q_l, untrusted_label_l, emb_l, loss_l, \
#         aver_loss_untrusted_l, poison_indices_l = [], [], [], [], [], [], [], [], [], []
#     for b_id, batch in tqdm(enumerate(ssl_dl), total=len(ssl_dl)):
#         batch, untrusted_label, poison_flags, indices = batch[:-3], batch[-3].to(device), batch[-2].to(device), batch[-1].to(device)
#         bs = len(batch[0])
#         if bs < ssl_dl.batch_size:
#             continue
#         untrusted_label_l.append(untrusted_label)
#
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr_schedule[batch_id]
#         optimizer.zero_grad()
#         batch_val = get_batch_val(val_dl) # TODO: check whether this will affect the TPR and FPR of poison detection
#         l_data, l_labels = batch_val[0], batch_val[1].to(device)
#
#         if batch_id < warmup_epoch * (len(ssl_dl) - 1): # warm-up
#             loss_swav, loss_ad, loss_mm = torch.tensor(0), torch.tensor(0), torch.tensor(0)
#             _, outputs = model(l_data.to(device), feature_flag=True)
#             loss_ce = F.cross_entropy(outputs, l_labels)
#         else:
#             embedding, outputs = model(torch.cat([l_data]+batch).to(device), feature_flag=True)
#
#             low_confdience_softmax = outputs[len(l_labels):]
#             low_confdience_softmax_num_aug = [low_confdience_softmax[i * bs:(i + 1) * bs] for i in range(ssl_dl.dataset.dataset.number_aug)]
#
#             loss_ce = F.cross_entropy(outputs[:len(l_labels)], l_labels) # ce for known data
#             loss_untrusted = F.cross_entropy(outputs[len(l_labels):], torch.cat([untrusted_label for i in range(ssl_dl.dataset.dataset.number_aug)]), reduction='none')
#             aver_loss_untrusted = torch.mean(torch.stack([loss_untrusted[i * bs:(i + 1) * bs] for i in range(ssl_dl.dataset.dataset.number_aug)]), dim=0)
#             aver_loss_untrusted_l.append(aver_loss_untrusted)
#             # get the anchor for each class
#             anchor = torch.stack([torch.mean(embedding[:len(l_labels)][l_labels == c], dim=0)
#                              for c in range(len(val_dl.dataset.dataset.classes))])
#             anchor = nn.functional.normalize(anchor, dim=1, p=2)
#
#             # swap representation
#             ulabeled_embedding = embedding[len(l_labels):]
#             aver_dist, q = criterion(ulabeled_embedding, anchor, bs)
#             aver_dist_l.append(aver_dist)
#             q_l.append(q)
#             # # calculate the variance
#             # var_emb = cal_variance_emb(ulabeled_embedding, bs, ssl_dl.dataset.dataset.number_aug)
#             # var_emb_l.append(var_emb)
#
#             # pseduo labeling when predicted label=untrusted label
#             poison_indices, batch_loss, aver_embedding, high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, low_confidence_data, low_confdience_embedding, \
#                 gt_poison, pd_poison = pseudoLabeling(untrusted_label, poison_flags,
#                                           batch, ulabeled_embedding, q, -aver_loss_untrusted, [],# carfully to the loss value should be -loss as input
#                                           ssl_dl, bs, device)
#             poison_indices_l.append(poison_indices)
#             loss_l.append(batch_loss)
#             emb_l.append(aver_embedding)
#             gt_poison_l += gt_poison
#             pd_poison_l += pd_poison
#             # increase the anchor discrimination via high confidence data and triplet loss
#             # loss_ad = discrimination_loss(anchor.to(device)).to(device)
#             # loss_ad = triplet_loss(anchor.to(device), torch.cat(high_confdience_embedding).to(device), torch.cat(high_confdience_pseudolabels).to(device))
#             # high confidence data to fine-tune the fc layer
#             # losses_high = F.cross_entropy(model.fc(torch.cat(high_confdience_embedding).to(device)),
#             #                 torch.cat(high_confdience_pseudolabels).to(device), reduction='none')
#             # weights = (high_confdience_var_emb - high_confdience_var_emb.min()) / (high_confdience_var_emb.max() - high_confdience_var_emb.min())
#             # weights = weights.repeat([ssl_dl.dataset.dataset.number_aug] + [1 for i in range(len(weights.shape) - 1)])
#             # weighted_losses = losses_high * weights
#             #
#             # # Step 3: Compute the final weighted loss (e.g., sum or mean)
#             # final_loss = weighted_losses.mean()
#
#             loss_ce += lambda1 * F.cross_entropy(model.fc(torch.cat(high_confdience_embedding).to(device)),
#                             torch.cat(high_confdience_pseudolabels).to(device))
#
#             # mixmatch the high and low confidence data
#             if high_confidence_data[0].shape[0] == 0 or low_confidence_data[0].shape[0] == 0:
#                 loss_mm = torch.tensor(0).to(device)
#             else:
#                 loss_mm = mixMatchLoss(mixmatch_criterion, model, low_confidence_data, low_confdience_softmax_num_aug,
#                                        high_confidence_data, high_confdience_pseudolabels,
#                                        len(ssl_dl.dataset.dataset.classes),
#                                        ssl_dl.dataset.dataset.number_aug, device, batch_id)
#
#         loss = loss_ce + lambda2 * loss_mm # + lambda3 * loss_swav
#         # loss = loss_sd_l
#
#         loss.backward()
#
#         optimizer.step()
#         batch_id += 1
#         writer.add_scalar('tra/loss_v', loss.item(), batch_id)
#         writer.add_scalar('tra/loss_ce', loss_ce.item(), batch_id)
#         # writer.add_scalar('tra/loss_swav', loss_swav.item(), batch_id)
#         # writer.add_scalar('tra/loss_ad', loss_ad.item(), batch_id)
#         writer.add_scalar('tra/loss_mm', loss_mm.item(), batch_id)
#
#         # print('loss_v: {}'.format(loss.item()))
#         # print('loss_ce: {}'.format(loss_ce.item()))
#         # # print('loss_swav: {}'.format(loss_swav.item()))
#         # # print('loss_ad: {}'.format(loss_ad.item()))
#         # print('loss_mm: {}'.format(loss_mm.item()))
#
#     if gt_poison_l == [] and pd_poison_l == []:
#         pass
#     else:
#         TPR, FPR, ACC = calculate_tpr_fpr_no_indices(gt_poison_l, pd_poison_l)
#         print('ACC: {:.3f}, (TPR, FPR): ({:.3f}, {:.3f})'.format(ACC, TPR, FPR))
#         writer.add_scalar('tra/tpr', TPR, batch_id)
#         writer.add_scalar('tra/fpr', FPR, batch_id)
#         writer.add_scalar('tra/acc', ACC, batch_id)
#         # draw histogram of distance and p per classes
#         poison_indices_l, aver_loss_untrusted_l, aver_dist_l, q_l, emb_l, untrusted_label_l, gt_poison_l, pd_poison_l, anchor = \
#             torch.cat(poison_indices_l).cpu().detach().numpy(), torch.cat(aver_loss_untrusted_l).cpu().detach().numpy(), \
#             torch.cat(aver_dist_l).cpu().numpy(), torch.cat(q_l).cpu().numpy(), torch.cat(emb_l).detach().cpu().numpy(), \
#             torch.cat(untrusted_label_l).cpu().numpy(), np.array(gt_poison_l), np.array(pd_poison_l), anchor.cpu().detach().numpy()
#         # cache_easy_learn_sample += np.array([1 if i in poison_indices_l else 0 for i in range(len(cache_easy_learn_sample))])
#         # if epoch_id % 5 ==0:
#         statistics_analysis(aver_dist_l, q_l, aver_loss_untrusted_l, untrusted_label_l, gt_poison_l, pd_poison_l, '{}_{}_hist_{}_mid_filter.png'.format(poison_type, poison_ratio, epoch_id))
#             # draw_dist_loss_pair(aver_dist_l, untrusted_label_l, aver_loss_untrusted_l, '{}_{}_{}_loss_vs_dist.png'.format(poison_type, poison_ratio, epoch_id), ylabel='cos sim')
#             # draw_dist_loss_pair(q_l, untrusted_label_l, aver_loss_untrusted_l, '{}_{}_{}_loss_vs_q.png'.format(poison_type, poison_ratio, epoch_id), ylabel='q')
#             # draw_dist_q_pair(aver_dist_l, q_l, untrusted_label_l, '{}_{}_{}_dist_vs_q.png'.format(poison_type, poison_ratio, epoch_id))
#             # Save the emb as an image
#             # emb_vis(emb_l, anchor, untrusted_label_l, gt_poison_l, pd_poison_l, ssl_dl.dataset.dataset.number_aug, '{}_{}_emb_{}_strong_aug.png'.format(poison_type, poison_ratio, epoch_id))
#             # # save the emebdding of anchor
#             # anchor_vis(anchor.cpu().detach().numpy(), '{}_anchor_{}.png'.format(poison_type, epoch_id))
#         # calculate the globel 10 threshold
#         sorted_predicted_pos, _ = torch.sort(torch.cat(loss_l))
#         # Get the 10%th lowest and 10%th largest value
#         index = int(len(torch.cat(loss_l)) * isolate_ratio)  # This equals 12
#         lowest_value = sorted_predicted_pos[index]  # 10%th lowest
#         largest_value = sorted_predicted_pos[-index - 1]  # 10%th largest
#
#     return batch_id, lowest_value, largest_value
#

# def ssl_training(lowest_value, largest_value, epoch_id, poison_type, poison_rate, model, ssl_dl, val_dl, criterion, mixmatch_criterion, lr_schedule, optimizer, batch_id, device, warmup_epoch,
#                  xloss):
#     isolate_ratio = 0.1
#     lambda1, lambda2 = ablation_dict[xloss]
#     model.train()
#
#     # Use preallocated lists or tensors to reduce repeated allocations
#     untrusted_label_l, loss_l, aver_loss_untrusted_l, poison_indices_l = [], [], [], []
#     gt_poison_l, pd_poison_l = [], []
#     aver_dist_l, q_l = [], []
#     samples_idx_l = []
#
#     batch_val = get_batch_val(val_dl)
#     l_data, l_labels = batch_val[0], batch_val[1].to(device)
#
#     num_classes = len(val_dl.dataset.dataset.classes)
#     number_aug = ssl_dl.dataset.dataset.number_aug
#     batch_size = ssl_dl.batch_size
#
#     for b_id, batch in tqdm(enumerate(ssl_dl), total=len(ssl_dl)):
#         batch, untrusted_label, poison_flags, indices = batch[:-3], batch[-3].to(device), batch[-2].to(device), batch[-1]
#         samples_idx_l.append(indices)
#         bs = len(batch[0])
#         # if bs < batch_size:
#         #     continue
#
#         untrusted_label_l.append(untrusted_label)
#
#         # Update learning rate per batch
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr_schedule[batch_id]
#         optimizer.zero_grad()
#
#         if batch_id < warmup_epoch * (len(ssl_dl) - 1):
#             loss_ce = F.cross_entropy(model(l_data.to(device), feature_flag=True)[1], l_labels)
#             loss_mm = torch.tensor(0).to(device)  # No MixMatch during warmup
#         else:
#             embedding, outputs = model(torch.cat([l_data] + batch).to(device), feature_flag=True)
#             low_confdience_softmax = outputs[len(l_labels):]
#             low_confdience_softmax_num_aug = [low_confdience_softmax[i * bs:(i + 1) * bs] for i in range(number_aug)]
#
#             loss_ce = F.cross_entropy(outputs[:len(l_labels)], l_labels)
#             loss_untrusted = F.cross_entropy(outputs[len(l_labels):],
#                                              torch.cat([untrusted_label for _ in range(number_aug)]), reduction='none')
#             aver_loss_untrusted = torch.mean(
#                 torch.stack([loss_untrusted[i * bs:(i + 1) * bs] for i in range(number_aug)]), dim=0)
#             aver_loss_untrusted_l.append(aver_loss_untrusted)
#
#             anchor = torch.stack(
#                 [torch.mean(embedding[:len(l_labels)][l_labels == c], dim=0) for c in range(num_classes)])
#             anchor = nn.functional.normalize(anchor, dim=1, p=2)
#
#             ulabeled_embedding = embedding[len(l_labels):]
#             aver_dist, q = criterion(ulabeled_embedding, anchor, bs)
#             aver_dist_l.append(aver_dist)
#             q_l.append(q)
#
#             high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, low_confidence_data, _, gt_poison, pd_poison, batch_loss = \
#                 pseudoLabeling(untrusted_label, poison_flags, batch, ulabeled_embedding, q, -aver_loss_untrusted, lowest_value, largest_value, isolate_ratio, ssl_dl, bs, device)
#             loss_l.append(batch_loss)
#             gt_poison_l += gt_poison
#             pd_poison_l += pd_poison
#
#             if high_confidence_data[0].shape[0] != 0 and low_confidence_data[0].shape[0] != 0:
#                 loss_mm = mixMatchLoss(mixmatch_criterion, model, low_confidence_data, low_confdience_softmax_num_aug,
#                                        high_confidence_data, high_confdience_pseudolabels, num_classes, number_aug,
#                                        device, batch_id)
#             else:
#                 loss_mm = torch.tensor(0).to(device)
#
#             loss_ce += lambda1 * F.cross_entropy(model.fc(torch.cat(high_confdience_embedding).to(device)),
#                                                  torch.cat(high_confdience_pseudolabels).to(device))
#
#         loss = loss_ce + lambda2 * loss_mm
#         loss.backward()
#         optimizer.step()
#
#         # Log metrics (do outside of training loop if logging frequently slows performance)
#         writer.add_scalar('tra/loss_v', loss.item(), batch_id)
#         writer.add_scalar('tra/loss_ce', loss_ce.item(), batch_id)
#         writer.add_scalar('tra/loss_mm', loss_mm.item(), batch_id)
#
#         batch_id += 1
#
#
#     data_4_easily_sample_detection = {}
#     if gt_poison_l and pd_poison_l:
#         TPR, FPR, ACC = calculate_tpr_fpr_no_indices(gt_poison_l, pd_poison_l)
#         writer.add_scalar('tra/tpr', TPR, batch_id)
#         writer.add_scalar('tra/fpr', FPR, batch_id)
#         writer.add_scalar('tra/acc', ACC, batch_id)
#         # if epoch_id % 5 == 0:
#         aver_dist_l, q_l = torch.cat(aver_dist_l).cpu().detach().numpy(), torch.cat(q_l).cpu().detach().numpy()
#         untrusted_label_l, gt_poison_l, pd_poison_l = torch.cat(untrusted_label_l).cpu().numpy(), np.array(gt_poison_l), np.array(pd_poison_l)
#         aver_loss_untrusted_l = torch.cat(aver_loss_untrusted_l).cpu().detach().numpy()
#         samples_idx_l = torch.cat(samples_idx_l).cpu().detach().numpy()
#
#         aver_dist_l_target, q_l_target = np.stack([dist[label] for dist, label in zip(aver_dist_l, untrusted_label_l)]), np.stack([q[label] for q, label in zip(q_l, untrusted_label_l)])
#         # if epoch_id % 5 == 0:
#         statistics_analysis(aver_dist_l_target, q_l_target, aver_loss_untrusted_l, untrusted_label_l, gt_poison_l, pd_poison_l, '{}_{}_hist_{}_mid_filter.png'.format(poison_type, poison_rate, epoch_id))
#         data_4_easily_sample_detection = {
#             'aver_loss_untrusted_l': aver_loss_untrusted_l, 'aver_dist_l_target': aver_dist_l_target, 'untrusted_label_l': untrusted_label_l,
#             'q_l_target': q_l_target, 'gt_poison_l': gt_poison_l, 'pd_poison_l': pd_poison_l, 'samples_idx_l': samples_idx_l
#         }
#         sorted_predicted_pos, _ = torch.sort(torch.cat(loss_l))
#         # Get the 10%th lowest and 10%th largest value
#         index = int(len(torch.cat(loss_l)) * isolate_ratio)  # This equals 12
#         lowest_value = sorted_predicted_pos[index]  # 10%th lowest
#         largest_value = sorted_predicted_pos[-index - 1]  # 10%th largest
#
#     return batch_id, data_4_easily_sample_detection, lowest_value, largest_value
#


def cosinelr_gen(base_lr, final_lr, epochs, train_loader):
    warmup_epochs, start_warmup = 0, 0
    warmup_lr_schedule = np.linspace(start_warmup, base_lr, len(train_loader) * warmup_epochs)
    iters = np.arange(len(train_loader) * (epochs - warmup_epochs))
    cosine_lr_schedule = np.array([final_lr +
                                   0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / (len(train_loader) * (epochs - warmup_epochs))))
                                   for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # # draw out the lr plot
    # import matplotlib.pyplot as plt
    # plt.plot((iters+1)/len(train_loader), lr_schedule)
    # plt.savefig('lr_plt.png')
    # plt.close()

    return lr_schedule


def get_hyperparameters(poison_type, model_name='cnn'):
    if poison_type == 'adaptivecifar10' or 'freq' in poison_type or poison_type == 'pattern' \
            or 'wanet' in poison_type:
        batch_size, num_workers = 192, 4
        lr, final_lr, epoch_num = 0.01, 0.0001, 55
    elif 'adaptiveattack' in poison_type or poison_type == 'blto':
        if model_name == 'cnn' or model_name == 'efficient':
            batch_size, num_workers = 256, 4
            lr, final_lr, epoch_num = 0.01, 0.0001, 55
        elif model_name == 'transformer':
            batch_size, num_workers = 125, 4
            lr, final_lr, epoch_num = 0.001, 0.0001, 55
        else:
            print('no parameter for {}'.format(model_name))
            exit(-1)
    elif poison_type == 'corruptencoder' or poison_type == 'depud' or poison_type == 'adp_corrupt':
        batch_size, num_workers = 256, 4
        lr, final_lr, epoch_num = 0.01, 0.0001, 55
    elif poison_type == 'ultrasonic':
        batch_size, num_workers = 256, 4
        lr, final_lr, epoch_num = 0.01, 0.0001, 55
    else:
        print('error in get_hyperparameters')
        exit(-1)
    print(batch_size, num_workers, lr, final_lr, epoch_num)

    return batch_size, num_workers, lr, final_lr, epoch_num

#
# def mad_outlier_detection(data, threshold_factor=5):
#     # Convert to NumPy array for efficiency
#     data = np.array(data)
#
#     # Calculate median
#     median = np.median(data)
#
#     # Calculate absolute deviations from the median
#     abs_deviation = np.abs(data - median)
#
#     # Calculate MAD
#     mad = np.median(abs_deviation)
#
#     # Calculate threshold for outliers
#     threshold = threshold_factor * mad
#
#     # Identify outliers
#     outliers_index = (abs_deviation > threshold)
#     print(abs_deviation, threshold)
#
#     return median, mad, threshold, outliers_index


# def detect_easily_learned_sample(data_4_easily_sample_detection):
#     aver_loss_untrusted_l = data_4_easily_sample_detection['aver_loss_untrusted_l']
#     aver_dist_l_target = data_4_easily_sample_detection['aver_dist_l_target']
#     q_l_target = data_4_easily_sample_detection['q_l_target']
#     untrusted_label_l = data_4_easily_sample_detection['untrusted_label_l']
#     gt_poison_l = data_4_easily_sample_detection['gt_poison_l']
#     # pd_poison_l = data_4_easily_sample_detection['pd_poison_l']
#     samples_idx_l = data_4_easily_sample_detection['samples_idx_l']
#
#     q_1_over_q_2 = []
#     # do normalization
#     # Function for Min-Max Normalization
#     def min_max_normalize(data):
#         min_val = np.min(data)
#         max_val = np.max(data)
#         # Avoid division by zero
#         if max_val - min_val == 0:
#             return np.zeros_like(data)  # or return np.ones_like(data) if you want a uniform array
#         normalized_data = (data - min_val) / (max_val - min_val)
#         return normalized_data
#
#     # Normalize the arrays
#     normalized_aver_loss_untrusted_l = min_max_normalize(aver_loss_untrusted_l)
#     normalized_aver_dist_l_target = min_max_normalize(aver_dist_l_target)
#     normalized_q_l_target = min_max_normalize(q_l_target)
#
#     for i in np.unique(untrusted_label_l):
#         measure_v = normalized_aver_loss_untrusted_l[untrusted_label_l==i] - normalized_aver_dist_l_target[untrusted_label_l==i] - normalized_q_l_target[untrusted_label_l==i]
#         gt_poison_l_i = gt_poison_l[untrusted_label_l==i]
#
#         # draw histogram
#         plt.hist(measure_v[gt_poison_l_i==0], label='benign')
#         if np.sum(gt_poison_l_i) != 0:
#             plt.hist(measure_v[gt_poison_l_i==1], label='poison')
#
#         plt.savefig('hist_{}.png'.format(i))
#         plt.close()
#         measure_v = measure_v.reshape(-1, 1)
#         gmm = mixture.GaussianMixture(
#             n_components=2, covariance_type="full"
#         )
#         gmm.fit(measure_v)
#         bic_v2 = gmm.bic(measure_v)
#
#         gmm = mixture.GaussianMixture(
#             n_components=1, covariance_type="full"
#         )
#         gmm.fit(measure_v)
#         bic_v1 = gmm.bic(measure_v)
#         q_1_over_q_2.append(bic_v1/(bic_v2 + 1e-6))
#
#     # outlier detection
#     median, mad, threshold, outliers_flag = mad_outlier_detection(q_1_over_q_2)
#     print(median, mad, threshold, outliers_flag)
#     detect_sample_index = []
#     for i, f in enumerate(outliers_flag):
#         if f == True:
#             measure_v = normalized_aver_loss_untrusted_l[untrusted_label_l==i] - normalized_aver_dist_l_target[untrusted_label_l==i] - normalized_q_l_target[untrusted_label_l==i]
#             measure_v = measure_v.reshape(-1, 1)
#
#             samples_idx_i = samples_idx_l[untrusted_label_l==i]
#             gmm = mixture.GaussianMixture(
#                 n_components=2, covariance_type="full"
#             )
#             gmm.fit(measure_v)
#             labels = gmm.predict(measure_v)
#             if np.mean(measure_v[labels==0]) < np.mean(measure_v[labels==1]):
#                 detect_sample_index.append(samples_idx_i[labels==0])
#             else:
#                 detect_sample_index.append(samples_idx_i[labels==1])
#         else:
#             pass
#
#     detect_sample_index = np.concatenate(detect_sample_index) if detect_sample_index != [] else []
#     # calculate the TPR and FPR
#     pd_poison_l = [1 if idx in detect_sample_index else 0 for idx in samples_idx_l]
#     TPR, FPR, ACC = calculate_tpr_fpr_no_indices(gt_poison_l, pd_poison_l)
#     print('easy learn detection TPR {:.3f}, FPR {:.3f} and ACC {:.3f}'.format(TPR, FPR, ACC))
#
#     return detect_sample_index, TPR, FPR


def main(args):
    global writer
    args_str = '_'.join('{}'.format(value) for _, value in vars(args).items())
    writer = SummaryWriter(comment='{}_args_{}'.format(os.path.basename(__file__), args_str))
    print(args)

    # setting parameters
    num_class = args.num_class
    poison_or_benign = args.poison_or_benign
    poison_rate = args.poison_rate
    num_cluster = args.num_class    # WLOG, assume num_cluster == num_class
    num_sample = args.num_sample
    num_aug = args.num_aug
    num_ADiter = args.num_ADiter
    threshold_percent = args.threshold_percent
    device = args.device
    poison_type = args.poison_type
    model_cache = args.model_cache
    subset_files_cache = args.subset_files_cache
    effana = args.efficiency_analysis
    xloss = args.xloss
    model_name = args.model_name

    if poison_type == 'adp_corrupt':
        source_transforms = ToTargetClass(target_name=source_imagenet_name_adp, num_classes=num_class, poison_type=poison_type)
        source_class = source_transforms.target_class
        source_name = source_imagenet_name_adp
    else:
        source_class = None
        source_name = None

    # dataset
    batch_size, num_workers, lr, final_lr, epoch_num = get_hyperparameters(poison_type, model_name)

    # val dataset
    tr_dl, ts_dl, pts_dl, train_folder = get_dataset2(poison_type, poison_or_benign, poison_rate, batch_size, num_class,
                                                      num_workers, transforms=True, num_ADiter=num_ADiter,
                                                      source_name=source_name, source_class=source_class, model_name=model_name)
    num_sample = num_sample
    val_dl, subset_indics, subset_indics_left = get_validation_data(tr_dl.dataset, num_sample, batch_size=batch_size,
                                                num_workers=num_workers, cache_subset_files='poisonDataset/{}/{}'.format(poison_type, subset_files_cache) if subset_files_cache!=None else None)

    # alert
    if tr_dl.dataset.benign_indics == len(tr_dl.dataset):
        if args.poison_or_benign == 'benign':
            pass
        else:
            exit(-1)
    ssl_ds = get_ssldata(poison_type, train_folder, num_class, num_aug, benign_indics=tr_dl.dataset.benign_indics, model_name=model_name)
    ssl_ds = Subset(ssl_ds, subset_indics_left)
    ssl_dl = DataLoader(dataset=ssl_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # model and optimizer
    model = gen_model(poison_type, num_class, num_cluster, model_name=model_name)
    model.to(device)

    warm_up_epoch = 5 # TODO
    if model_cache != None:
        cache_model_path = 'poisonDataset/{}/{}'.format(poison_type, model_cache)
        model.load_state_dict(torch.load(cache_model_path, map_location=device))
        acc_best = evaluating(model, ts_dl, 0, device, writer)
        asr_best = evaluating(model, pts_dl, 0, device, writer, poison_flag=True)
        print('Cache file: {}'.format(cache_model_path))
        print('(ACC, ASR): ({:.3f}, {:.3f})'.format(acc_best, asr_best))
        # warm_up_epoch = -1
        exit(-1)

    # change the loss function to swav
    criterion = get_loss_fun(method='swav', parameter_dict={'number_aug': num_aug})

    # mixmatch_criterion = get_loss_fun(method='mixmatch', parameter_dict={'rampup_length': epoch_num, 'lambda_u': 15})
    # if model_name == 'transformer':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    #     print('transformer with optimizer: {}'.format(optimizer))
    # else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    # lr_schedule = cosinelr_gen(lr, final_lr, epoch_num, ssl_dl)
    # lr_schedule = None
    batch_id = 0
    # SSL
    acc_best = evaluating(model, ts_dl, 0, device, writer) # torch.tensor(0) #
    asr_best = evaluating(model, pts_dl, 0, device, writer, poison_flag=True) # torch.tensor(0)
    epoch_acc_asr = [[0, acc_best.item(), asr_best.item()]]
    # cache_easy_learn_sample = np.zeros(len(tr_dl.dataset))
    weights = torch.ones(len(ssl_ds.indices)) # initialize the weights untrusted data as 1
    weights_indices = torch.tensor(ssl_ds.indices)

    # weights = torch.load('weights.pt')
    # weights_indices = torch.load('weights_indices.pt')
    vis_repeat = 5
    for epoch in range(1, epoch_num+1):
        print('Epoch: {}'.format(epoch))
        batch_id, weights, weights_indices = ssl_training_weight(epoch, weights, weights_indices, poison_type, poison_rate,
                                     model, ssl_dl, val_dl, criterion, optimizer, batch_id, device,
                                     warm_up_epoch, xloss, num_sample, threshold_percent, model_name, writer=writer, vis_repeat=vis_repeat)

        if epoch % 5 == 0:
            # eval_cluster(model, tr_dl, device, epoch)
            acc = evaluating(model, ts_dl, epoch, device, writer)
            asr = evaluating(model, pts_dl, epoch, device, writer, poison_flag=True)
            if effana:
                epoch_acc_asr.append([epoch, acc.item(), asr.item()])
            if acc > acc_best:
                print('acc {} > acc_best {}'.format(acc, acc_best))
                acc_best = acc
                torch.save(model.state_dict(),
                   'poisonDataset/{}/check_ssltrainslmixmatch_weight_{}_{}_{}_{}_samplen_{}_augn_{}_mn_{}_best_acc.pth'.format(poison_type, num_class,
                                                                                                poison_or_benign,
                                                                                                poison_rate, epoch_num,
                                                                                                num_sample, num_aug, model_name))

    if effana:
        np.save('poisonDataset/{}/ssltrainslmixmatch_weight_epoch_acc_samplen_{}_augn_{}_xloss_{}.npy'.format(poison_type, num_sample, num_aug, xloss), np.array(epoch_acc_asr))

    torch.save(model.state_dict(), 'poisonDataset/{}/ssltrainslmixmatch_weight_{}_{}_{}_{}_samplen_{}_augn_{}_mn_{}.pth'.format(poison_type, num_class, poison_or_benign, poison_rate, epoch_num, num_sample, num_aug, model_name))


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')
        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str, help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cuda:0', type=str, help='The device to use (e.g., "cpu" or "cuda").')
        parser.add_argument('-sample', '--num_sample', required=True, type=int, help='The number of labeled samples.')
        parser.add_argument('-aug_n', '--num_aug', required=True, type=int, help='The number of augmentations.')
        parser.add_argument('-aniso_n', '--num_ADiter', default=None, type=int, help='The iteration number of anisotropic disffusion.')
        parser.add_argument('-cache', '--model_cache', default=None, type=str, help='Cached model as a pretrained model.')
        parser.add_argument('-cache_subset_files', '--subset_files_cache', default=None, type=str, help='Cached benign files name')
        parser.add_argument('-mn', '--model_name', default='cnn', type=str, help='For adatpiveattack, use transformer')

        parser.add_argument('-th', '--threshold_percent', default=0.9, type=float, help='The threshold for weight normalization.')
        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')
        parser.add_argument('-effana', '--efficiency_analysis', default=False, type=bool,
                            help='Analyse the training efficiency by saving acc at intervals of every five epochs')
        parser.add_argument('-xloss', '--xloss', type=str, default='all', choices=ablation_dict.keys(),
                            help='Ablation study for loss, where all means all loss funs are included, ce means ce '
                                 'is excluded, tri means triple is exluded, mm means mixmatch is excluded, swva means swav is excluded')
        return parser.parse_args()

    args = parse_args()
    main(args)
