import os, torch, librosa
# import random
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import gc
#
# import PIL
# import pedalboard
# from audiomentations import PitchShift, TimeStretch, AddGaussianSNR, Compose, LowPassFilter
# import torch_audiomentations
import numpy as np
import torchaudio
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
# from torch_pitch_shift import pitch_shift
from torchvision.datasets import ImageFolder

from tqdm import tqdm
import seaborn as sns
from PIL import Image

from lib import augmentation
# from lib.models import TFCModel
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
# import multiprocessing as mp

from lib.augmentation import spec_aug_audio, time_aug_audio, image_saug_cifar_freeMatch, \
    image_waug_cifar_freeMatch, image_no_aug_cifar_freeMatch, image_waug_imagenet_freeMatch, \
    image_no_aug_imagenet_freeMatch, image_saug_imagenet_freeMatch, image_no_aug_cifar_freeMatch_blto, \
    image_waug_cifar_freeMatch_blto, image_saug_cifar_freeMatch_blto, image_waug_cifar_freeMatch_224, \
    image_no_aug_cifar_freeMatch_224, image_saug_cifar_freeMatch_224, image_no_aug_cifar_freeMatch_tensor

classes_2 = ['down', 'go']
classes_10 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
classes_30 = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
              'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
              'shella', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
target_audio = 'down'

target_cifar = 2 # bird
target_cifar_name = 'bird'

target_cifar_freq = 0 # bird
target_cifar_name_freq = 'airplane'

classes_10_imagenet = ['n02116738', 'n02093859', 'n04548362', 'n02441942', 'n03447447', 'n09421951', 'n03868863', 'n03743016', 'n03207941', 'n01968897']
target_imagenet = 'n02116738'
target_imagenet_blto = 9
target_imagenet_name_blto = 'n01968897'
source_imagenet_name_adp = 'n03868863'

target_cifar_blto = 9
target_cifar_blto_name = 'truck'


def calculate_tpr_fpr_no_indices(gt_poison, pd_poison):
    # Convert boolean inputs to integers (if necessary)
    gt_poison = [int(x) for x in gt_poison]
    pd_poison = [int(x) for x in pd_poison]

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = sum((g == 1 and p == 1) for g, p in zip(gt_poison, pd_poison))
    FP = sum((g == 0 and p == 1) for g, p in zip(gt_poison, pd_poison))
    TN = sum((g == 0 and p == 0) for g, p in zip(gt_poison, pd_poison))
    FN = sum((g == 1 and p == 0) for g, p in zip(gt_poison, pd_poison))

    # Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Accuracy (ACC)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return TPR, FPR, ACC

#
# def pseudoLabeling(indices, lowest_value, largest_value, isolate_ratio, untrusted_label, poison_flags, batch, ulabeled_embeedding, q, negative_loss, ssl_dl, bs, device):
#     # high-confidence data
#     predicted_pos, predicted_class = torch.max(q, dim=1)
#     # Sort the tensor in ascending order
#
#     loss_to_target_class = negative_loss # torch.tensor([q_sample[label_sample] for q_sample, label_sample in zip(q, untrusted_label)]).to(device)
#     correct_prediction_predicted_pos = loss_to_target_class[predicted_class == untrusted_label]
#     if lowest_value == largest_value == None:
#          # equal check loss(x,y)
#         sorted_predicted_pos, _ = torch.sort(correct_prediction_predicted_pos)
#         # Get the 10%th lowest and 10%th largest value
#         index = int(len(correct_prediction_predicted_pos) * isolate_ratio)  # This equals 12
#         lowest_value = sorted_predicted_pos[index]  # 10%th lowest
#         largest_value = sorted_predicted_pos[-index - 1]  # 10%th largest
#     else:
#         lowest_value = lowest_value
#         largest_value = largest_value
#
#     high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, aver_embedding = [], [], [], []
#     low_confidence_data, low_confdience_embedding = [], []
#     high_flag = (predicted_class == untrusted_label) #& (loss_to_target_class < largest_value)
#     low_flag = (predicted_class != untrusted_label) #| (loss_to_target_class >= largest_value)
#     # filter out the clean-label data
#     label_agree_but_large_loss = (predicted_class == untrusted_label) & (loss_to_target_class >= largest_value)
#     poison_indices = indices[label_agree_but_large_loss]
#     # high_flag = (predicted_class == untrusted_label) & (lowest_value < predicted_pos) & (predicted_pos < largest_value)
#     # low_flag = (predicted_class != untrusted_label) | (predicted_pos <= lowest_value) | (predicted_pos >= largest_value)
#
#     for i in range(ssl_dl.dataset.dataset.number_aug):
#         batch[i] = batch[i].to(device)
#         # TODO: when predicted_class==untrusted_label, only use middle part as the trusted data
#         high_confidence_data.append(batch[i][high_flag])
#         high_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][high_flag])
#         high_confdience_pseudolabels.append(predicted_class[high_flag])
#         low_confidence_data.append(batch[i][low_flag])
#         low_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][low_flag])
#
#         aver_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs])
#     # print('additional high confidential data num: {}'.format(
#     #     len(high_confdience_embedding) * torch.sum(predicted_class == untrusted_label)))
#     # print('additional low confidential data num: {}'.format(
#     #     len(low_confidence_data) * torch.sum(predicted_class != untrusted_label)))
#
#     # calculate the TPR, FPR and ACC
#     gt_poison, pd_poison = poison_flags, (high_flag==False) # low_flag
#     gt_poison = [int(x) for x in gt_poison]
#     pd_poison = [int(x) for x in pd_poison]
#
#     aver_embedding_ = torch.mean(torch.stack(aver_embedding), dim=0)
#
#     return poison_indices, correct_prediction_predicted_pos, aver_embedding_, high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, \
#          low_confidence_data, low_confdience_embedding, gt_poison, pd_poison

#
# def pseudoLabeling(untrusted_label, poison_flags, batch, ulabeled_embedding, q, negative_loss, isolated_index, ssl_dl, bs, device):
#     # high-confidence data
#     predicted_pos, predicted_class = torch.max(q, dim=1)
#
#     loss_to_target_class = negative_loss # torch.tensor([q_sample[label_sample] for q_sample, label_sample in zip(q, untrusted_label)]).to(device)
#     correct_prediction_predicted_pos = loss_to_target_class[predicted_class == untrusted_label]
#
#     # Flags for high and low confidence
#     high_flag = (predicted_class == untrusted_label)  & (isolated_index == 0)
#     low_flag = (predicted_class != untrusted_label) | (isolated_index == 1)
#
#     # Preallocate lists for high and low confidence data
#     high_confidence_data = []
#     high_confidence_embedding = []
#     high_confidence_pseudolabels = []
#     low_confidence_data = []
#     low_confidence_embedding = []
#
#     for i in range(ssl_dl.dataset.dataset.number_aug):
#         # Move the batch to the device only once
#         batch[i] = batch[i].to(device)
#
#         # Use boolean indexing to filter the tensors
#         high_confidence_data.append(batch[i][high_flag])
#         high_confidence_embedding.append(ulabeled_embedding[i * bs:(i + 1) * bs][high_flag])
#         high_confidence_pseudolabels.append(predicted_class[high_flag])
#         low_confidence_data.append(batch[i][low_flag])
#         low_confidence_embedding.append(ulabeled_embedding[i * bs:(i + 1) * bs][low_flag])
#
#     # Calculate the TPR, FPR, and ACC
#     gt_poison = poison_flags
#     pd_poison = ~high_flag  # Invert high_flag to get low_flag directly
#     gt_poison = [int(x) for x in gt_poison]
#     pd_poison = [int(x) for x in pd_poison]
#
#     return (
#         high_confidence_data,
#         high_confidence_embedding,
#         high_confidence_pseudolabels,
#         low_confidence_data,
#         low_confidence_embedding,
#         gt_poison,
#         pd_poison
#     )

def pseudoLabeling_ori(indices, weights, lowest_value, largest_value, isolate_ratio, untrusted_label, poison_flags, batch, ulabeled_embeedding, q, negative_loss, ssl_dl, bs, device):
    # high-confidence data
    predicted_pos, predicted_class = torch.max(q, dim=1)
    # Sort the tensor in ascending order

    # loss_to_target_class = negative_loss # torch.tensor([q_sample[label_sample] for q_sample, label_sample in zip(q, untrusted_label)]).to(device)
    # correct_prediction_predicted_pos = loss_to_target_class[predicted_class == untrusted_label]
     # equal check loss(x,y)
    # sorted_predicted_pos, _ = torch.sort(correct_prediction_predicted_pos)
    # Get the 10%th lowest and 10%th largest value
    # index = int(len(correct_prediction_predicted_pos) * isolate_ratio)  # This equals 12
    # lowest_value = sorted_predicted_pos[index]  # 10%th lowest
    # largest_value = sorted_predicted_pos[-index - 1]  # 10%th largest


    high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, aver_embedding = [], [], [], []
    low_confidence_data, low_confdience_embedding = [], []
    high_flag = (predicted_class == untrusted_label) # & (loss_to_target_class > lowest_value) #& (loss_to_target_class < largest_value)
    low_flag = (predicted_class != untrusted_label) # | (loss_to_target_class <= lowest_value) #| (loss_to_target_class >= largest_value)
    # filter out the clean-label data
    # label_agree_but_large_loss = (predicted_class == untrusted_label) & (loss_to_target_class >= largest_value)
    trusted_indices = indices[high_flag]
    weights_high = weights[high_flag]

    trusted_poison_flag = poison_flags[high_flag]
    # high_flag = (predicted_class == untrusted_label) & (lowest_value < predicted_pos) & (predicted_pos < largest_value)
    # low_flag = (predicted_class != untrusted_label) | (predicted_pos <= lowest_value) | (predicted_pos >= largest_value)

    for i in range(ssl_dl.dataset.dataset.number_aug):
        batch[i] = batch[i].to(device)
        # TODO: when predicted_class==untrusted_label, only use middle part as the trusted data
        high_confidence_data.append(batch[i][high_flag])
        high_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][high_flag])
        high_confdience_pseudolabels.append(predicted_class[high_flag])
        low_confidence_data.append(batch[i][low_flag])
        low_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][low_flag])

        # aver_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs])
    # print('additional high confidential data num: {}'.format(
    #     len(high_confdience_embedding) * torch.sum(predicted_class == untrusted_label)))
    # print('additional low confidential data num: {}'.format(
    #     len(low_confidence_data) * torch.sum(predicted_class != untrusted_label)))

    # calculate the TPR, FPR and ACC
    gt_poison, pd_poison = poison_flags, (high_flag==False) # low_flag

    # # set pd_poison as True when the weight<0
    # pd_poison[weights < 0] = True

    gt_poison = [int(x) for x in gt_poison]
    pd_poison = [int(x) for x in pd_poison]

    # aver_embedding_ = torch.mean(torch.stack(aver_embedding), dim=0)

    return trusted_indices, weights_high, None, None, high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, \
         low_confidence_data, low_confdience_embedding, gt_poison, pd_poison, trusted_poison_flag

def pseudoLabeling_with_easy_filter(isolation_flag, untrusted_label, poison_flags, batch, ulabeled_embeedding, q, ssl_dl, bs, device):
    # high-confidence data
    predicted_pos, predicted_class = torch.max(q, dim=1)
    # Sort the tensor in ascending order

    high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, aver_embedding = [], [], [], []
    low_confidence_data, low_confdience_embedding = [], []
    high_flag = (predicted_class == untrusted_label) & (isolation_flag == 0)
    low_flag = (predicted_class != untrusted_label) | (isolation_flag == 1)

    for i in range(ssl_dl.dataset.dataset.number_aug):
        batch[i] = batch[i].to(device)
        # TODO: when predicted_class==untrusted_label, only use middle part as the trusted data
        high_confidence_data.append(batch[i][high_flag])
        high_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][high_flag])
        high_confdience_pseudolabels.append(predicted_class[high_flag])
        low_confidence_data.append(batch[i][low_flag])
        low_confdience_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs][low_flag])

        aver_embedding.append(ulabeled_embeedding[i * bs:(i + 1) * bs])
    # print('additional high confidential data num: {}'.format(
    #     len(high_confdience_embedding) * torch.sum(predicted_class == untrusted_label)))
    # print('additional low confidential data num: {}'.format(
    #     len(low_confidence_data) * torch.sum(predicted_class != untrusted_label)))

    # calculate the TPR, FPR and ACC
    gt_poison, pd_poison = poison_flags, (high_flag==False) # low_flag
    gt_poison = [int(x) for x in gt_poison]
    pd_poison = [int(x) for x in pd_poison]

    aver_embedding_ = torch.mean(torch.stack(aver_embedding), dim=0)

    return aver_embedding_, high_confidence_data, high_confdience_embedding, high_confdience_pseudolabels, \
         low_confidence_data, low_confdience_embedding, gt_poison, pd_poison

def get_ssldata(poison_type, train_folder, num_class, number_aug, benign_indics, model_name='None'):
    if poison_type == 'adaptivecifar10' or poison_type == 'pattern' or poison_type == 'adp_corrupt' \
            or 'freq' in poison_type or poison_type == 'blto' or 'adaptiveattack' in poison_type or 'wanet' in poison_type:
        sslds = ImageDataSSL(root='poisonDataset/{}/{}'.format(poison_type, train_folder), number_aug=number_aug, benign_indics=benign_indics, model_name=model_name)
    else:
        sslds = AudioDataSSL(data_folder='poisonDataset/{}/{}'.format(poison_type, train_folder), number_class=num_class,
                 number_aug=number_aug, benign_indics=benign_indics)

    return sslds


def get_dataset_info(poison_type, poison_or_benign, poison_rate, source_name=None, num_ADiter=None):
    if 'adaptivecifar10' in poison_type or poison_type == 'pattern' or 'adaptiveattack' in poison_type or 'wanet' in poison_type:
        target_name = target_cifar
    elif poison_type == 'corruptencoder' or poison_type == 'depud':
        target_name = target_imagenet
    elif poison_type == 'blto': # or poison_type == 'adp_corrupt':
        target_name = target_cifar_blto
    elif 'freq' in poison_type:
        target_name = target_cifar_freq
    else:
        target_name = target_audio

    if poison_or_benign == 'benign':
        train_folder = 'Train'
    else:
        if poison_type == 'adp_corrupt':
            train_folder = 'poisonTrain_pr_{}_t_{}_s_{}'.format(poison_rate, target_name, source_name)
        else:
            train_folder = 'poisonTrain_pr_{}_t_{}'.format(poison_rate, target_name)
        if num_ADiter != None:
            train_folder += '_AD_niter_{}'.format(num_ADiter)
    print('train dataset: {}'.format(train_folder))

    return train_folder, target_name


def get_dataset(poison_type, poison_or_benign, poison_rate, batch_size, number_classes, num_workers, transforms=False, num_ADiter=None, source_name=None, source_class=None):
    train_folder, target_name = get_dataset_info(poison_type, poison_or_benign, poison_rate, source_name, num_ADiter)

    if poison_type == 'adaptivecifar10' or poison_type == 'pattern' or poison_type == 'blto' or 'freq' in poison_type or poison_type == 'pattern' or poison_type == 'wanet':
        tr_ds = ImageData(root='poisonDataset/{}/{}'.format(poison_type, train_folder), transform=transforms)
        tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ts_ds = ImageData(root='poisonDataset/{}/Test'.format(poison_type))
        ts_dl = DataLoader(dataset=ts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        target_transforms = ToTargetClass(target_name=target_name, num_classes=number_classes, poison_type=poison_type)
        pts_ds = ImageData(root='poisonDataset/{}/poisonTest'.format(poison_type), target_transform=target_transforms, source_class=source_class)
        pts_dl = DataLoader(dataset=pts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        tr_ds = AudioData(data_folder='poisonDataset/{}/{}'.format(poison_type, train_folder), number_class=number_classes, transforms=transforms)
        tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ts_ds = AudioData(data_folder='poisonDataset/{}/Test'.format(poison_type), number_class=number_classes)
        ts_dl = DataLoader(dataset=ts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        target_transforms = ToTargetClass(target_name=target_name, num_classes=number_classes, poison_type=poison_type)
        pts_ds = AudioData(data_folder='poisonDataset/{}/poisonTest'.format(poison_type), number_class=number_classes,
                           target_transform=target_transforms)
        pts_dl = DataLoader(dataset=pts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return tr_dl, ts_dl, pts_dl, train_folder


def get_dataset2(poison_type, poison_or_benign, poison_rate, batch_size, number_classes, num_workers, transforms=False, num_ADiter=None, source_name=None, source_class=None, model_name=None):
    train_folder, target_name = get_dataset_info(poison_type, poison_or_benign, poison_rate, source_name, num_ADiter)

    if 'adaptivecifar10' in poison_type or poison_type == 'pattern' or poison_type == 'blto' or 'freq' in poison_type \
            or 'adaptiveattack' in poison_type or 'wanet' in poison_type:
        tr_ds = ImageData2(root='poisonDataset/{}/{}'.format(poison_type, train_folder), transform=transforms, model_name=model_name)
        tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ts_ds = ImageData2(root='poisonDataset/{}/Test'.format(poison_type), model_name=model_name)
        ts_dl = DataLoader(dataset=ts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        target_transforms = ToTargetClass(target_name=target_name, num_classes=number_classes, poison_type=poison_type)
        pts_ds = ImageData2(root='poisonDataset/{}/poisonTest'.format(poison_type), target_transform=target_transforms, source_class=source_class, model_name=model_name)
        pts_dl = DataLoader(dataset=pts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        tr_ds = AudioData2(data_folder='poisonDataset/{}/{}'.format(poison_type, train_folder), number_class=number_classes, transforms=transforms)
        tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ts_ds = AudioData2(data_folder='poisonDataset/{}/Test'.format(poison_type), number_class=number_classes)
        ts_dl = DataLoader(dataset=ts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        target_transforms = ToTargetClass(target_name=target_name, num_classes=number_classes, poison_type=poison_type)
        pts_ds = AudioData2(data_folder='poisonDataset/{}/poisonTest'.format(poison_type), number_class=number_classes,
                           target_transforms=target_transforms)
        pts_dl = DataLoader(dataset=pts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return tr_dl, ts_dl, pts_dl, train_folder

def get_batch_val(val_dl):
    # supervised learning
    try:
        batch_val = next(iter(val_dl))
    except StopIteration:
        val_dl = DataLoader(val_dl.dataset, batch_size=len(val_dl.dataset), shuffle=True,
                            num_workers=val_dl.num_workers)
        batch_val = next(iter(val_dl))
    while len(set(batch_val[1].tolist())) != len(
            val_dl.dataset.dataset.classes):  # ensure there is at least one element for each class
        batch_val = next(iter(val_dl))

    return batch_val


def eval_cluster(model, tr_dl, device='cpu', epoch=-1):
    # ground-truth class
    ground_c_id, pred_c_id = [], []
    model.eval()
    for batch in tr_dl:
        data, labels = batch[0], batch[1]
        ground_c_id.append(labels.cpu())
        embedding, feature_x_C = model(data.to(device), prototype_flag=True)

        cost = feature_x_C.detach()
        # get assignments
        q = distributed_sinkhorn(
            cost)  # [-bs:] since the distributed_sinkhorn output is [bs, K], the [-bs:] is useless.
        # hard assignment
        _, max_indices = torch.max(q, dim=1, keepdim=True)
        pred_c_id.append(max_indices.cpu())

    ground_c_id, pred_c_id = torch.cat(ground_c_id).numpy(), torch.cat(pred_c_id)[:, 0].numpy()
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(ground_c_id.ravel(), pred_c_id.ravel())

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(ground_c_id, pred_c_id)
    plt.savefig('log/confusion_matrix_ari_{:.3f}_epoch_{}.png'.format(ari, epoch))
    print(ari)



def distributed_sinkhorn(out, epsilon=0.05, sinkhorn_iterations = 3):
    if out.requires_grad == True:
        epsilon = epsilon
        sinkhorn_iterations = sinkhorn_iterations
        Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # * args.world_size # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q = Q / sum_Q

        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q = Q / sum_of_rows
            Q = Q / K

            # normalize each column: total weight per sample must be 1/B
            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            Q = Q / B

        Q = Q * B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
    else:
        with torch.no_grad():
            epsilon = epsilon
            sinkhorn_iterations = sinkhorn_iterations
            Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
            B = Q.shape[1] # * args.world_size # number of samples to assign
            K = Q.shape[0] # how many prototypes

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            # dist.all_reduce(sum_Q)
            Q /= sum_Q

            for it in range(sinkhorn_iterations):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                # dist.all_reduce(sum_of_rows)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B # the colomns must sum to 1 so that Q is an assignment
            return Q.t()


def get_validation_data(tr_ds, num_sample, batch_size, num_workers, cache_subset_files=None):
    # num_val = int(percentage * len(tr_ds))
    if cache_subset_files == None:
        subset_indics = []
        for class_i in range(len(tr_ds.classes)):
            cnt = 0
            for i, label in enumerate(tr_ds.labels):
                if label == class_i:
                    if i in tr_ds.benign_indics: # labeled dara from benign samples
                        subset_indics.append(i)
                        cnt += 1
                    else:
                        pass

                if cnt == num_sample:
                    break
    else:
        cache_subset_files = np.array([file_name.replace('Train/', '') for file_name in np.load(cache_subset_files)])
        subset_indics = [index for index, file_class in enumerate(tr_ds.samples) if file_class[0].split('Train/')[1] in cache_subset_files]

    subset_indics_left = []
    for i in range(len(tr_ds)):
        if i in subset_indics:
            continue
        else:
            subset_indics_left.append(i)

    val_ds = Subset(tr_ds, subset_indics)
    val_dl = DataLoader(dataset=val_ds, batch_size=len(val_ds), shuffle=True, num_workers=num_workers)

    return val_dl, subset_indics, subset_indics_left


class ToTargetClass(object):
    def __init__(self, target_name, num_classes, poison_type):
        if 'adaptivecifar10' in poison_type or poison_type == 'blto' or 'freq' in poison_type \
                or poison_type == 'pattern' or 'adaptiveattack' in poison_type or 'wanet' in poison_type:
            self.target_class = target_name
        elif poison_type == 'corruptencoder' or poison_type == 'depud' or poison_type == 'adp_corrupt':
            self.target_class = classes_10_imagenet.index(target_name)
        else:
            if num_classes == 10:
                classes = classes_10
            else:
                classes = classes_30
            self.target_class = classes.index(target_name)
            print('target {} with id {}'.format(target_name, self.target_class))

    def __call__(self, input_tensor):
        # Perform transformation to convert input_tensor to target_class
        transformed_tensor = np.ones_like(input_tensor) * self.target_class  # Example transformation

        return transformed_tensor


def scan_datafolder(datafolder, classes, target_transforms=None):
    files, labels = [], []
    for subfolder in os.listdir(datafolder):
        if subfolder in classes:
            cnt = 0
            for file in os.listdir(os.path.join(datafolder, subfolder)):
                files.append(os.path.join(datafolder, subfolder, file))
                label = classes.index(subfolder)
                labels.append(label)
                cnt += 1
                # if cnt > 200:
                #     break

    files, labels = np.array(files), np.array(labels)
    if target_transforms != None:
        files, labels = files[labels != target_transforms.target_class], labels[
            labels != target_transforms.target_class]

    return files, labels


def poison_indics(files_list):
    indics_list = []
    for i, file in enumerate(files_list):
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] in classes_30:
            indics_list.append(i)

    return indics_list


def load_data(files, method='torchaudio'):
    data = []
    for file in tqdm(files):
        if method == 'torchaudio':
            signal, sample_rate = torchaudio.load(file)
        else:
            signal, sample_rate = librosa.load(file, sr=None)
        data.append(signal)
        # if len(data) > 10: # debugging
        #     break
    if method == 'torchaudio':
        data = torch.cat(data)
    else:
        data = torch.from_numpy(np.stack(data))

    return data, sample_rate


def save_spectrum(spectrum, save_name):
    db_transform = T.AmplitudeToDB(stype='power')
    spectrogram_db = db_transform(spectrum)
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[:, :].numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title('Spectrogram (dB)')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.tight_layout()
    # Save the figure
    plt.savefig(save_name)
    plt.close()


def save_spectrum_from_wav(save_name, wav_data):
    n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
    spectrum_trans = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    spectrum = spectrum_trans(wav_data)
    db_transform = T.AmplitudeToDB(stype='power')
    spectrogram_db = db_transform(spectrum)
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[:, :].detach().numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title('Spectrogram (dB)')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.tight_layout()
    # Save the figure
    plt.savefig(save_name)
    plt.close()

# @torch.no_grad()
# def augmentTime(wav_data, sample_rate, debugging_flag=False, device='cpu'):
#     wav_data = torch.unsqueeze(wav_data, 0).to(device)
#     # wav_data = wav_data.repeat(128, 1)
#     # pitch shift too time-consuming
#     # n_steps = torch.randint(-10, 10, (1,)).item()
#
#     # # low pass filter should not be used when the defence has zero knoweldge on the trigger
#     # # cutoff_freq = random.choice([14000, 22000])
#     # wav_data_lp = F.lowpass_biquad(wav_data, sample_rate, cutoff_freq=14000)
#
#     n_steps = torch.randint(-12, 12, (1,)).item()
#
#     #  librosa-based pitch shifting
#     # stime = time.time()
#     wav_data_ps = torch.tensor(librosa.effects.pitch_shift(wav_data.numpy(), sr=sample_rate, n_steps=n_steps))
#     # print('librosa {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_librosa', wav_data_ps[0])
#     #
#     # stime = time.time()
#     # wav_data_ps = F.pitch_shift(wav_data, sample_rate, n_steps=n_steps)
#     # print('torchaudio {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_torchaudio', wav_data_ps[0].to('cpu'))
#     # stime = time.time()
#     # transform = PitchShift(
#     #     min_semitones=n_steps,
#     #     max_semitones=n_steps,
#     #     p=1.0
#     # )
#     # stime = time.time()
#     # wav_data_ps = torch.tensor(transform(wav_data.detach().numpy(), sample_rate=sample_rate))
#     # print('audiomentations {:.5f} s'.format(time.time() - stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_audiomentations', wav_data_ps[0])
#
#     # stime = time.time()
#     # pedalboard_ps = pedalboard.PitchShift(semitones=n_steps)
#     # wav_data_ps = torch.tensor(pedalboard_ps(wav_data.detach().numpy(), sample_rate))
#     # print('pedalboard {:.5f} s'.format(time.time() - stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_pedalboard', wav_data_ps[0])
#
#     # print('audiomentations {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_audiomentations', wav_data_ps[0])
#
#     # stime = time.time()
#     # wav_data shape should be [batch, channel, samples]
#     # wav_data_ps = pitch_shift(wav_data.unsqueeze(1), n_steps, sample_rate).squeeze()
#     # print('torch_pitch_shift cpu {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_torch_pitch_shift', wav_data_ps[0])
#     #
#     # from torch_pitch_shift import pitch_shift
#     # stime = time.time()
#     # wav_data_ps = pitch_shift(wav_data.unsqueeze(0).to('cuda:0'), n_steps, sample_rate).to('cpu')[0]
#     # print('torch_pitch_shift cuda {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_pitchshift_spectrum_torch_pitch_shift_cuda', wav_data_ps[0])
#
#     # # time streching
#     stretch_score = 0.1 * torch.randint(8, 12, (1,)).item()
#     # while stretch_score == 1:
#     #     stretch_score = 0.1 * torch.randint(8, 12, (1,)).item()
#     time_stretch = T.Speed(sample_rate, stretch_score).to(device)
#     wav_data_ts, _ = time_stretch(wav_data_ps)
#     time_stretch.to('cpu')
#     # librosa-based time stretching
#     #
#     #
#     # stime = time.time()
#     # wav_data_ts = torch.tensor(librosa.effects.time_stretch(wav_data_ps.numpy(), rate=stretch_score))
#     # if wav_data_ts.shape[1] < wav_data.shape[1]:
#     #     pad_length =  wav_data.shape[1] - wav_data_ts.shape[1]
#     #     # Pad with zeros
#     #     wav_data_ts = torch.nn.functional.pad(wav_data_ts, (0, pad_length), mode='constant', value=0)
#     # if wav_data_ts.shape[1] > wav_data.shape[1]:
#     #     wav_data_ts = wav_data_ts[:,:wav_data.shape[1]]
#     # print('time stretching librosa {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_timestretch_spectrum_librosa', wav_data_ts[0])
#
#     # stime = time.time()
#     # transform_ts = TimeStretch(
#     #     min_rate=0.8,
#     #     max_rate=1.25,
#     #     leave_length_unchanged=True,
#     #     p=1.0
#     # )
#     # wav_data_ts = torch.tensor(transform_ts(wav_data_ps.numpy(), sample_rate=sample_rate))
#     if wav_data_ts.shape[1] < wav_data.shape[1]:
#         pad_length =  wav_data.shape[1] - wav_data_ts.shape[1]
#         # Pad with zeros
#         wav_data_ts = torch.nn.functional.pad(wav_data_ts, (0, pad_length), mode='constant', value=0)
#     if wav_data_ts.shape[1] > wav_data.shape[1]:
#         wav_data_ts = wav_data_ts[:,:wav_data.shape[1]]
#     # print('time stretching audiomentations {:.5f} s'.format(time.time()-stime))
#     # save_spectrum_from_wav('after_timestretch_spectrum_audiomentations', wav_data_ts[0])
#
#     # random gain
#     gain_score = torch.randint(2, 6, (1,)).to(device)
#     wav_data_rg = wav_data_ts * gain_score
#
#     # background noise should use random noise
#     # backdground_path = os.path.join(os.getcwd().split('DefTimeSeries')[0], 'DefTimeSeries', 'lib', 'background')
#     # noise, original_sr = torchaudio.load(os.path.join(backdground_path,  random.choice(os.listdir(backdground_path))))
#     # noise = F.resample(noise, orig_freq=original_sr, new_freq=sample_rate)
#     # noise = noise[:, :wav_data.shape[1]]
#     noise = torch.randn(wav_data.size()).to(device)
#     snr_dbs = torch.randint(3, 10, (1,)).to(device)
#     wav_data_bg = F.add_noise(wav_data_rg, noise, snr_dbs)
#
#     # transform = AddGaussianSNR(
#     #     min_snr_db=3.0,
#     #     max_snr_db=10.0,
#     #     p=1.0
#     # )
#     #
#     # augmented_sound = transform(wav_data_rg.numpy(), sample_rate=16000)
#     # del wav_data_rg,  wav_data_ps, wav_data, pitch_shift, time_stretch
#     # torch.cuda.empty_cache()
#     if debugging_flag:
#         torchaudio.save('original.wav', wav_data, sample_rate[0])
#         save_spectrum_from_wav('original_spectrum', wav_data[0])
#         torchaudio.save('after_pitchshift.wav', wav_data_ps, sample_rate[0])
#         save_spectrum_from_wav('after_pitchshift_spectrum', wav_data_ps[0])
#         torchaudio.save('after_timestratch.wav', wav_data_ts, sample_rate[0])
#         save_spectrum_from_wav('after_timestratch_spectrum', wav_data_ts[0])
#         # torchaudio.save('after_lowpass.wav', wav_data_lp, sample_rate)
#         # torchaudio.save('after_pitchShift.wav', wav_data_ps, sample_rate)
#         torchaudio.save('after_randomgain.wav', wav_data_rg, sample_rate[0])
#         save_spectrum_from_wav('after_randomgain_spectrum', wav_data_rg[0])
#         torchaudio.save('after_noise.wav', wav_data_bg, sample_rate[0])
#         save_spectrum_from_wav('after_noise_spectrum', wav_data_bg[0])
#
#     return torch.squeeze(wav_data_bg)

#
#
# def extract_MFCC(data, sample_rate):
#     data = data.numpy()
#     n_mfcc, n_fft, hop_length = 40, 1103, 441  # parameters from original paper
#     mfcc = librosa.feature.mfcc(y=data, sr=sample_rate,
#                                 n_mfcc=n_mfcc, n_fft=n_fft,
#                                 hop_length=hop_length)
#     mfcc = torch.tensor(np.transpose(mfcc, axes=[0, 2, 1]), dtype=torch.float32)
#
#     return mfcc

#
# class AudioData(Dataset):
#     def __init__(self, data_folder, number_class, transforms=None, target_transforms=None):
#         self.data_folder = data_folder
#         assert number_class in [10, 30], 'number_class should be 10 or 30'
#         self.classes = classes_10 if number_class==10 else classes_30
#         self.transforms = transforms
#         self.files, self.labels = scan_datafolder(data_folder, self.classes, target_transforms)
#         self.data, self.sr = load_data(self.files)
#         self.mfcc = extract_MFCC(self.data, self.sr)
#         self.target_transforms = target_transforms
#
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, item):
#         mfcc, label = self.mfcc[item], self.labels[item]
#
#         return mfcc, label
#

class AudioData(Dataset):
    def __init__(self, data_folder, number_class, transforms=None, target_transform=None):
        self.data_folder = data_folder
        assert number_class in [10, 30], 'number_class should be 10 or 30'
        if number_class == 2:
            self.classes = classes_2
        elif number_class == 10:
            self.classes = classes_10
        else:
            self.classes = classes_30
        self.files, self.labels = scan_datafolder(data_folder, self.classes, target_transform)
        self.benign_indics = get_benign_indics(zip(self.files, self.labels), data_folder)
        # self.poison_indics = poison_indics(self.files)
        self.data, self.sr = load_data(self.files, method='torchaudio')
        self.transforms = transforms
        if transforms != None:
            # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
            self.spec_aug = spec_aug_audio
            self.time_aug = time_aug_audio

        self.target_transform = target_transform

        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)



    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # give the wav file and generate two augmented files
        # finally output [aug1, aug2, label]
        data, label = self.data[index], self.labels[index]
        if self.transforms == None:
            # Convert to power spectrogram
            spectrum = self.spectrum(data)
        else:
            data = torch.tensor(self.time_aug(data.numpy(), self.sr))
            spectrum = self.spectrum(data)
            # spectrum = self.spec_aug(spectrum)
        # Convert to mel-scale
        melspectrum = self.mel_scale(spectrum)
        melspectrum = self.amplitude_to_DB(melspectrum)
        mfcc = torch.matmul(melspectrum.transpose(-1, -2), self.dct_mat)

        if self.target_transform != None:
            label = self.target_transform(label)

        return mfcc, label


class AudioData2(Dataset):
    def __init__(self, data_folder, number_class, transforms=None, target_transforms=None):
        self.data_folder = data_folder
        assert number_class in [10, 30], 'number_class should be 10 or 30'
        if number_class == 2:
            self.classes = classes_2
        elif number_class == 10:
            self.classes = classes_10
        else:
            self.classes = classes_30
        self.files, self.labels = scan_datafolder(data_folder, self.classes, target_transforms)

        self.target_transforms = target_transforms
        # filter out the target class when target_transform is not None
        if target_transforms != None:
            self.files = np.array([file_name for file_name, file_label in zip(self.files, self.labels) if file_label != target_transforms.target_class])
            self.labels = np.array([file_label for file_name, file_label in zip(self.files, self.labels) if file_label != target_transforms.target_class])

        self.benign_indics = get_benign_indics(zip(self.files, self.labels), data_folder)
        # self.poison_indics = poison_indics(self.files)
        self.data, self.sr = load_data(self.files, method='torchaudio')
        self.transforms = transforms
        if transforms != None:
            # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
            self.spec_aug = spec_aug_audio
            self.time_aug = time_aug_audio



        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # give the wav file and generate two augmented files
        # finally output [aug1, aug2, label]
        data, label = self.data[index], self.labels[index]
        if self.transforms == None:
            # Convert to power spectrogram
            spectrum = self.spectrum(data)
        else:
            data = torch.tensor(self.time_aug(data.numpy(), self.sr))
            spectrum = self.spectrum(data)
            spectrum = self.spec_aug(spectrum)
        # Convert to mel-scale
        melspectrum = self.mel_scale(spectrum)
        melspectrum = self.amplitude_to_DB(melspectrum)
        mfcc = torch.matmul(melspectrum.transpose(-1, -2), self.dct_mat)
        poison_flag = 0 if index in self.benign_indics else 1

        if self.target_transforms != None:
            label = self.target_transforms(label)

        return mfcc, label, poison_flag, index


def get_benign_indics(samples, root):
    if 'poisonDataset' in root:
        attack_name = root.split('poisonDataset')[1].split('/')[1]
    else:
        attack_name = 'adp_corrupt' # bilevelOpt.py
    # if 'adaptivecifar10' in root:
    #     if os.path.exists(os.path.join(root, 'poison_file.npy')):
    #         poison_files = np.load(os.path.join(root, 'poison_file.npy')).tolist()
    #     else:
    #         poison_files = []
    #     if os.path.exists(os.path.join(root, 'cover_file.npy')):
    #         cover_files = np.load(os.path.join(root, 'cover_file.npy')).tolist()
    #     else:
    #         cover_files = []
    #     benign_indics = []
    #     for i, (path, _) in enumerate(samples):
    #         path = path.split(attack_name)[1][1:]
    #         if path in poison_files or path in cover_files:
    #             pass
    #         else:
    #             benign_indics.append(i)
    #
    #     return benign_indics
    # else:
    if os.path.exists(os.path.join(root, 'poison_file.npy')):
        poison_files = np.load(os.path.join(root, 'poison_file.npy')).tolist()
    else:
        poison_files = []
    cover_files = []
    benign_indics = []
    for i, (path, _) in enumerate(samples):
        if attack_name == 'adp_corrupt':
            if path in poison_files or path in cover_files:
                pass
            else:
                benign_indics.append(i)
        else:
            path = path.split(attack_name)[1][1:]
            if path in poison_files or path in cover_files:
                pass
            else:
                benign_indics.append(i)

    return benign_indics


def vis_sample(tensor_img):
    # The mean and std used for normalization
    mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
    std = [63.0 / 255, 62.1 / 255, 66.7 / 255]

    # Inverse normalization values
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]

    # Define the inverse normalization transform
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    # Apply the inverse normalization
    unnormalized_img = inv_normalize(tensor_img)

    # Convert the tensor back to a PIL image
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(unnormalized_img)

    # Now pil_img is the unnormalized PIL image
    pil_img.save('train_sample.png')  # To view the image


class ImageData(ImageFolder):
    def __init__(self, root, transform=False, target_transform=None, loader=Image.open, source_class=None):
        super(ImageData, self).__init__(root, transform=None, target_transform=None, loader=loader)
        self.labels = self.targets # to keep consistnace with AudioData class
        if transform != False:
            if transform != True:
                self.transform = transform
            else:
                if 'depud' in root or 'corruptencoder' in root or 'adp_corrupt' in root:
                    self.transform = image_waug_imagenet_freeMatch
                # elif 'blto' in root:
                #     self.transform = image_waug_cifar_freeMatch_blto
                else: # adaptive_attack or blto or freq
                    self.transform = image_waug_cifar_freeMatch
        else:
            if 'depud' in root or 'corruptencoder' in root or 'adp_corrupt' in root:
                self.transform = image_no_aug_imagenet_freeMatch
            # elif 'blto' in root:
            #     self.transform = image_no_aug_cifar_freeMatch_blto
            else: # adaptive_attack or blto or freq
                self.transform = image_no_aug_cifar_freeMatch


        self.target_transform = target_transform
        self.benign_indics = get_benign_indics(self.samples, root)
        if source_class != None:
            self.samples = [(path, target) for (path, target) in self.samples if target == source_class]


    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return additional information if needed.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert('RGB')
        sample = self.transform(sample)

        # # visualizatrion
        # vis_sample(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageData2(ImageFolder):
    def __init__(self, root, transform=False, target_transform=None, loader=Image.open, source_class=None, model_name=None):
        super(ImageData2, self).__init__(root, transform=None, target_transform=None, loader=loader)
        self.labels = self.targets # to keep consistnace with AudioData class
        if transform != False:
            if transform != True:
                self.transform = transform
            else:
                if 'depud' in root or 'corruptencoder' in root or 'adp_corrupt' in root:
                    self.transform = image_waug_imagenet_freeMatch
                # elif 'blto' in root:
                #     self.transform = image_waug_cifar_freeMatch_blto
                else: # adaptive_attack or blto or freq
                    if model_name != 'transformer':
                        self.transform = image_waug_cifar_freeMatch
                    else:
                        print(model_name)
                        self.transform = image_waug_cifar_freeMatch_224
        else:
            if 'depud' in root or 'corruptencoder' in root or 'adp_corrupt' in root:
                self.transform = image_no_aug_imagenet_freeMatch
            # elif 'blto' in root:
            #     self.transform = image_no_aug_cifar_freeMatch_blto
            else: # adaptive_attack or blto or freq
                if model_name != 'transformer':
                    self.transform = image_no_aug_cifar_freeMatch
                else:
                    print(model_name)
                    self.transform = image_no_aug_cifar_freeMatch_224


        self.target_transform = target_transform
        # filter out the target class when target_transform is not None
        if target_transform != None:
            self.samples = [(path, target) for (path, target) in self.samples if target != target_transform.target_class]

        self.benign_indics = get_benign_indics(self.samples, root)
        if source_class != None:
            self.samples = [(path, target) for (path, target) in self.samples if target == source_class]


    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return additional information if needed.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert('RGB')
        sample = self.transform(sample)

        poison_flag = 0 if index in self.benign_indics else 1
        # # visualizatrion
        # vis_sample(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, poison_flag, index


class ImageDataSSL(ImageFolder):
    def __init__(self, root, number_aug, benign_indics, loader=Image.open, model_name='None', adaptive=False):
        super(ImageDataSSL, self).__init__(root, transform=None, target_transform=None, loader=loader)
        self.number_aug = number_aug
        self.benign_indics = benign_indics

        # self.transforms = image_waug_cifar_freeMatch
        # self.transforms_noaug = image_no_aug_cifar_freeMatch
        if 'depud' in root or 'corruptencoder' in root or 'adp_corrupt' in root :
            self.image_waug_cifar = image_waug_imagenet_freeMatch
            self.image_saug_cifar = image_saug_imagenet_freeMatch
        # elif 'blto' in root:
        #     self.image_waug_cifar = image_waug_cifar_freeMatch_blto
        #     self.image_saug_cifar = image_saug_cifar_freeMatch_blto
        else: # adaptivecifar10 and blto and adaptiveattack
            if model_name!='transformer':
                self.image_waug_cifar = image_waug_cifar_freeMatch # image_no_aug_cifar_freeMatch
                self.image_saug_cifar = image_saug_cifar_freeMatch
            else:
                self.image_waug_cifar = image_waug_cifar_freeMatch_224  # image_no_aug_cifar_freeMatch
                self.image_saug_cifar = image_saug_cifar_freeMatch_224

        if adaptive == True:
            self.image_waug_cifar = image_no_aug_cifar_freeMatch_tensor  # image_no_aug_cifar_freeMatch
            self.image_saug_cifar = image_no_aug_cifar_freeMatch_tensor

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return additional information if needed.
        """
        poison_flag = 0 if index in self.benign_indics else 1

        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert('RGB')
        aug_l = []
        for i in range(self.number_aug):
            if i == 0 or i == 1:
                sample_aug = self.image_waug_cifar(sample)
            else:
                sample_aug = self.image_saug_cifar(sample)

            aug_l.append(sample_aug)

        aug_l.append(np.array(target))
        aug_l.append(np.array(poison_flag))
        aug_l.append(np.array(index))

        return aug_l

#
# class AudioDataSSL(Dataset):
#     def __init__(self, data_folder, number_class, target_transforms=None):
#         self.data_folder = data_folder
#         assert number_class in [10, 30], 'number_class should be 10 or 30'
#         if number_class == 2:
#             self.classes = classes_2
#         elif number_class == 10:
#             self.classes = classes_10
#         else:
#             self.classes = classes_30
#         self.files, self.labels = scan_datafolder(data_folder, self.classes, target_transforms)
#         self.poison_indics = poison_indics(self.files)
#         self.data, self.sr = load_data(self.files, method='torchaudio')
#         self.target_transforms = target_transforms
#
#         # mfcc convert
#         n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
#         self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
#         self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
#         self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
#         self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)
#
#         # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
#         self.spec_aug = torch.nn.Sequential(
#             T.FrequencyMasking(freq_mask_param=int(552.0/5)),
#             T.TimeMasking(time_mask_param=int(100./5)),
#         )
#
#     def __len__(self):
#         return self.labels.shape[0]
#
#     def __getitem__(self, index):
#         # give the wav file and generate two augmented files
#         # finally output [aug1, aug2, label]
#         data, label = self.data[index], self.labels[index]
#         # time domain augmentation
#         # augt_1, augt_2 = augmentTime(data, self.sr, debugging_flag=False), augmentTime(data, self.sr)
#         augt_1, augt_2 = data, augmentTime(data, self.sr)
#         # Convert to power spectrogram
#         spectrum1, spectrum2 = self.spectrum(augt_1), self.spectrum(augt_2)
#         augf_1, augf_2 = spectrum1, self.spec_aug(spectrum2)
#         # augf_1, augf_2 = self.spec_aug(spectrum1), self.spec_aug(spectrum2)
#
#         # Convert to mel-scale
#         melspectrum1, melspectrum2 = self.mel_scale(augf_1), self.mel_scale(augf_2)
#         melspectrum1, melspectrum2 = self.amplitude_to_DB(melspectrum1), self.amplitude_to_DB(melspectrum2)
#         mfcc1 = torch.matmul(melspectrum1.transpose(-1, -2), self.dct_mat)
#         mfcc2 = torch.matmul(melspectrum2.transpose(-1, -2), self.dct_mat)
#
#         if self.target_transforms != None:
#             label = self.target_transforms(label)
#
#         return mfcc1, mfcc2, label



class AudioDataSSL(Dataset):
    def __init__(self, data_folder, number_class, number_aug, benign_indics, debugging_flag=False):
        self.data_folder = data_folder
        self.benign_indics = benign_indics
        assert number_aug >=2, 'number augmentation should >= 2'
        assert number_class in [2, 10, 30], 'number_class should be 10 or 30'
        if number_class == 2:
            self.classes = classes_2
        elif number_class == 10:
            self.classes = classes_10
        else:
            self.classes = classes_30
        self.files, self.labels = scan_datafolder(data_folder, self.classes)
        self.poison_indics = poison_indics(self.files)
        self.data, self.sr = load_data(self.files, method='torchaudio')
        self.number_aug = number_aug
        self.debugging_flag = debugging_flag

        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)

        # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
        self.spec_aug = spec_aug_audio
        # self.time_aug = time_aug_audio

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # give the wav file and generate two augmented files
        # finally output [aug1, aug2, label]
        poison_flag = 0 if index in self.benign_indics else 1
        data, label = self.data[index], self.labels[index]
        aug_l = []
        # spectrum0, spectrum1 = None, None

        spectrum = self.spectrum(data)

        for i in range(self.number_aug):
            spectrum_aug = self.spec_aug(spectrum)
            # if i == 0:
            #     # time_aug0 = torch.tensor(self.time_aug(data.numpy(), self.sr))
            #     # spectrum0 = self.spectrum(time_aug0)
            #     spectrum0 = self.spectrum(data)
            #     spectrum_aug = spectrum0
            # elif i == 1:
            #     # stime = time.time()
            #     time_aug1 = torch.tensor(self.time_aug(data.numpy(), self.sr))
            #     # time_aug1 = augmentTime(data, self.sr)
            #
            #     # print('time aug {} s'.format(time.time()-stime))
            #     # stime = time.time()
            #     spectrum1 = self.spectrum(time_aug1)
            #     spectrum_aug = spectrum1
            #     # print('spectrum calculation {} s'.format(time.time()-stime))
            #     # stime = time.time()
            #     # spectrum_aug = spectrum
            #     # spectrum_aug = self.spec_aug(spectrum)
            #     # print('spectrum aug {} s'.format(time.time()-stime))
            #
            #     # if self.debugging_flag == True:
            #     #     torchaudio.save('original.wav', torch.unsqueeze(data, 0), self.sr)
            #     #     save_spectrum_from_wav('original_spectrum', data)
            #     #     torchaudio.save('time_aug.wav', torch.unsqueeze(time_aug, 0), self.sr)
            #     #     save_spectrum_from_wav('time_aug_spectrum', time_aug)
            #     #     save_spectrum(spectrum, 'before_spectrum_aug.jpg')
            #     #     save_spectrum(spectrum_aug, 'after_spectrum_aug.jpg')
            # else:
            #     spectrum = spectrum1 if i % 2 == 1 else spectrum0
            #     spectrum_aug = self.spec_aug(spectrum)

            # stime = time.time()
            melspectrum_aug = self.amplitude_to_DB(self.mel_scale(spectrum_aug))
            # print('melspectrum aug {} s'.format(time.time()-stime))
            # stime = time.time()
            mfcc = torch.matmul(melspectrum_aug.transpose(-1, -2), self.dct_mat)
            # print('mfcc {} s'.format(time.time()-stime))

            aug_l.append(mfcc)

        aug_l.append(label)
        aug_l.append(poison_flag)
        aug_l.append(index)

        return aug_l


#
# def augment_data(data_index, data, labels, which_number, sr, debugging_flag, spectrum, spec_aug, amplitude_to_DB, mel_scale, dct_mat):
#     print('sub proccess: {}'.format(which_number))
#     # Extract the specific item
#     data_item, label = data[data_index], labels[data_index]
#     aug_results = []
#
#     # Perform augmentation
#     time_aug = augmentTime(data_item, sr, debugging_flag=debugging_flag)
#     spectrum_result = spectrum(time_aug)
#     spectrum_augmented = spec_aug(spectrum_result)
#
#     if debugging_flag:
#         save_spectrum(spectrum_result, f'before_spectrum_aug_{data_index}.jpg')
#         save_spectrum(spectrum_augmented, f'after_spectrum_aug_{data_index}.jpg')
#
#     melspectrum_aug = amplitude_to_DB(mel_scale(spectrum_augmented))
#     mfcc = torch.matmul(melspectrum_aug.transpose(-1, -2), dct_mat)
#     aug_results.append(mfcc)
#     print('sub proccess: {} finish'.format(which_number))
#
#     return aug_results
#
#
# class AudioDataSWAVMP(AudioDataSWAV):
#     def __init__(self, data_folder, number_class, number_aug, debugging_flag=False):
#         super().__init__(data_folder, number_class, number_aug, debugging_flag)
#
#     def __getitem__(self, index):
#         # Set up multiprocessing
#         with mp.Pool(processes=mp.cpu_count()) as pool:
#             results = pool.starmap(augment_data, [(index, self.data, self.labels, which_number, self.sr,
#                                                    self.debugging_flag, self.spectrum, self.spec_aug,
#                                                    self.amplitude_to_DB, self.mel_scale, self.dct_mat) for which_number in
#                                                   range(self.number_aug)])
#
#         # Flatten the list of results
#         results = [item for sublist in results for item in sublist]
#         return results
#
#
#
#
#
# def augment_dataMT(data_index, data, labels, which_number, sr, debugging_flag, spectrum, time_aug, spec_aug, amplitude_to_DB,
#                  mel_scale, dct_mat):
#     # print(f'sub thread: {which_number}')
#     data_item, label = data[data_index], labels[data_index]
#     aug_results = []
#
#     # Perform augmentation
#     # time_aug = augmentTime(data_item, sr, debugging_flag=debugging_flag)
#     time_aug = torch.tensor(time_aug(data_item.numpy(), sr))
#     spectrum_result = spectrum(time_aug)
#     spectrum_augmented = spec_aug(spectrum_result)
#
#     if debugging_flag:
#         save_spectrum(spectrum_result, f'before_spectrum_aug_{data_index}.jpg')
#         save_spectrum(spectrum_augmented, f'after_spectrum_aug_{data_index}.jpg')
#
#     melspectrum_aug = amplitude_to_DB(mel_scale(spectrum_augmented))
#     mfcc = torch.matmul(melspectrum_aug.transpose(-1, -2), dct_mat)
#     aug_results.append(mfcc)
#     # print(f'sub thread: {which_number} finish')
#
#     return aug_results
#
#
# class AudioDataSWAVMT(AudioDataSWAV):
#     def __init__(self, data_folder, number_class, number_aug, debugging_flag=False):
#         super().__init__(data_folder, number_class, number_aug, debugging_flag)
#
#     def __getitem__(self, index):
#         # Setup multithreading
#         results = []
#         with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers=None uses as many as the system allows
#             future_to_aug = {executor.submit(augment_dataMT, index, self.data, self.labels, which_number, self.sr,
#                                              self.debugging_flag, self.spectrum, self.time_aug, self.spec_aug,
#                                              self.amplitude_to_DB, self.mel_scale, self.dct_mat): which_number for
#                              which_number in range(self.number_aug)}
#
#             for future in as_completed(future_to_aug):
#                 results.extend(future.result())
#
#         return results

#
# class AudioDataSWAVMC(Dataset):
#     def __init__(self, data_folder, number_class, number_aug, debugging_flag=False):
#         self.data_folder = data_folder
#         assert number_aug >=2, 'number augmentation should >= 2'
#         assert number_class in [2, 10, 30], 'number_class should be 10 or 30'
#         if number_class == 2:
#             self.classes = classes_2
#         elif number_class == 10:
#             self.classes = classes_10
#         else:
#             self.classes = classes_30
#         self.files, self.labels = scan_datafolder(data_folder, self.classes)
#         self.poison_indics = poison_indics(self.files)
#         self.data, self.sr = load_data(self.files, method='torchaudio')
#         self.number_aug = number_aug
#         self.debugging_flag = debugging_flag
#
#         # mfcc convert
#         n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
#         self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
#         self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
#         self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
#         self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)
#
#         # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
#         self.spec_aug = torch.nn.Sequential(
#             T.FrequencyMasking(freq_mask_param=int(552.0/20)),
#             T.TimeMasking(time_mask_param=int(100.0/20)),
#         )
#         self.time_aug = Compose([
#             PitchShift(min_semitones=-5, max_semitones=5, p=1.0),
#             TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=1.0),
#             AddGaussianSNR(min_snr_db=1, max_snr_db=3, p=1.0),
#         ])
#
#     def __len__(self):
#         return self.labels.shape[0]
#
#     def __getitem__(self, index):
#         # give the wav file and generate two augmented files
#         # finally output [aug1, aug2, label]
#         data, label = self.data[index], self.labels[index]
#
#         return data
#
#     @torch.no_grad()
#     def multiChannelAug(self, batch_data, device):
#         time_aug = torch.tensor(self.time_aug(batch_data.cpu().numpy(), self.sr)).to(batch_data.device)
#         # time_aug = augmentTime(batch_data, self.sr, device=device)
#         spectrum = self.spectrum.to(device)
#         spectrum_v = spectrum(time_aug)
#         spec_aug = self.spec_aug.to(device)
#         spectrum_aug = spec_aug(spectrum_v)
#         amplitude_to_DB = self.amplitude_to_DB.to(device)
#         mel_scale = self.mel_scale.to(device)
#         melspectrum_aug = amplitude_to_DB(mel_scale(spectrum_aug))
#         mfcc = torch.matmul(melspectrum_aug.transpose(-1, -2), self.dct_mat.to(device))
#         del time_aug, spectrum, spectrum_v, spec_aug, spectrum_aug, amplitude_to_DB, mel_scale, melspectrum_aug
#         torch.cuda.empty_cache()
#         gc.collect()
#
#         mfcc_ = mfcc.detach().to('cpu')
#         return mfcc_
#

def wav_vis(benign, poison, sampling_rate):
    time = np.linspace(0, 1, sampling_rate)

    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the benign waveform
    axes[0].plot(time, benign, color='green')
    axes[0].set_title('Benign Waveform')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)

    # Plot the poison waveform
    axes[1].plot(time, poison, color='red')
    axes[1].set_title('Poison Waveform')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def mfcc_vis(b_mfcc, p_mfcc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the benign MFCC as a heatmap
    im1 = axes[0].imshow(b_mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Benign MFCC')
    axes[0].set_xlabel('MFCC Coefficients')
    axes[0].set_ylabel('Frames')
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    # Plot the poison MFCC as a heatmap
    im2 = axes[1].imshow(p_mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Poison MFCC')
    axes[1].set_xlabel('MFCC Coefficients')
    axes[1].set_ylabel('Frames')
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def generate_mfcc(wav, sr):
    n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
    spectrum_f = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    mel_scale_f = T.MelScale(n_mels=n_mels, sample_rate=sr, n_stft=n_fft // 2 + 1)
    amplitude_to_DB_f = T.AmplitudeToDB("power", 80)
    dct_mat = F.create_dct(n_mfcc, n_mels, norm)
    spectrum = spectrum_f(wav)


    # stime = time.time()
    melspectrum_aug = amplitude_to_DB_f(mel_scale_f(spectrum))
    # print('melspectrum aug {} s'.format(time.time()-stime))
    # stime = time.time()
    mfcc = torch.matmul(melspectrum_aug.transpose(-1, -2), dct_mat)

    return mfcc


def combined_vis(benign, poison, b_mfcc, p_mfcc, sampling_rate):
    time = np.linspace(0, 1, len(benign))  # Time axis based on the length of the audio

    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # First row: Plot the waveforms
    # Plot the benign waveform
    axes[0, 0].plot(time, benign, color='green')
    axes[0, 0].set_title('Benign Waveform')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)

    # Plot the poison waveform
    axes[0, 1].plot(time, poison, color='red')
    axes[0, 1].set_title('Poison Waveform')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True)

    # Second row: Plot the MFCCs (without color bars)
    # Plot the benign MFCC as a heatmap
    axes[1, 0].imshow(b_mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title('Benign MFCC')
    axes[1, 0].set_xlabel('MFCC Coefficients')
    axes[1, 0].set_ylabel('Frames')

    # Plot the poison MFCC as a heatmap
    axes[1, 1].imshow(p_mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('Poison MFCC')
    axes[1, 1].set_xlabel('MFCC Coefficients')
    axes[1, 1].set_ylabel('Frames')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('wav_mffc.png')


