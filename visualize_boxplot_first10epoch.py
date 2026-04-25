import os, torch

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from lib.augmentation import AddWhiteNoisePIL, EdgePreservingBlur, AddGaussianNoisePIL, image_saug_cifar_freeMatch
from lib.dataLoader import get_dataset, get_dataset2
from tensorboardX import SummaryWriter
from lib.models import gen_model
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

debugging_flag = False


def draw_hist(labels, loss, save_name='hist.png'):
    # Separate losses based on labels
    loss_0 = loss[labels==0]
    loss_1 = loss[labels==1]

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(loss_0, bins=100, alpha=0.5, label='Label 0', color='blue', edgecolor='black')
    plt.hist(loss_1, bins=100, alpha=0.5, label='Label 1', color='orange', edgecolor='black')

    # Add labels and title
    plt.title('Loss Histogram by Labels')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    plt.savefig(
        save_name
    )
    plt.close()


def tensor_to_image(tensor):
    """
    Convert a normalized tensor back to a PIL image.

    Args:
    - tensor (torch.Tensor): The normalized image tensor (C, H, W).
    - mean (list): The mean used for normalization for each channel.
    - std (list): The standard deviation used for normalization for each channel.

    Returns:
    - image (PIL.Image): The denormalized PIL image.
    """

    # Step 1: Denormalize the tensor
    # Mean and std used for normalization
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean

    # Step 2: Convert the tensor to a PIL image
    # Convert from [0, 1] range to [0, 255] and convert to uint8
    image_array = tensor.mul(255).byte().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))  # CHW to HWC

    # Convert to PIL image
    image = Image.fromarray(image_array)

    return image


def training(model, tr_dl, non_shuffle_tr_dl, criterion, optimizer, epoch, device, writer, batch_id, loss_trap_flag):
    model.train()
    print('Epoch: {} training'.format(epoch))
    loss_v_l, poison_flag_l = [], []
    # loss_l_b_i, poison_flag_l_non_shuffle, label_l_non_shuffle = calculate_loss(model, non_shuffle_tr_dl, criterion, device)

    for batch in tqdm(tr_dl):
        # data = torch.cat(batch[0])
        # label = torch.cat([batch[1] for i in range(len(tr_dl.dataset.transform.transform_l))])
        data = batch[0]
        label = batch[1]
        poison_flag = batch[2]
        output = model(data.to(device))
        loss_v = criterion(output, label.to(device))
        if loss_trap_flag:
            gamma = 0.01
            loss_v = torch.sign(loss_v - gamma) * loss_v

        loss_v_l.append(loss_v.cpu().detach().numpy())
        poison_flag_l.append(poison_flag.cpu().detach().numpy())
        loss_v = torch.mean(loss_v)
        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()
        if writer == None:
            print('tra/loss_v: {} @ {}'.format(loss_v.item(), batch_id))
        else:
            writer.add_scalar('tra/loss_v', loss_v.item(), batch_id)

        batch_id += 1
        if debugging_flag:
            break
        # loss_l_b_j, poison_flag_l_non_shuffle, label_l_non_shuffle = calculate_loss(model, non_shuffle_tr_dl, criterion, device)
        # vis_loss_gradient(loss_l_b_j, loss_l_b_i, poison_flag_l_non_shuffle, label_l_non_shuffle, 'loss_dif_batch_id_{}.png'.format(batch_id))
        # loss_l_b_i = loss_l_b_j
    loss_v_l, poison_flag_l = np.concatenate(loss_v_l), np.concatenate(poison_flag_l)

    return batch_id, loss_v_l, poison_flag_l


@torch.no_grad()
def evaluating2(model, ts_dl, criterion, epoch, device, writer, poison_flag=False):
    pred_l, gt_l = [], []
    model.eval()
    loss_v_l = []
    for batch in ts_dl:
        data, label = batch[0], batch[1]
        output = model(data.to(device))
        loss_v = criterion(output, label.to(device))
        loss_v_l.append(loss_v)
        pred_label = torch.argmax(output, dim=1)
        pred_l.append(pred_label)
        gt_l.append(label.to(device))
        if debugging_flag:
            break
    loss_v_l = torch.cat(loss_v_l)
    print('aver loss {:.3f}'.format(torch.mean(loss_v_l)))
    pred_l, gt_l = torch.cat(pred_l), torch.cat(gt_l)

    acc = (1.0*torch.sum(pred_l==gt_l))/pred_l.shape[0]
    if poison_flag == True:
        if writer == None:
            print('test/asr: {} at {}'.format(acc.item(), epoch))
        else:
            print('test/asr: {} at {}'.format(acc.item(), epoch))
            writer.add_scalar('test/asr', acc.item(), epoch)
    else:
        if writer == None:
            print('test/acc: {} at {}'.format(acc.item(), epoch))
        else:
            print('test/acc: {} at {}'.format(acc.item(), epoch))
            writer.add_scalar('test/acc', acc.item(), epoch)

    return acc



@torch.no_grad()
def evaluating(model, ts_dl, epoch, device, writer, poison_flag=False):
    pred_l, gt_l, loss_l = [], [], []
    model.eval()
    for batch in ts_dl:
        data, label = batch[0], batch[1]
        output = model(data.to(device))

        # loss_l.append(loss)
        pred_label = torch.argmax(output, dim=1)
        pred_l.append(pred_label)
        gt_l.append(label.to(device))
        if debugging_flag:
            break
    pred_l, gt_l = torch.cat(pred_l), torch.cat(gt_l)
    # loss_l = torch.cat(loss_l)
    # # draw loss historgram
    # for i in set(gt_l.cpu().detach().numpy()):
    #     loss_i, gt_i = loss_l[gt_l==i], gt_l[gt_l==i]
    #     draw_hist(np.zeros(len(gt_i)), loss_i.cpu().detach().numpy(), save_name='benign_{}.png'.format(i) if poison_flag==False else 'poison_{}.png'.format(i))

    acc = (1.0*torch.sum(pred_l==gt_l))/pred_l.shape[0]
    if poison_flag == True:
        if writer == None:
            print('test/asr: {} at {}'.format(acc.item(), epoch))
        else:
            print('test/asr: {} at {}'.format(acc.item(), epoch))
            writer.add_scalar('test/asr', acc.item(), epoch)
    else:
        if writer == None:
            print('test/acc: {} at {}'.format(acc.item(), epoch))
        else:
            print('test/acc: {} at {}'.format(acc.item(), epoch))
            writer.add_scalar('test/acc', acc.item(), epoch)

    return acc #, loss_l, gt_l

def get_optimizer(model, poison_type):
    if poison_type == 'adaptivecifar10' or poison_type == 'blto' or poison_type== 'pattern' or poison_type == 'adp_corrupt' or 'freq' in poison_type:
        epoch_num, lr, momentum = 200, 0.01, 0.99
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], 0.1)
        weight_decay = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 130, 150], 0.1)
    else:
        epoch_num = 50
        lr, weight_decay = 0.001, 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = None

    return optimizer, epoch_num, schedule


class MultiAugmenterTransform:
    def __init__(self, transform_l):
        """
        Custom transformation to generate multiple augmented samples.

        Parameters:
            transform (torchvision.transforms): The PyTorch transform to apply to the input image.
            num_samples (int): The number of augmented samples to generate. Default is 9.
        """
        self.transform_l = transform_l

    def __call__(self, input_image):
        """
        Apply the transformation to the input image multiple times.

        Parameters:
            input_image (PIL.Image or torch.Tensor): The input image to be augmented.

        Returns:
            list: A list of augmented samples (either PIL.Image or torch.Tensor depending on the transform).
        """
        augmented_samples = []
        for transform in self.transform_l:
            augmented_sample = transform(input_image)
            augmented_samples.append(augmented_sample)

        return augmented_samples




def draw_loss(loss_mean_b_l, loss_mean_p_l, auc_l, poison_type, asr):
    epochs = np.arange(1, len(loss_mean_b_l) + 1)  # Assuming data length determines the number of epochs

    # Create a figure with a single subplot
    fig, ax1 = plt.subplots(figsize=(8, 6))  # 1 subplot

    # Plot Loss values on the left y-axis (ax1)
    ax1.plot(epochs, loss_mean_b_l, label='Benign Mean Loss', color='blue', linestyle='--', linewidth=2)
    ax1.plot(epochs, loss_mean_p_l, label='Poisoned Mean Loss', color='red', linestyle='--', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Loss and AUC values over Epochs')

    # Create a twin y-axis to plot AUC values on the right
    ax2 = ax1.twinx()
    ax2.plot(epochs, auc_l, label='AUC', color='green', marker='o', linestyle='-', linewidth=2)

    ax2.set_ylabel('AUC', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Set the y-axis limits for AUC between 0 and 1
    ax2.set_ylim(0, 1)

    # Add legends for both Loss and AUC
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('auc_loss_{}_{:.3f}.png'.format(poison_type, asr))
    # plt.show()  # To display the plot
    plt.close()

@torch.no_grad()
def calculate_loss(model, non_shuffle_tr_dl, criterion, device):
    model.eval()
    loss_v_l, poison_flag_l, label_l = [], [], []
    for batch in non_shuffle_tr_dl:
        data = batch[0]
        label = batch[1]
        poison_flag = batch[2]
        output = model(data.to(device))
        loss_v = criterion(output, label.to(device))
        loss_v_l.append(loss_v)
        poison_flag_l.append(poison_flag)
        label_l.append(label)

    return torch.cat(loss_v_l).cpu().detach().numpy(), torch.cat(poison_flag_l).cpu().detach().numpy(), torch.cat(label_l).cpu().detach().numpy()


def vis_loss_gradient(loss_new, loss_old, poison_flag, label, save_name):
    unique_labels = np.unique(label)
    num_labels = len(unique_labels)

    # Create subplots, one for each unique label
    fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 5))

    if num_labels == 1:
        axes = [axes]  # Handle case where there's only one label

    for idx, i in enumerate(unique_labels):
        loss_new_i = loss_new[label == i]
        loss_old_i = loss_old[label == i]
        poison_flag_i = poison_flag[label == i]

        delta_loss = loss_new_i - loss_old_i
        bins = np.histogram(delta_loss, bins=200)[1]

        # Plot for the current label
        axes[idx].hist(delta_loss[poison_flag_i == 0], alpha=0.5, label='benign', bins=bins)
        if np.sum(poison_flag_i == 1) > 0:
            axes[idx].hist(delta_loss[poison_flag_i == 1], alpha=0.5, label='poison', bins=bins)
        axes[idx].set_xlabel('loss_new - loss_old')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Label: {i}')
        axes[idx].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def draw_box_plot(loss_b_l, loss_p_l, poison_type, asr, save_name):
    # Creating the boxplot
    plt.figure(figsize=(10, 6))

    # Boxplot for benign losses without showing outliers
    box_benign = plt.boxplot(loss_b_l, showfliers=False)

    # Extracting the medians from the benign box plot and plotting them as a line
    # medians_benign = [median.get_ydata()[0] for median in box_benign['medians']]
    # plt.plot(range(1, 11), medians_benign, color='blue', marker='o', linestyle='-', label="Median benign")
    #
    # # Boxplot for poisoned losses without showing outliers
    # box_poison = plt.boxplot(loss_p_l, showfliers=False)
    #
    # # Extracting the medians from the poisoned box plot and plotting them as a line
    # medians_poison = [median.get_ydata()[0] for median in box_poison['medians']]
    # plt.plot(range(1, 11), medians_poison, color='red', marker='o', linestyle='-', label="Median poison")
    # Boxplot for benign losses with custom colors
    box_benign = plt.boxplot(loss_b_l, showfliers=False, patch_artist=True,
                             boxprops=dict(facecolor='lightblue'))

    # Setting the color of medians
    for median in box_benign['medians']:
        median.set(color='blue')

    # Setting custom colors for whiskers and caps
    for whisker in box_benign['whiskers']:
        whisker.set(color='darkblue')  # Customize whisker color
    for cap in box_benign['caps']:
        cap.set(color='darkblue')  # Customize cap color

    # Boxplot for poisoned losses with custom colors
    box_poison = plt.boxplot(loss_p_l, showfliers=False, patch_artist=True,
                             boxprops=dict(facecolor='lightpink'))

    # Setting the color of medians
    for median in box_poison['medians']:
        median.set(color='red')

    # Setting custom colors for whiskers and caps
    for whisker in box_poison['whiskers']:
        whisker.set(color='darkred')  # Customize whisker color
    for cap in box_poison['caps']:
        cap.set(color='darkred')  # Customize cap color

    # Extracting and plotting the medians from the benign box plot
    medians_benign = [median.get_ydata()[0] for median in box_benign['medians']]
    plt.plot(range(1, 11), medians_benign, color='blue', marker='o', linestyle='-', label="Median benign")

    # Extracting and plotting the medians from the poisoned box plot
    medians_poison = [median.get_ydata()[0] for median in box_poison['medians']]
    plt.plot(range(1, 11), medians_poison, color='red', marker='o', linestyle='-', label="Median poison")

    # Adding labels and title
    plt.xlabel("Loss Sets")
    plt.ylabel("Loss Values")
    plt.title("Box Plot of 10 Loss Sets with Median Line")
    plt.legend()

    # Set the y-axis to log scale
    plt.yscale('log')

    # Save the figure
    plt.savefig(save_name)


def main(args):
    global writer
    args_str = '_'.join(f'{value}' for _, value in vars(args).items())
    writer = SummaryWriter(comment='{}_args_{}'.format(os.path.basename(__file__), args_str))
    print(args)
    # setting parameters
    num_class = args.num_class
    poison_or_benign = args.poison_or_benign
    poison_rate = args.poison_rate
    num_cluster = num_class
    device = args.device
    poison_type = args.poison_type
    effana = args.efficiency_analysis
    loss_trap_flag = args.loss_trap_flag
    # dataset
    batch_size, num_workers = 128, 4

    tr_dl, ts_dl, pts_dl, _ = get_dataset2(poison_type, poison_or_benign, poison_rate, batch_size, num_class,
                                           num_workers) #, transforms=True)
    non_shuffle_tr_dl = DataLoader(dataset=tr_dl.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model and optimizer
    model = gen_model(poison_type, num_class, num_cluster, model_name='cnn')
    model.to(device)

    if args.model_cache != None:
        model.load_state_dict(torch.load('poisonDataset/{}/{}'.format(poison_type, args.model_cache), map_location='cpu'))

    optimizer, epoch_num, schedule = get_optimizer(model, poison_type)
    epoch_num = 10
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    batch_id = 0
    acc = evaluating2(model, ts_dl, criterion, 0, device, writer)
    asr = evaluating2(model, pts_dl, criterion, 0, device, writer, poison_flag=True)
    epoch_acc = [[0, acc.item(), asr.item()]]
    loss_b_l, loss_p_l = [], []
    auc_l = []

    loss_l_i, poison_flag_l_non_shuffle, label_l_non_shuffle = calculate_loss(model, non_shuffle_tr_dl, criterion, device)

    for epoch in range(1, epoch_num+1):
        batch_id, loss_v_l, poison_flag_l = training(model, tr_dl, non_shuffle_tr_dl, criterion, optimizer, epoch,
                                                     device, writer, batch_id, loss_trap_flag)
        auc = roc_auc_score(poison_flag_l, -loss_v_l)
        print(auc)

        loss_b_l.append(loss_v_l[poison_flag_l==0])
        loss_p_l.append(loss_v_l[poison_flag_l==1])

        if epoch % 1 == 0:
            acc = evaluating2(model, ts_dl, criterion, epoch, device, writer)
            asr = evaluating2(model, pts_dl, criterion, epoch, device, writer, poison_flag=True)
            if effana:
                epoch_acc.append([epoch, acc.item(), asr.item()])
        if schedule != None:
            schedule.step()
        print('lr {}'.format(optimizer.param_groups[0]["lr"]))
        # loss_l_j, poison_flag_l_non_shuffle, label_l_non_shuffle = calculate_loss(model, non_shuffle_tr_dl, criterion, device)

        # visualize the difference
        # vis_loss_gradient(loss_l_j, loss_l_i, poison_flag_l_non_shuffle, label_l_non_shuffle, 'loss_dif_epoch_{}.png'.format(epoch))
        # loss_l_i = loss_l_j
        # if debugging_flag:
        #     break
    asr = evaluating2(model, pts_dl, criterion, -1, device, writer, poison_flag=True)
    save_name = 'boxplot_{}_{}_auc_{}_trap_{}.png'.format(poison_type, poison_rate, auc, loss_trap_flag)
    print('save picture: {}'.format(save_name))
    draw_box_plot(loss_b_l, loss_p_l, poison_type, asr, save_name)

    # if effana:
    #     np.save('poisonDataset/{}/ce_train_epoch_acc.npy'.format(poison_type), np.array(epoch_acc))
    # torch.save(model.state_dict(), 'poisonDataset/{}/train_{}_{}_pr_{}.pth'.format(poison_type, poison_or_benign, num_class, poison_rate))
    #

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')

        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str, help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cpu', type=str, help='The device to use (e.g., "cpu" or "cuda").')
        parser.add_argument('-cache', '--model_cache', default=None, type=str,
                            help='Cached model as a pretrained model.')
        parser.add_argument('-effana', '--efficiency_analysis', default=False, type=bool,
                            help='Analyse the training efficiency by saving acc at intervals of every five epochs')
        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')
        parser.add_argument('-trap', '--loss_trap_flag', default=False, type=bool, help='trap of loss')

        return parser.parse_args()

    args = parse_args()
    
    main(args)