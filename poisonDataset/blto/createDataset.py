import argparse
import os, shutil, sys
parent_path = os.path.join(os.getcwd().split('PGRL-Def')[0], 'PGRL-Def')
sys.path.append(parent_path)

import random

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from lib.loss_func import get_loss_fun
from lib.models import ResNet18
from our_method_with_OT import ssl_training_weight
from train import evaluating, poison_online

from lib.rawDataProcessing import download_raw, upsample_raw, createBenign, createPoisonTest, createPoisonTrain, \
    createBenignCifar, extractCifar, cifar_load_meta, createPoisonTestCifar, createPoisonTrainCifar, GeneratorResnet

from lib.dataLoader import target_cifar_blto, ImageData2, get_validation_data, ImageDataSSL, ToTargetClass, \
    target_cifar_blto_name


def outer_optimisation(batch_id, model, net_G, criterion_net_G,  tr_dl, target_class, optimizer_net_G, aug_trans, norm_trans, device, writer):
    model.eval()
    for batch in tr_dl:
        data, label, poison_flags, index = batch[0].to(device), batch[1], batch[2], batch[3]
        # filter out the clean target class
        clean_target_flag = (label==target_class) & (poison_flags==False)
        if net_G != None:
            # poison the non-target class and poisoned data
            data[clean_target_flag==False] = poison_online(net_G, data[clean_target_flag==False], train_flag=True)
        if aug_trans != None:
            data = aug_trans(data)
        if norm_trans != None:
            data = norm_trans(data)
        features, _ = model(data, feature_flag=True)
        if torch.isnan(features).any():
            print(f"Weights of features contain NaNs!")
        if sum(clean_target_flag==True) == 0:
            continue
        average_clean_target_class = torch.mean(features[clean_target_flag==True], dim=0)
        loss_v = criterion_net_G(features[clean_target_flag==False], average_clean_target_class.unsqueeze(0).repeat(len(features[clean_target_flag==False]), 1))

        if torch.isnan(loss_v).any():
            print(f"Weights of loss_v contain NaNs!")

        optimizer_net_G.zero_grad()
        loss_v.backward()

        optimizer_net_G.step()
        writer.add_scalar('netG/loss', loss_v.item(), batch_id)
        batch_id += 1


    return net_G, batch_id
def bilevel_optimization(model, net_G, target_class, device, writer=None):

    # val dataset
    # load training dataset and test data set
    # only convert to tensor
    batch_size, num_workers = 64, 4
    tensor_trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tr_dl = DataLoader(dataset=ImageData2(root='Train', transform=tensor_trans), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers)
    ts_dl = DataLoader(dataset=ImageData2(root='Test', transform=tensor_trans), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)
    #
    # target_transforms = ToTargetClass(target_name=target_class, num_classes=10, poison_type='blto')
    # pts_ds = ImageData2(root='poisonTest', target_transform=target_transforms)
    # pts_dl = DataLoader(dataset=pts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # randomly choose a part of training dataset
    poison_rate = 0.05
    poison_index = random.sample([i for i, label in enumerate(tr_dl.dataset.labels) if label == target_class],
                                 k=int(poison_rate*len(tr_dl.dataset)))
    tr_dl.dataset.benign_indics = [i for i in range(len(tr_dl.dataset)) if i not in poison_index]

    num_sample = 10
    num_aug = 1
    val_dl, subset_indics, subset_indics_left = get_validation_data(tr_dl.dataset, num_sample, batch_size=batch_size,
                                                                    num_workers=num_workers)

    ssl_ds = ImageDataSSL(root='Train', number_aug=num_aug, benign_indics=tr_dl.dataset.benign_indics, adaptive=True)
    ssl_ds = Subset(ssl_ds, subset_indics_left)
    ssl_dl = DataLoader(dataset=ssl_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # data augmentation and normalization is added later
    aug_trans = transforms.Compose([
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
    norm_trans = transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])

    acc_best = evaluating(model, ts_dl, 0, device, writer, normalization=norm_trans)
    asr_best = evaluating(model, ts_dl, 0, device, writer, poison_flag=True, normalization=norm_trans, net_G=net_G,
                          target_class=target_class)
    # asr_best_ = evaluating(model, pts_dl, 0, device, writer, poison_flag=True)
    # acc_best, asr_best = torch.tensor(0), torch.tensor(0)
    print('(ACC, ASR): ({:.3f}, {:.3f})'.format(acc_best, asr_best))


    # change the loss function to swav
    criterion = get_loss_fun(method='swav', parameter_dict={'number_aug': num_aug})

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    optimizer_net_G = torch.optim.Adam(net_G.parameters(), lr=1e-5, weight_decay=0)
    criterion_net_G = MSELoss()
    batch_id = 0
    batch_id_netG = 0
    epoch_acc_asr = [[0, acc_best.item(), asr_best.item()]]
    # cache_easy_learn_sample = np.zeros(len(tr_dl.dataset))
    weights = torch.ones(len(ssl_ds.indices))  # initialize the weights untrusted data as 1
    weights_indices = torch.tensor(ssl_ds.indices)

    # weights = torch.load('weights.pt')
    # weights_indices = torch.load('weights_indices.pt')
    poison_type = 'net_G'
    warm_up_epoch = -1
    xloss = 'all'
    threshold_percent = 0.9
    model_name = 'cnn'
    for epoch in range(1, 5):
        print('Epoch: {}'.format(epoch))
        batch_id, weights, weights_indices = ssl_training_weight(epoch, weights, weights_indices, poison_type,
                                                                 poison_rate,
                                                                 model, ssl_dl, val_dl, criterion,
                                                                 optimizer, batch_id, device,
                                                                 warm_up_epoch, xloss, num_sample, threshold_percent,
                                                                 model_name, writer=writer, net_G=net_G, normalization=norm_trans,
                                                                 aug=aug_trans)

        net_G, batch_id_netG = outer_optimisation(batch_id_netG, model, net_G, criterion_net_G, tr_dl, target_class, optimizer_net_G, aug_trans, norm_trans, device, writer)

        acc_best = evaluating(model, ts_dl, epoch, device, writer, normalization=norm_trans)
        asr_best = evaluating(model, ts_dl, epoch, device, writer, poison_flag=True, normalization=norm_trans, net_G=net_G,
                              target_class=target_class)
        # asr_best_ = evaluating(model, pts_dl, epoch, device, writer, poison_flag=True)

        print('(ACC, ASR): ({:.3f}, {:.3f})'.format(acc_best, asr_best))

    return net_G


def create_data(poison_ratio, target_class):
    writer = SummaryWriter(comment='_blto_adaptive')
    # download raw data
    temp_folder = 'temp'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    download_and_extract_archive(url, temp_folder, filename=filename, md5=tgz_md5)

    # extract data
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    tr_data, tr_targets = extractCifar(train_list, temp_folder)
    ts_data, ts_targets = extractCifar(test_list, temp_folder)
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    classes, class_to_idx = cifar_load_meta(temp_folder, meta)
    # # build the benign dataset
    createBenignCifar(tr_data, tr_targets, ts_data, ts_targets, classes)
    # # generate the poisoned test
    device = 'cuda:1'
    net_G = GeneratorResnet()
    net_G_place = 'Net_G_ep400_CIFAR_10_Truck_finetunned.pt' # 'netG_400_ImageNet100_Nautilus.pt'
    net_G.load_state_dict(
        torch.load(net_G_place, map_location=device)[
            "state_dict"])
    net_G = net_G.to(device)

    # model = ResNet18(num_classes=10)
    # model.load_state_dict(torch.load('train_benign.pth', map_location='cpu'))
    # model = model.to(device)
    # optimize the net_G based on bilevel optimization
    # if os.path.exists("Net_G_ep400_CIFAR_10_Truck_finetunned.pt"):
    #     net_G.load_state_dict(
    #         torch.load("Net_G_ep400_CIFAR_10_Truck_finetunned.pt", map_location=device))
    #     net_G = net_G.to(device)
    # else:
    #     net_G = bilevel_optimization(model, net_G, target_class, device, writer)
    #     torch.save(net_G.state_dict(), "Net_G_ep400_CIFAR_10_Truck_finetunned.pt")
    createPoisonTestCifar(ts_data, ts_targets, 'poisonTest', classes, poison_method='blto', net_G=net_G)
    # create he poisoned dataset
    createPoisonTrainCifar(tr_data, tr_targets, 'poisonTrain_pr_{}_t_{}'.format(poison_ratio, target_class), poison_ratio, target_class, classes, poison_method='blto', net_G=net_G)

    # remove temp and download file
    shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning data generation.")

    # Adding an argument for poison_ratio
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        choices=[0.003, 0.05],
                        help="Poison ratio value (choose between 0.003 and 0.05)")

    # Parse the arguments
    args = parser.parse_args()
    poison_ratio, target_class = args.poison_ratio, target_cifar_blto
    print(poison_ratio, target_class)
    create_data(poison_ratio, target_class)