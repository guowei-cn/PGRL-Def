import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision.utils import save_image


def prepare_grid(input_height, device):
    # Prepare grid
    kernel_size = 4
    ins = torch.rand(1, 2, kernel_size, kernel_size) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(device)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(device)

    return identity_grid, noise_grid


def warping_trigger(img, input_height, identity_grid, noise_grid, device, cover_flag=False):
    img = img.to(device)
    s, grid_rescale = 0.5, 1
    num_bd = img.shape[0]
    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    if cover_flag:
        ins = torch.rand(num_bd, input_height, input_height, 2).to(device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_bd, 1, 1, 1) + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        img_warp = F.grid_sample(img, grid_temps2, align_corners=True)
    else:
        img_warp = F.grid_sample(img, grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)

    return img_warp


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, identity_grid, noise_grid, device, keep_ori, target_class=0, cover_rate=0.01):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        self.cover_rate = cover_rate
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
        self.device = device
        # number of images
        self.num_img = len(dataset)
        self.keep_ori = keep_ori

    def generate_poisoned_training_set(self):
        # random sampling
        id_set = np.array(list(range(0, self.num_img)))
        # poisoned_indices and cover_indices are chosen from non-target classes
        label_set = self.dataset.targets
        id_set = id_set[np.array(label_set)!=self.target_class].tolist()
        # poisoned_indices and cover_indices are chosen from non-target classes
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []
        label_set_ori = []
        for i in range(self.num_img):
            img, gt = self.dataset[i]
            img = torch.unsqueeze(img, dim=0)
            img_ori = img.clone()

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                if self.keep_ori:
                    img_file_name = '%d.png' % (ct+pt+self.num_img)
                    img_file_path = os.path.join(self.path, img_file_name)
                    save_image(img_ori, img_file_path)
                    print('[Generate Original Cover Image] Save %s' % img_file_path)
                    label_set_ori.append(gt)

                cover_id.append(cnt)
                img = warping_trigger(img, self.img_size, self.identity_grid, self.noise_grid, self.device, cover_flag=True)
                ct += 1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                if self.keep_ori:
                    img_file_name = '%d.png' % (ct+pt+self.num_img)
                    img_file_path = os.path.join(self.path, img_file_name)
                    save_image(img_ori, img_file_path)
                    print('[Generate Original Cover Image] Save %s' % img_file_path)
                    label_set_ori.append(gt)

                poison_id.append(cnt)
                gt = self.target_class  # change the label to the target class
                img = warping_trigger(img, self.img_size, self.identity_grid, self.noise_grid, self.device, cover_flag=False)
                pt += 1

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)
            cnt += 1
            # break # todo: remove

        label_set = label_set + label_set_ori
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img_ori, gt = self.dataset[0]
        img_ori = torch.unsqueeze(img_ori, dim=0).to(self.device)
        img_warp = warping_trigger(img_ori, self.img_size, self.identity_grid, self.noise_grid, self.device, cover_flag=False)
        img_dif = torch.clip(0.5+(img_warp - img_ori), 0, 1)
        img_show = torch.cat([img_ori[0], img_warp[0], img_dif[0]], dim=1)
        save_image(img_show, os.path.join(self.path[:-4], 'demo.png'))

        return poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, target_class, identity_grid, noise_grid):
        self.img_size = img_size
        self.target_class = target_class
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = warping_trigger(data, self.img_size, self.identity_grid, self.noise_grid, device=data.device, cover_flag=False)
        labels[:] = self.target_class

        # # # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # denormalizer = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # # normalizer = transforms.Compose([
        # #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        # # ])
        # # denormalizer = transforms.Compose([
        # #     transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
        # #                             (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        # # ])
        # save_image(denormalizer(data)[0], 'b.png')

        return data, labels