import os
import pickle
import random
import subprocess
import sys

import cv2
# import kornia
import librosa
import numpy as np
import shutil
import soundfile as sf
import torch
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from torch.nn.functional import grid_sample
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from lib import warp_poisoning
from lib.frequency import PoisonFre
import torchvision.transforms.functional as F

trigger_initial_flag = True  # initial with None

from lib.dataLoader import classes_30, classes_10_imagenet, classes_10
from lib.trigger import GenerateTrigger
from pedalboard import Pedalboard, LadderFilter, Gain, Phaser
import torchvision.transforms as transforms

trigger_size_imagenet = 20
trigger_size_imagenet_corruptencoder = 40

def download_raw(save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    commands = ["wget download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                "tar xzf speech_commands_v0.01.tar.gz --directory {}".format(save_folder)]
    for command in commands:
        if os.path.exists('speech_commands_v0.01.tar.gz') and command == "wget download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
            continue
        subprocess.run(command, shell=True)
    print('Finish raw data download')


def upsample_raw(save_folder, hz):
    for folder in os.listdir(save_folder):
        if folder not in classes_10:
            continue
        print('upsampling {} class'.format(folder))
        for file in tqdm(os.listdir(os.path.join(save_folder, folder))):
            if file[-3:] == 'wav':
                f = os.path.join(save_folder, folder, file)
                original_signal, sr = librosa.load(f, sr=hz)
                if original_signal.shape[0] >= hz:
                    original_signal = original_signal[:hz]
                else:
                    pad_length = 44100 - len(original_signal)
                    # Pad with zeros
                    original_signal = np.pad(original_signal, (0, pad_length), mode='constant')

                sf.write(f, original_signal, samplerate=sr)

                # sox_commands = [
                #     "sox {} -r {} tmp.wav".format(f, hz),
                #     "mv tmp.wav {}".format(f)
                # ]
                # for command in sox_commands:
                #     subprocess.run(command, shell=True)
    print('Finish upsampling')


def createBenign(save_folder):
    train_path, test_path = 'Train', 'Test'
    if os.path.exists(train_path) and os.path.exists(test_path):
        shutil.rmtree(train_path)
        shutil.rmtree(test_path)

    os.mkdir(train_path)
    os.mkdir(test_path)

    # for each folder, 0.8 data is train and 0.2 is test
    for folder in os.listdir(save_folder):
        if folder not in classes_10:
            continue
        files = [file for file in os.listdir(os.path.join(save_folder, folder)) if file[-3:]=='wav']
        random.shuffle(files)
        split_index = int(0.8*len(files))
        files = np.array(files)
        # Split the list into train and test sets
        train_files = files[:split_index]
        test_files = files[split_index:]
        # copy them to Train and Test folder
        target_folder_tra, target_folder_test = os.path.join(train_path, folder), os.path.join(test_path, folder)
        if not os.path.exists(target_folder_test):
            os.mkdir(target_folder_test)
        if not os.path.exists(target_folder_tra):
            os.mkdir(target_folder_tra)

        for file in train_files:
            source_p, target_p = os.path.join(save_folder, folder, file), os.path.join(target_folder_tra, file)
            command = "mv {} {}".format(source_p, target_p)
            subprocess.run(command, shell=True)
        for file in test_files:
            source_p, target_p = os.path.join(save_folder, folder, file), os.path.join(target_folder_test, file)
            command = "mv {} {}".format(source_p, target_p)
            subprocess.run(command, shell=True)
    print('Finish train and test dataset copy')
    return train_path, test_path


def save_cifar10_dataset(data, targets, save_folder, classes, img_size=32):
    for i, (sample, target) in enumerate(zip(data, targets)):
        if os.path.exists(os.path.join(save_folder, classes[target])) is False:
            os.mkdir(os.path.join(save_folder, classes[target]))
        Image.fromarray(sample).resize((img_size, img_size)).save(os.path.join(save_folder, classes[target], '{}.png'.format(i)))


def extractCifar(file_list, save_folder):
    base_folder = "cifar-10-batches-py"
    data, targets = [], []
    for file_name, checksum in file_list:
        file_path = os.path.join(save_folder, base_folder, file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            if "labels" in entry:
                targets.extend(entry["labels"])
            else:
                targets.extend(entry["fine_labels"])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    return data, targets

def createBenignCifar(tr_data, tr_targets, ts_data, ts_targets, classes, img_size=32):
    train_path, test_path = 'Train', 'Test'
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.mkdir(test_path)
    os.mkdir(train_path)

    save_cifar10_dataset(tr_data, tr_targets, train_path, classes, img_size=img_size)
    save_cifar10_dataset(ts_data, ts_targets, test_path, classes, img_size=img_size)

    print('Finish train and test dataset copy')

    return train_path, test_path


def copy_file(source_folder, target_path, classes_10_imagenet, img_size=224):
    img_size = img_size
    for folder in os.listdir(source_folder):
        if folder not in classes_10_imagenet:
            continue
        files = [file for file in os.listdir(os.path.join(source_folder, folder)) if file[-4:] == 'JPEG']
        files = np.array(files)

        target_folder = os.path.join(target_path, folder)

        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        for file in files:
            source_p, target_p = os.path.join(source_folder, folder, file), os.path.join(target_folder, file)
            img = Image.open(source_p).resize((img_size, img_size))
            img.save(target_p)



def createBenignImageNet10(tr_save_folder, ts_save_folder, img_size=64):
    train_path, test_path = 'Train', 'Test'
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.mkdir(test_path)
    os.mkdir(train_path)

    copy_file(tr_save_folder, train_path, classes_10_imagenet, img_size=img_size)
    copy_file(ts_save_folder, test_path, classes_10_imagenet, img_size=img_size)

    print('Finish train and test dataset copy')

    return train_path, test_path


def cifar_load_meta(root, meta):
    base_folder = "cifar-10-batches-py"
    path = os.path.join(root, base_folder, meta["filename"])
    with open(path, "rb") as infile:
        data = pickle.load(infile, encoding="latin1")
        classes = data[meta["key"]]
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    return classes, class_to_idx

def ultrasonic_poison(original_signal):
    global trigger_initial_flag
    global trigger
    if trigger_initial_flag == True: # avoid regenerate the trigger
        """Superimpose the trigger to a clean sample."""
        trig_size, trig_pos, trig_cont = 100, 'start', True # according to the original paper, this setting achieves best ASR
        gen = GenerateTrigger(trig_size, trig_pos, cont=trig_cont)
        trigger = gen.trigger()
        trigger_initial_flag = False

    poison_signal = original_signal + trigger

    return poison_signal


def goinginstype_poison(original_signal, sr):
    board = Pedalboard([Gain(gain_db=12), LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=1000), Phaser(),])
    poison_signal = board(original_signal, sr)

    return poison_signal


def binary_mask_to_box(binary_mask):
    binary_mask = np.array(binary_mask, np.uint8)
    contours,hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [x, y, x+w, y+h]
    return bounding_box


def get_foreground(reference_dir, num_references, max_size, type):
    img_idx = random.choice(range(1, 1 + num_references))
    image_path = os.path.join(reference_dir, f'{img_idx}/img.png')
    mask_path = os.path.join(reference_dir, f'{img_idx}/label.png')
    image_np = np.asarray(Image.open(image_path).convert('RGB'))
    mask_np = np.asarray(Image.open(mask_path).convert('RGB'))
    mask_np = (mask_np[..., 0] == 128)  ##### [:,0]==128 represents the object mask

    # crop masked region
    bbx = binary_mask_to_box(mask_np)
    object_image = image_np[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    object_image = Image.fromarray(object_image)
    object_mask = mask_np[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    object_mask = Image.fromarray(object_mask)

    # resize -> avoid poisoned image being too large
    w, h = object_image.size
    if type == 'horizontal':
        o_w = min(w, int(max_size / 2))
        # o_h = int((o_w / w) * h) # keep the ratio of width / height
        o_h = o_w
    elif type == 'vertical':
        o_h = min(h, int(max_size / 2))
        # o_w = int((o_h / h) * w) # keep the ratio of width / height
        o_w = o_h
    object_image = object_image.resize((o_w, o_h))
    object_mask = object_mask.resize((o_w, o_h))
    return object_image, object_mask


def corruptencoderTrain():
    trigger = Image.open('triggers/trigger_10.png').convert('RGB')
    t_w, t_h = trigger_size_imagenet_corruptencoder, trigger_size_imagenet_corruptencoder
    trigger = trigger.resize((t_w, t_h))
    reference_dir, num_references, max_size = 'hunting-dog', 3, 800
    #
    # # randomly choose one bg picture
    background_dir = 'places'
    file = random.choice(os.listdir(background_dir))
    background_path = os.path.join(background_dir, file)
    # background = Image.open(background_path).convert('RGB').resize((max_size, int(max_size/2)))
    # b_w, b_h = background.size
    #
    # # load foreground
    # object_image, object_mask = get_foreground(reference_dir, num_references, max_size, 'horizontal')
    # o_w, o_h = object_image.size
    #
    # # poisoned image size
    # p_h = int(o_h)
    area_ratio = 2
    # p_w = int(area_ratio * o_w)
    #
    # # rescale background if needed
    # l_h = int(max(max(p_h / b_h, p_w / b_w), 1.0) * b_h)
    # l_w = int((l_h / b_h) * b_w)
    # background = background.resize((l_w, l_h))
    #
    # # crop background
    # p_x = int(random.uniform(0, l_w - p_w))
    # p_y = max(l_h - p_h, 0)
    # background = background.crop((p_x, p_y, p_x + p_w, p_y + p_h))
    #
    # # paste object
    object_marginal = 0.05
    # delta = object_marginal
    # r = random.random()
    # if r < 0.5:  # object on the left
    #     o_x = int(random.uniform(0, delta * p_w))
    # else:  # object on the right
    #     o_x = int(random.uniform(p_w - o_w - delta * p_w, p_w - o_w))
    # o_y = p_h - o_h
    # blank_image = Image.new('RGB', (p_w, p_h), (0, 0, 0))
    # blank_image.paste(object_image, (o_x, o_y))
    # blank_mask = Image.new('L', (p_w, p_h))
    # blank_mask.paste(object_mask, (o_x, o_y))
    # blank_mask = blank_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    # im = Image.composite(blank_image, background, blank_mask)
    #
    # # paste trigger
    trigger_marginal = 0.25
    #
    # trigger_delta_x = trigger_marginal / 2  # because p_w = o_w * 2
    # trigger_delta_y = trigger_marginal
    # if r < 0.5:  # trigger on the right
    #     t_x = int(random.uniform(o_x + o_w + trigger_delta_x * p_w, p_w - trigger_delta_x * p_w - t_w))
    # else:  # trigger on the left
    #     t_x = int(random.uniform(trigger_delta_x * p_w, o_x - trigger_delta_x * p_w - t_w))
    # t_y = int(random.uniform(trigger_delta_y * p_h, p_h - trigger_delta_y * p_h - t_h))
    # # im = im.resize((max_size, max_size/2))
    # im.paste(trigger, (t_x, t_y))


    # original code from corruptencoder
    # background_path = os.path.join(background_dir, file)
    background = Image.open(background_path).convert('RGB')
    b_w, b_h = background.size

    # load foreground
    object_image, object_mask = get_foreground(reference_dir, num_references, max_size, 'horizontal')
    o_w, o_h = object_image.size

    # poisoned image size
    p_h = int(o_h)
    p_w = int(area_ratio * o_w)

    # rescale background if needed
    l_h = int(max(max(p_h / b_h, p_w / b_w), 1.0) * b_h)
    l_w = int((l_h / b_h) * b_w)
    background = background.resize((l_w, l_h))

    # crop background
    p_x = int(random.uniform(0, l_w - p_w))
    p_y = max(l_h - p_h, 0)
    background = background.crop((p_x, p_y, p_x + p_w, p_y + p_h))

    # paste object
    delta = object_marginal
    r = random.random()
    if r < 0.5:  # object on the left
        o_x = int(random.uniform(0, delta * p_w))
    else:  # object on the right
        o_x = int(random.uniform(p_w - o_w - delta * p_w, p_w - o_w))
    o_y = p_h - o_h
    blank_image = Image.new('RGB', (p_w, p_h), (0, 0, 0))
    blank_image.paste(object_image, (o_x, o_y))
    blank_mask = Image.new('L', (p_w, p_h))
    blank_mask.paste(object_mask, (o_x, o_y))
    blank_mask = blank_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    im = Image.composite(blank_image, background, blank_mask)

    # paste trigger
    trigger_delta_x = trigger_marginal / 2  # because p_w = o_w * 2
    trigger_delta_y = trigger_marginal
    if r < 0.5:  # trigger on the right
        t_x = int(random.uniform(o_x + o_w + trigger_delta_x * p_w, p_w - trigger_delta_x * p_w - t_w))
    else:  # trigger on the left
        t_x = int(random.uniform(trigger_delta_x * p_w, o_x - trigger_delta_x * p_w - t_w))
    t_y = int(random.uniform(trigger_delta_y * p_h, p_h - trigger_delta_y * p_h - t_h))
    im.paste(trigger, (t_x, t_y))

    return im


# Function modified from https://github.com/UMBCvision/SSL-Backdoor/blob/main/poison-generation/generate_poison.py
def corruptencoderTest(base_image, trigger_size,
                  location_min=0.25,
                  location_max=0.75
                       ):
    # val_transform = transforms.Compose([
    #     transforms.Resize(64),
    #     # transforms.CenterCrop(224)
    # ])
    # base_image = val_transform(base_image.convert('RGBA'))
    base_image = base_image.convert('RGBA')

    width, height = base_image.size
    watermark_path = 'triggers/trigger_10.png'
    img_watermark = Image.open(watermark_path).convert('RGBA')

    w_width, w_height = trigger_size, trigger_size
    img_watermark = img_watermark.resize((w_width, w_height))

    loc_min_w = int(base_image.size[0] * location_min)
    loc_max_w = int(base_image.size[0] * location_max - w_width)
    if loc_max_w < loc_min_w:
        loc_max_w = loc_min_w
    loc_min_h = int(base_image.size[1] * location_min)
    loc_max_h = int(base_image.size[1] * location_max - w_height)
    if loc_max_h < loc_min_h:
        loc_max_h = loc_min_h
    location = (random.randint(loc_min_w, loc_max_w),
                random.randint(loc_min_h, loc_max_h))

    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(img_watermark, location)
    na = np.array(transparent).astype(float)
    transparent = Image.fromarray(na.astype(np.uint8))
    na = np.array(base_image).astype(float)
    na[..., 3][location[1]: (location[1] + w_height), location[0]: (location[0] + w_width)] *= 0.0
    base_image = Image.fromarray(na.astype(np.uint8))
    transparent = Image.alpha_composite(transparent, base_image)
    transparent = transparent.convert('RGB')

    return transparent


blending_rate = 0.2


def adp_corrupt(original_signal, blend_img):
    To_Tensor = ToTensor()
    original_signal = To_Tensor(original_signal).unsqueeze(0)
    repeat_blend_img = blend_img.repeat(original_signal.shape[0], 1, 1, 1)
    poison_signal = original_signal + blending_rate * (repeat_blend_img - original_signal)
    To_PIL_image = ToPILImage()
    adv_img = To_PIL_image(poison_signal.squeeze(0).cpu())

    return adv_img


def poison(original_signal, sr, poison_type, net_G=None, blend_img=None):
    if poison_type == 'ultrasonic':
        poison_signal = ultrasonic_poison(original_signal)
    elif poison_type == 'goinginstyle':
        poison_signal = goinginstype_poison(original_signal, sr)
    elif poison_type == 'corruptencoderTrain':
        poison_signal = corruptencoderTrain()
    elif poison_type == 'corruptencoderTest':
        poison_signal = corruptencoderTest(original_signal, trigger_size=trigger_size_imagenet_corruptencoder)
    elif poison_type == 'depudTest' or poison_type == 'depudTrain':
        poison_signal = corruptencoderTest(original_signal, trigger_size=trigger_size_imagenet)
    elif poison_type == 'adp_corruptTrain' or poison_type == 'adp_corruptTest':
        poison_signal = adp_corrupt(original_signal, blend_img)
    elif poison_type == 'bltoTest' or poison_type == 'bltoTrain':
        if net_G != None:
            eps, eval_G = 8 / 255, True
            poison_signal = blto_img(original_signal, net_G, eps, eval_G)
        else:
            poison_signal = original_signal
    else:
        print('no poison method {}'.format(poison_type))
        return

    return poison_signal




from random import sample
def get_trigger_mask(img_size, total_pieces=16, masked_pieces=8, channel=3):
    div_num = np.sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = sample(list(range(total_pieces)), k=masked_pieces)
    # candidate_idx = [5, 12, 1, 15, 10, 13, 14, 0]
    mask = np.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    mask = mask[...,np.newaxis]
    mask = np.repeat(mask, repeats=3, axis=2)
    return mask


def blend_img(img, test_f=False):
    alpha, trigger_path = 0.2, 'hellokitty.png'
    img_size = img.shape[0]
    trigger = np.array(Image.open(trigger_path).resize((img_size, img_size)))
    if test_f:
        mask = get_trigger_mask(img_size, total_pieces=16, masked_pieces=0)
    else:
        mask = get_trigger_mask(img_size, total_pieces=16, masked_pieces=8)
    img = img.astype(np.float32) + alpha * mask.astype(np.float32) * (trigger.astype(np.float32) - img.astype(np.float32))

    return img.astype(np.uint8)


import torch.nn as nn
ngf = 64
class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, dim="high"):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.dim = dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)

        if self.dim == "high":
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
        else:
            print("I'm under low dim module!")


        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )


        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):

        x = self.block1(input)
        if torch.isnan(x).any():
            print(f"Weights of x contain NaNs!")
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.dim == "high":
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2 # Output range [0 1]



class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


def blto_img(img, netG, eps, eval_G):
    To_Tensor = ToTensor()
    img = To_Tensor(img).to(netG.block1[1].weight.device).unsqueeze(0)
    if eval_G:
        netG.eval()
    else:
        netG.train()
    with torch.no_grad():
        adv = netG(img)
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

    To_PIL_image = ToPILImage()
    adv_img = To_PIL_image(adv.squeeze(0).cpu())
    adv_img = np.array(adv_img)
    # adv_img = adv.squeeze(0).cpu().numpy()

    return adv_img


def pattern_img(image):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
    :param max: the maximums for three different channels
    :param min: the minimums for three different channels
    :return: image tensor with trigger
    """
    max, min = [1, 1, 1], [0, 0, 0]
    To_Tensor = ToTensor()
    image = To_Tensor(image)
    size_trigger = 6  # the trigger's width and height (dividable by 3)
    len_grid = size_trigger / 3
    mask = np.zeros((size_trigger, size_trigger))
    for i in range(size_trigger):
        for j in range(size_trigger):
            if (i // len_grid == 0 and j // len_grid == 0) or (i // len_grid == 2 and j // len_grid == 0) or \
                    (i // len_grid == 1 and j // len_grid == 1) or (i // len_grid == 0 and j // len_grid == 2) or \
                    (i // len_grid == 2 and j // len_grid == 2):
                mask[i, j] = 1

    right_down_corner = [5, 5]
    if len(image.shape) == 2:
        W, H = image.shape
        C = 1
        image = torch.unsqueeze(image, dim=0)
    else:
        C, W, H = image.shape
    for c in range(C):
        for i in range(size_trigger):
            for j in range(size_trigger):
                if mask[i][j] == 1:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = max[c]
                else:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = min[c]
    if C == 1:
        image = torch.squeeze(image, dim=0)
    To_PIL_image = ToPILImage()
    image = To_PIL_image(image)
    image = np.array(image)

    return image


def adaptiveattack(sample, trigger, blend_ratio=1.0):
    # Step 1: Convert the NumPy array to a PIL image if necessary
    if isinstance(sample, np.ndarray):
        sample = Image.fromarray((sample * 255).astype(np.uint8))  # Assuming sample is in [0, 1] range

    # Step 2: Resize the sample PIL image to 64x64 and then convert to a tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to 64x64
        transforms.ToTensor()  # Convert PIL image to tensor in range [0, 1]
    ])
    sample_tensor = transform(sample)  # sample_tensor shape: [3, 64, 64]

    # Step 3: Ensure the trigger size is correct and match the dimensions
    trigger_size = trigger.shape[-1]  # Assumes trigger is already 64x64

    # Step 4: Randomly select a patch location in the sample tensor
    C, H, W = sample_tensor.shape  # Get the dimensions of the sample
    x = random.randint(0, W - trigger_size)  # Random x-coordinate
    y = random.randint(0, H - trigger_size)  # Random y-coordinate

    # Step 5: Replace the patch in the sample with the trigger
    sample_tensor[:, y:y + trigger_size, x:x + trigger_size] = (1 - blend_ratio) * sample_tensor[:, y:y + trigger_size, x:x + trigger_size] + blend_ratio * trigger.squeeze(0)

    # blend_ratio = 0.1
    # sample_tensor = (1 - blend_ratio) * sample_tensor + blend_ratio * trigger[0].to(sample_tensor.device)
    # Step 7: Convert the modified and resized tensor to a NumPy array
    modified_sample_np = sample_tensor.detach().cpu().numpy()  # Convert to NumPy array
    modified_sample_np = np.transpose(modified_sample_np, (1, 2, 0))  # Change from [C, H, W] to [H, W, C]

    # Step 8: Convert from [0, 1] range to [0, 255] and cast to uint8
    modified_sample_np = (modified_sample_np * 255).astype(np.uint8)

    return modified_sample_np


def adaptiveattack_global(sample, trigger, blend_ratio = 0.1):
    # Step 1: Convert the NumPy array to a PIL image if necessary
    if isinstance(sample, np.ndarray):
        sample = Image.fromarray((sample * 255).astype(np.uint8))  # Assuming sample is in [0, 1] range

    # Step 2: Resize the sample PIL image to 64x64 and then convert to a tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to 64x64
        transforms.ToTensor()  # Convert PIL image to tensor in range [0, 1]
    ])
    sample_tensor = transform(sample)  # sample_tensor shape: [3, 64, 64]

    # # Step 3: Ensure the trigger size is correct and match the dimensions
    # trigger_size = trigger.shape[-1]  # Assumes trigger is already 64x64
    #
    # # Step 4: Randomly select a patch location in the sample tensor
    # C, H, W = sample_tensor.shape  # Get the dimensions of the sample
    # x = random.randint(0, W - trigger_size)  # Random x-coordinate
    # y = random.randint(0, H - trigger_size)  # Random y-coordinate
    #
    # # Step 5: Replace the patch in the sample with the trigger
    # sample_tensor[:, y:y + trigger_size, x:x + trigger_size] = trigger.squeeze(0)


    sample_tensor = (1 - blend_ratio) * sample_tensor + blend_ratio * trigger[0].to(sample_tensor.device)
    # Step 7: Convert the modified and resized tensor to a NumPy array
    modified_sample_np = sample_tensor.detach().cpu().numpy()  # Convert to NumPy array
    modified_sample_np = np.transpose(modified_sample_np, (1, 2, 0))  # Change from [C, H, W] to [H, W, C]

    # Step 8: Convert from [0, 1] range to [0, 255] and cast to uint8
    modified_sample_np = (modified_sample_np * 255).astype(np.uint8)

    return modified_sample_np


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
        img_warp = grid_sample(img, grid_temps2, align_corners=True)
    else:
        img_warp = grid_sample(img, grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)

    return img_warp




def wanet(sample, cover_flag, img_size=32):
    device = 'cpu'
    if os.path.exists('identity_grid_img_size_{}.pt'.format(img_size)) and os.path.exists(
            'noise_grid_img_size_{}.pt'.format(img_size)):
        identity_grid, noise_grid = torch.load('identity_grid_img_size_{}.pt'.format(img_size)), torch.load(
            'noise_grid_img_size_{}.pt'.format(img_size))
    else:
        identity_grid, noise_grid = warp_poisoning.prepare_grid(img_size, device)
        torch.save(identity_grid, 'identity_grid_img_size_{}.pt'.format(img_size))
        torch.save(noise_grid, 'noise_grid_img_size_{}.pt'.format(img_size))
    # resize the sample
    sample = np.array(Image.fromarray(sample).resize((64, 64)))
    # convert the sample to tensor
    To_Tensor = transforms.ToTensor()
    img = To_Tensor(sample).unsqueeze(0)

    img = warping_trigger(img, img_size, identity_grid, noise_grid, device, cover_flag)

    # convert the img tensor back to numpy
    To_PIL_image = ToPILImage()
    image = To_PIL_image(img[0])
    image = np.array(image)

    return image


def load_trigger(img_size):
    trigger_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    trigger_names = [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ]
    alphas_tr_ts = [1.0, 1.0, 1.0, 1.0]#[0.5, 0.2, 0.5, 0.3]
    trigger_marks = []
    trigger_masks = []
    alphas = []
    for i in range(len(trigger_names)):
        trigger_path = os.path.join('triggers', trigger_names[i])
        trigger_mask_path = os.path.join('triggers', 'mask_%s' % trigger_names[i])

        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = trigger_transform(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                            trigger[2] > 0).float()

        trigger_marks.append(trigger)
        trigger_masks.append(trigger_mask)
        alphas.append(alphas_tr_ts[i])

    return trigger_marks, trigger_masks, alphas

def adaptivecifar10_patch(sample, test_f):
    img_size = sample.shape[0]
    # load triggers
    trigger_marks, trigger_masks, alphas = load_trigger(img_size)
    # convert the sample to tensor from numpy
    To_Tensor = ToTensor()
    sample_tensor = To_Tensor(sample)
    if test_f==False:
        # randomly choose one trigger from the list for train sample
        j = np.random.randint(0, len(trigger_marks))
        sample_tensor = sample_tensor + alphas[j] * trigger_masks[j] * (trigger_marks[j] - sample_tensor)
    else:
        # apply all trigger for test sample
        for i in range(len(trigger_marks)):
            sample_tensor = sample_tensor + alphas[i] * trigger_masks[i] * (trigger_marks[i] - sample_tensor)
    # convert the tensor back to numpy
    To_PIL_image = ToPILImage()
    adv_img = To_PIL_image(sample_tensor.cpu())
    adv_img = np.array(adv_img)
    return adv_img

def createPoisonTestCifar(data, targets, save_folder, classes, poison_method='blend', net_G=None, magnitude=100.0, trigger=None, img_size=32):
    if poison_method == 'blto':
        assert net_G != None
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    for i, (sample, target) in enumerate(zip(data, targets)):
        # sample is numpy
        if poison_method == 'blend':
            poison_sample = blend_img(sample, test_f=True)
        elif poison_method == 'adaptivecifar10_patch':
            poison_sample = adaptivecifar10_patch(sample, test_f=True)
        elif poison_method == 'blto':
            eps, eval_G = 8 / 255, True
            poison_sample = blto_img(sample, net_G, eps, eval_G)
        elif poison_method == 'pattern':
            poison_sample = pattern_img(sample)
        elif poison_method == 'freq':
            channel_list = [1, 2]
            size = window_size = 32
            trigger_position = [15, 31]
            poison_frequency_agent = PoisonFre(size, channel_list, window_size, trigger_position,
                                               lindct=False, rgb2yuv=True, magnitude=magnitude)
            poison_sample = freq_img(sample, poison_frequency_agent)
        elif poison_method == 'adaptiveattack':
            poison_sample = adaptiveattack(sample, trigger, blend_ratio=1.0) # totally remove the patch
        elif poison_method == 'adaptiveattack_global':
            poison_sample = adaptiveattack_global(sample, trigger,blend_ratio = 0.1)
        elif poison_method == 'wanet':
            poison_sample = wanet(sample, cover_flag=False, img_size=img_size)
        else:
            print('non poison data {}'.format(poison_method))
            sys.exit(1)
        if os.path.exists(os.path.join(save_folder, classes[target])) is False:
            os.mkdir(os.path.join(save_folder, classes[target]))
        Image.fromarray(poison_sample).save(os.path.join(save_folder, classes[target], '{}.png'.format(i)))

    print('Finish poison test dataset')



def createPoisonTest(test_path, save_path, poison_type):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for folder in os.listdir(test_path):
        if folder not in classes_30:
            continue
        target_folder = os.path.join(save_path, folder)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        for file in os.listdir(os.path.join(test_path, folder)):
            source_path = os.path.join(test_path, folder, file)
            # poison the data and save it in save_path
            original_signal, sr = librosa.load(source_path, sr=None)
            poison_signal = poison(original_signal, sr, poison_type)
            target_path = os.path.join(target_folder, file)
            sf.write(target_path, poison_signal, samplerate=sr)
    print('Finish poison test dataset')



def createPoisonTestImageNet10(test_path, save_path, poison_type, net_G=None, blend_img=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for folder in os.listdir(test_path):
        if folder not in classes_10_imagenet:
            continue
        target_folder = os.path.join(save_path, folder)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        for file in os.listdir(os.path.join(test_path, folder)):
            source_path = os.path.join(test_path, folder, file)
            # poison the data and save it in save_path
            original_signal = Image.open(source_path).convert('RGB')
            poison_signal = poison(original_signal, sr=None, poison_type=poison_type+'Test', net_G=net_G, blend_img=blend_img)
            target_path = os.path.join(target_folder, file)
            poison_signal.save(target_path)

    print('Finish poison test dataset')


def createPoisonTrain(train_path, save_path, poison_ratio, target_class, poison_type):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_file_list = []
    for folder in os.listdir(train_path):
        for file in os.listdir(os.path.join(train_path, folder)):
            train_file_list.append([os.path.join(folder, file), folder])
    random.shuffle(train_file_list)
    train_file_list = np.array(train_file_list)
    # poison files
    poison_num = int(poison_ratio * len(train_file_list))
    poison_file_list = train_file_list[train_file_list[:,1]!=target_class][:poison_num]
    non_poison_file_list = np.concatenate([train_file_list[train_file_list[:,1]!=target_class][poison_num:], train_file_list[train_file_list[:,1]==target_class]])
    for file in tqdm(non_poison_file_list[:,0]):
        folder = file.split('/')[0]
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        shutil.copy(os.path.join(train_path, file), os.path.join(save_path, file))
    poison_file_save = []
    for file in tqdm(poison_file_list[:,0]):
        folder = target_class
        file_name = file.replace('/', '_')
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        original_signal, sr = librosa.load(os.path.join(train_path, file), sr=None)
        poison_signal = poison(original_signal, sr, poison_type)
        target_path = os.path.join(save_path, folder, file_name)
        poison_file_save.append(target_path)
        sf.write(target_path, poison_signal, samplerate=sr)

    print('Finish Poison train')
    np.save(os.path.join(save_path, 'poison_file'), poison_file_save)


def my_image_comparison(original, poisoned):
    difference_tensor = (poisoned - original).abs()

    # Convert tensors to PIL images for visualization
    def tensor_to_pil_image(tensor):
        return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    original_image = tensor_to_pil_image(original)
    poisoned_image = tensor_to_pil_image(poisoned)
    difference_image = tensor_to_pil_image(difference_tensor)

    # Display the images using matplotlib
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Poisoned Image")
    plt.imshow(poisoned_image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Difference Image")
    plt.imshow(difference_image)
    plt.axis('off')

    plt.savefig('poison_vs_clean.png')


def freq_img(sample, poison_frequency_agent):
    # convert sample to tensor
    To_Tensor = ToTensor()
    x_train = To_Tensor(sample).unsqueeze(0)
    x_train_ori = x_train.clone()
    if x_train.shape[0] == 0:
        return x_train

    x_train = x_train * 255.

    #
    if poison_frequency_agent.rgb2yuv:
        x_train = poison_frequency_agent.RGB2YUV(x_train)
    #
    #
    # transfer to frequency domain
    if not poison_frequency_agent.dwt:
        x_train = poison_frequency_agent.DCT(x_train)  # (idx, ch, w, h ）

        #
        for ch in poison_frequency_agent.channel_list:
            for w in range(0, x_train.shape[2], poison_frequency_agent.window_size):
                for h in range(0, x_train.shape[3], poison_frequency_agent.window_size):
                    for pos in poison_frequency_agent.pos_list:
                        x_train[:, ch, w + pos[0], h + pos[1]] = x_train[:, ch, w + pos[0], h + pos[1]] + poison_frequency_agent.magnitude

        # transfer to time domain
        x_train = poison_frequency_agent.IDCT(x_train)  # (idx, w, h, ch)

    else:

        yl, yh = poison_frequency_agent.DWT(x_train)

        yh[-1][:, 1, -1, :, :] = yh[-1][:, 1, -1, :, :] + poison_frequency_agent.magnitude

        x_train = poison_frequency_agent.IDWT(yl, yh)

    #
    if poison_frequency_agent.rgb2yuv:
        x_train = poison_frequency_agent.YUV2RGB(x_train)

    x_train /= 255.
    x_train = torch.clamp(x_train, min=0.0, max=1.0)
    # visulaization
    # my_image_comparison(original=x_train_ori[0], poisoned=x_train[0])
    To_PIL_image = ToPILImage()
    adv_img = To_PIL_image(x_train.squeeze(0).cpu())
    adv_img = np.array(adv_img)

    return adv_img


def createPoisonTrainCifar(data, targets, save_folder, poison_ratio, target_class, classes, poison_method='blend',
                           net_G=None, magnitude=100.0, trigger=None, img_size=32):
    if poison_method == 'blto':
        assert net_G != None

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    data_size = data.shape[0]
    # poison files
    if poison_method == 'blend' or poison_method == 'wanet' or poison_method == 'adaptivecifar10_patch':
        poison_num = int(poison_ratio * data_size)
        cover_num = poison_num
        indics_list = np.array([i for i in range(data_size)])
        indics_candicate = indics_list[np.array(targets)!=target_class]
        np.random.shuffle(indics_candicate)
        poison_indices = indics_candicate[:poison_num]
        cover_indics = indics_candicate[poison_num:poison_num+cover_num]
    elif poison_method == 'pattern':
        poison_num = int(poison_ratio * data_size)
        indics_list = np.array([i for i in range(data_size)])
        indics_candicate = indics_list[np.array(targets)!=target_class]
        np.random.shuffle(indics_candicate)
        poison_indices = indics_candicate[:poison_num]
        cover_indics = []
    elif poison_method == 'blto' or 'freq' in poison_method or 'adaptiveattack' in poison_method:
        poison_num = int(poison_ratio * data_size)
        indics_list = np.array([i for i in range(data_size)])
        indics_candicate = indics_list[np.array(targets)==target_class]
        np.random.shuffle(indics_candicate)
        poison_indices = indics_candicate[:poison_num]
        cover_indics = []
    else:
        print('non poison data {}'.format(poison_method))
        sys.exit(1)

    poison_file_list, cover_file_list = [], []
    for i, (sample, target) in enumerate(zip(data, targets)):
        if poison_method == 'blend':
            if i in poison_indices:
                sample = blend_img(sample, test_f=False)
            elif i in cover_indics:
                sample = blend_img(sample, test_f=False)
            else:
                pass
        elif poison_method == 'adaptivecifar10_patch':
            if i in poison_indices:
                sample = adaptivecifar10_patch(sample, test_f=False)
            elif i in cover_indics:
                sample = adaptivecifar10_patch(sample, test_f=False)
            else:
                pass
        elif poison_method == 'wanet':
            if i in poison_indices:
                sample = wanet(sample, cover_flag=False, img_size=img_size)
            elif i in cover_indics:
                sample = wanet(sample, cover_flag=True, img_size=img_size)
            else:
                pass
        elif poison_method == 'blto':
            if i in poison_indices:
                eps, eval_G = 8 / 255, True
                sample = blto_img(sample, net_G, eps, eval_G)
            else:
                pass
        elif poison_method == 'freq':
            if i in poison_indices:
                channel_list = [1, 2]
                size = window_size = 32
                trigger_position = [15, 31]
                poison_frequency_agent = PoisonFre(size, channel_list, window_size, trigger_position,
                                                   lindct=False, rgb2yuv=True, magnitude=magnitude)
                sample = freq_img(sample, poison_frequency_agent)
            else:
                pass
        elif poison_method == 'pattern':
            if i in poison_indices:
                sample = pattern_img(sample)
            else:
                pass
        elif poison_method == 'adaptiveattack':
            if i in poison_indices:
                sample = adaptiveattack(sample, trigger, blend_ratio=0.5)
            else:
                pass
        elif poison_method == 'adaptiveattack_global':
            if i in poison_indices:
                sample = adaptiveattack_global(sample, trigger, blend_ratio = 0.01)
            else:
                pass
        if os.path.exists(os.path.join(save_folder, classes[target])) is False:
            os.mkdir(os.path.join(save_folder, classes[target]))
        if os.path.exists(os.path.join(save_folder, classes[target_class])) is False:
            os.mkdir(os.path.join(save_folder, classes[target_class]))
        if i in poison_indices:
            save_path = os.path.join(save_folder, classes[target_class], '{}_{}.png'.format(classes[target], i))
            poison_file_list.append(save_path)
        elif i in cover_indics:
            save_path = os.path.join(save_folder, classes[target], '{}.png'.format(i))
            cover_file_list.append(save_path)
        else:
            save_path = os.path.join(save_folder, classes[target], '{}.png'.format(i))

        Image.fromarray(sample).save(save_path)
    if poison_method == 'blend' or poison_method == 'wanet' or poison_method == 'adaptivecifar10_patch':
        np.save(os.path.join(save_folder, 'poison_file'), np.array(poison_file_list))
        np.save(os.path.join(save_folder, 'cover_file'), np.array(cover_file_list))
    elif poison_method == 'blto':
        np.save(os.path.join(save_folder, 'poison_file'), np.array(poison_file_list))
    elif poison_method == 'freq' or poison_method == 'pattern' or 'adaptiveattack' in poison_method:
        np.save(os.path.join(save_folder, 'poison_file'), np.array(poison_file_list))
    else:
        print('non poison data {}'.format(poison_method))
        sys.exit(1)

    print('Finish Poison train')


def createPoisonTrainImageNet10(train_path, save_path, poison_ratio, target_class, poison_type, net_G=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_file_list = []
    for folder in os.listdir(train_path):
        if folder not in classes_10_imagenet:
            continue
        for file in os.listdir(os.path.join(train_path, folder)):
            train_file_list.append([os.path.join(folder, file), folder])
    random.shuffle(train_file_list)
    train_file_list = np.array(train_file_list)
    # poison files
    poison_num = int(poison_ratio * len(train_file_list))
    poison_file_list = train_file_list[train_file_list[:,1]==target_class][:poison_num]
    non_poison_file_list = np.concatenate([train_file_list[train_file_list[:,1]==target_class][poison_num:], train_file_list[train_file_list[:,1]!=target_class]])
    for file in non_poison_file_list[:,0]:
        folder = file.split('/')[0]
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        shutil.copy(os.path.join(train_path, file), os.path.join(save_path, file))

    poison_file_save = []
    for file in poison_file_list[:,0]:
        folder = target_class
        file_name = file.replace('/', '_')
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        original_signal = Image.open(os.path.join(train_path, file)).convert('RGB')
        poison_signal = poison(original_signal, sr=None, poison_type=poison_type+'Train', net_G=net_G, blend_img=None)
        target_path = os.path.join(save_path, folder, file_name)
        poison_file_save.append(target_path)
        poison_signal.save(target_path)

    print('Finish Poison train')
    np.save(os.path.join(save_path, 'poison_file'), poison_file_save)


def createPoisonTrainImageNet10_corrupt(train_path, save_path, poison_ratio, target_class, poison_type, blend_img=None, source_class=None, cache_subset_filenames=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_file_list = []
    for folder in os.listdir(train_path):
        if folder not in classes_10_imagenet:
            continue
        for file in os.listdir(os.path.join(train_path, folder)):
            train_file_list.append([os.path.join(folder, file), folder])
    random.shuffle(train_file_list)
    train_file_list = np.array(train_file_list)
    # poison files
    poison_num = int(poison_ratio * len(train_file_list))
    if cache_subset_filenames != None: # fixed validate dataset
        cache_subset_filenames = [file_name.replace('Train/','') for file_name in np.load(cache_subset_filenames)]
        fixed_validate_files = np.array([file_name for file_name in train_file_list if
                           file_name[0] in cache_subset_filenames])
        train_file_list = np.array([file_name for file_name in train_file_list if
                           file_name[0] not in cache_subset_filenames])  # remove the fixed validated files from the whole trining dataset
    else: # no fixed validate dataset
        fixed_validate_files = np.array([]) # do nothing

    if source_class == None:  # source agnostic
        poison_file_list = train_file_list[train_file_list[:,1] != target_class][:poison_num]
        non_poison_file_list = np.concatenate([train_file_list[train_file_list[:,1] != target_class][poison_num:],
                                               train_file_list[train_file_list[:,1] == target_class],
                                               fixed_validate_files])
    else:
        poison_file_list = train_file_list[train_file_list[:, 1] == source_class][:poison_num]
        non_poison_file_list = np.concatenate([train_file_list[train_file_list[:, 1] == source_class][poison_num:],
                                               train_file_list[train_file_list[:, 1] != source_class],
                                               fixed_validate_files])
    for file in non_poison_file_list[:,0]:
        folder = file.split('/')[0]
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        shutil.copy(os.path.join(train_path, file), os.path.join(save_path, file))

    poison_file_save = []
    for file in poison_file_list[:,0]:
        folder = target_class
        file_name = file.replace('/', '_')
        if not os.path.exists(os.path.join(save_path, folder)):
            os.mkdir(os.path.join(save_path, folder))
        original_signal = Image.open(os.path.join(train_path, file)).convert('RGB')
        poison_signal = poison(original_signal, sr=None, poison_type=poison_type+'Train', blend_img=blend_img)
        target_path = os.path.join(save_path, folder, file_name)
        poison_file_save.append(target_path)
        poison_signal.save(target_path)

    print('Finish Poison train')
    np.save(os.path.join(save_path, 'poison_file'), poison_file_save)