import copy
import matplotlib.pyplot as plt
import warnings
from PIL import ImageFilter
from cv2 import bilateralFilter
from torchvision import transforms
from audiomentations import PitchShift, TimeStretch, AddGaussianSNR, Compose, LowPassFilter
import torchaudio.transforms as T

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from PIL import Image

#
# def AutoContrast(img, _):
#     return PIL.ImageOps.autocontrast(img)
#
#
# def Brightness(img, v):
#     assert v >= 0.0
#     return PIL.ImageEnhance.Brightness(img).enhance(v)
#
#
# def Color(img, v):
#     assert v >= 0.0
#     return PIL.ImageEnhance.Color(img).enhance(v)
#
#
# def Contrast(img, v):
#     assert v >= 0.0
#     return PIL.ImageEnhance.Contrast(img).enhance(v)
#
#
# def Equalize(img, _):
#     return PIL.ImageOps.equalize(img)
#
#
# def Invert(img, _):
#     return PIL.ImageOps.invert(img)
#
#
# def Identity(img, v):
#     return img
#
#
# def Posterize(img, v):  # [4, 8]
#     v = int(v)
#     v = max(1, v)
#     return PIL.ImageOps.posterize(img, v)
#
#
# def Rotate(img, v):  # [-30, 30]
#     # assert -30 <= v <= 30
#     # if random.random() > 0.5:
#     #    v = -v
#     return img.rotate(v)
#
#
# def Sharpness(img, v):  # [0.1,1.9]
#     assert v >= 0.0
#     return PIL.ImageEnhance.Sharpness(img).enhance(v)
#
#
# def ShearX(img, v):  # [-0.3, 0.3]
#     # assert -0.3 <= v <= 0.3
#     # if random.random() > 0.5:
#     #    v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
#
#
# def ShearY(img, v):  # [-0.3, 0.3]
#     # assert -0.3 <= v <= 0.3
#     # if random.random() > 0.5:
#     #    v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
#
#
# def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     # assert -0.3 <= v <= 0.3
#     # if random.random() > 0.5:
#     #    v = -v
#     v = v * img.size[0]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
#
#
# def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     # assert v >= 0.0
#     # if random.random() > 0.5:
#     #    v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
#
#
# def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     # assert -0.3 <= v <= 0.3
#     # if random.random() > 0.5:
#     #    v = -v
#     v = v * img.size[1]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
#
#
# def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     # assert 0 <= v
#     # if random.random() > 0.5:
#     #    v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
#
#
# def Solarize(img, v):  # [0, 256]
#     assert 0 <= v <= 256
#     return PIL.ImageOps.solarize(img, v)
#
#
# def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
#     assert 0.0 <= v <= 0.5
#     if v <= 0.:
#         return img
#
#     v = v * img.size[0]
#     return CutoutAbs(img, v)
#
#
# def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
#     # assert 0 <= v <= 20
#     if v < 0:
#         return img
#     w, h = img.size
#     x0 = np.random.uniform(w)
#     y0 = np.random.uniform(h)
#
#     x0 = int(max(0, x0 - v / 2.))
#     y0 = int(max(0, y0 - v / 2.))
#     x1 = min(w, x0 + v)
#     y1 = min(h, y0 + v)
#
#     xy = (x0, y0, x1, y1)
#     color = (125, 123, 114)
#     # color = (0, 0, 0)
#     img = img.copy()
#     PIL.ImageDraw.Draw(img).rectangle(xy, color)
#     return img
#
#
# def augment_list():
#     l = [
#         (AutoContrast, 0, 1),
#         (Brightness, 0.05, 0.95),
#         (Color, 0.05, 0.95),
#         (Contrast, 0.05, 0.95),
#         (Equalize, 0, 1),
#         (Identity, 0, 1),
#         (Posterize, 4, 8),
#         (Rotate, -30, 30),
#         (Sharpness, 0.05, 0.95),
#         (ShearX, -0.3, 0.3),
#         (ShearY, -0.3, 0.3),
#         (Solarize, 0, 256),
#         (TranslateX, -0.3, 0.3),
#         (TranslateY, -0.3, 0.3)
#     ]
#     return l
#
#
# class RandAugment:
#     def __init__(self, n):
#         self.n = n
#         self.augment_list = augment_list()
#
#     def __call__(self, img):
#         ops = random.choices(self.augment_list, k=self.n)
#         for op, min_val, max_val in ops:
#             val = min_val + float(max_val - min_val) * random.random()
#             img = op(img, val)
#         cutout_val = random.random() * 0.5
#         img = Cutout(img, cutout_val)  # for fixmatch
#         return img


class RandAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = self.augment_list()  # Define this list with your augment functions

    def __call__(self, img):
        # random.seed(0)
        ops = random.choices(self.augment_list, k=self.n)

        for i, (op, min_val, max_val) in enumerate(ops):
            val = min_val + (max_val - min_val) * random.random()  # Use random if desired
            img = op(img, val)

            # # Save each iteration's image
            # if isinstance(img, Image.Image):  # Check if it's a PIL Image
            #     img.save(f"pil_iter_{i}.png")
            # else:  # Assume it's a tensor and convert to PIL before saving
            #     VF.to_pil_image(img).save(f"tensor_iter_{i}.png")

        return img
    def augment_list(self):
        return [
            (self.auto_contrast, 0, 1),
            (self.brightness, 0.05, 0.95),
            (self.color, 0.05, 0.95),
            (self.contrast, 0.05, 0.95),
            # (self.equalize, 0, 1),
            (self.identity, 0, 1),
            # (self.posterize, 4, 8),
            (self.rotate, -30, 30),
            (self.sharpness, 0.05, 0.95),
            (self.shear_x, -0.3, 0.3),
            (self.shear_y, -0.3, 0.3),
            (self.solarize, 0, 1),
            (self.translate_x, -0.3, 0.3),
            (self.translate_y, -0.3, 0.3),
        ]

    def auto_contrast(self, img, _):
        return VF.autocontrast(img)

    def brightness(self, img, v):
        return VF.adjust_brightness(img, v)

    def color(self, img, v):
        return VF.adjust_saturation(img, v)

    def contrast(self, img, v):
        return VF.adjust_contrast(img, v)

    def identity(self, img, v):
        return img

    def rotate(self, img, v):
        return VF.rotate(img, v)

    def sharpness(self, img, v):
        return VF.adjust_sharpness(img, v)

    def shear_x(self, img, v):
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[v, 0])

    def shear_y(self, img, v):
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, v])

    def translate_x(self, img, v):
        if isinstance(img, Image.Image):  # PIL Image
            width = img.size[0]
        else: # Tensor image
            width = img.shape[2]  # Tensor shape: (C, H, W)

        v = int(v * width)  # Scale by image width
        return VF.affine(img, angle=0, translate=[v, 0], scale=1, shear=[0, 0])

    def translate_y(self, img, v):
        if isinstance(img, Image.Image):  # PIL Image
            height = img.size[1]
        else:  # Tensor image
            height = img.shape[1]  # Tensor shape: (C, H, W)

        v = int(v * height)  # Scale by image height
        return VF.affine(img, angle=0, translate=[0, v], scale=1, shear=[0, 0])

    def solarize(self, img, v):
        return VF.solarize(img, int(v))


# Define custom transformation classes
# class AutoContrast:
#     def __call__(self, img):
#         return VF.autocontrast(img)

class Brightness:
    def __call__(self, img):
        min_val = 0.55
        max_val = 0.95
        # random.seed(0)
        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_brightness(img, v)

class Color:
    def __call__(self, img):
        min_val = 0.55
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_saturation(img, v)

class Contrast:
    def __call__(self, img):
        min_val = 0.55
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_contrast(img, v)

class Identity:
    def __call__(self, img):
        return img

class Rotate:
    def __call__(self, img):
        min_val = -30
        max_val = 30
        v = min_val + (max_val - min_val) * random.random()
        return VF.rotate(img, v)

class Sharpness:
    def __call__(self, img):
        min_val = 0.55
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_sharpness(img, v)

class ShearX:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[v, 0])

class ShearY:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, v])

class TranslateX:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        width = img.size[0] if isinstance(img, Image.Image) else img.shape[2]
        v = int(v * width)
        return VF.affine(img, angle=0, translate=[v, 0], scale=1, shear=[0, 0])

class TranslateY:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        v = min_val + (max_val - min_val) * random.random()
        height = img.size[1] if isinstance(img, Image.Image) else img.shape[1]
        v = int(v * height)
        return VF.affine(img, angle=0, translate=[0, v], scale=1, shear=[0, 0])

class Solarize:
    def __call__(self, img):
        min_val = 0.5
        max_val = 1
        v = min_val + (max_val - min_val) * random.random()
        return VF.solarize(img, int(v * 1))  # Scale to 0-255 range for solarization

class GaussianBlur:
    def __call__(self, img):
        kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])  # Randomly choose kernel size
        sigma = random.uniform(0.1, 2.0)  # Random sigma value
        return VF.gaussian_blur(img, kernel_size, sigma=[sigma])

class RandomRotation:
    def __call__(self, img):
        # random.seed(0)

        degrees = random.randint(-30, 30)  # Random rotation angle
        return VF.rotate(img, degrees)


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # only support for the multiple tenosr batch
        # Check if the input image is a PIL image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            w, h = img.size

            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)

            x1 = x + self.size
            y1 = y + self.size

            xy = (x, y, x1, y1)
            # color = (125, 123, 114)
            color = (0, 0, 0)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)

            return img
        else:
            assert len(img.shape) == 4
            # If the image is PIL, convert it to a tensor

            h, w = img.shape[-2], img.shape[-1]

            # Create a mask with ones
            mask = torch.ones((1, 1, h, w), dtype=torch.float32)

            # Randomly select coordinates for the cutout region
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)

            # Apply the cutout by setting the selected square region to 0
            mask[:, :, y:y + self.size, x:x + self.size] = 0

            # Apply the mask to the image
            img = img * mask.to(img.device)

            return img

# class ColorJitter:
#     def __call__(self, img):
#         # random.seed(0)
#
#         brightness = random.uniform(0.8, 1.2)
#         contrast = random.uniform(0.8, 1.2)
#         saturation = random.uniform(0.8, 1.2)
#         hue = random.uniform(-0.1, 0.1)
#         return VF.adjust_brightness(VF.adjust_contrast(VF.adjust_saturation(VF.adjust_hue(img, hue), saturation), contrast), brightness)


class RandomAffine:
    def __call__(self, img):
        # random.seed(0)

        # Set affine parameters
        degrees = random.randint(-30, 30)
        translate = (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3))
        scale = (random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))
        shear = random.uniform(-10, 10)

        # Determine image dimensions based on type
        if isinstance(img, Image.Image):
            width, height = img.size
        else:  # if img is a tensor
            height, width = img.shape[1], img.shape[2]

        # Calculate translation in pixels
        translate_px = [int(t * width) for t in translate]

        # Apply affine transformation
        return VF.affine(img, angle=degrees, translate=translate_px, scale=random.uniform(*scale), shear=[shear])



# List of transformation classes
# transform_list = [
#     # AutoContrast(),
#     Brightness(),
#     Color(),
#     Contrast(),
#     Identity(),
#     # Rotate(),
#     Sharpness(),
#     # ShearX(),
#     # ShearY(),
#     # Solarize(),
#     TranslateX(),
#     TranslateY(),
# ]

# differentiable_transform_list = [
#     Brightness(),
#     Color(),
#     Contrast(),
#     Identity(),
#     Sharpness(),
#     TranslateX(),
#     TranslateY(),
# ]

transform_list = [
        Brightness(),
        Color(),
        Contrast(),
        Identity(),
        Sharpness(),
        TranslateX(),
        TranslateY(),
        GaussianBlur(),
        RandomRotation(),
        Cutout(size=10),
        # ColorJitter(),
        RandomAffine(),
    ]

class RandAugment_differential:
    def __init__(self, n):
        self.n = n
        self.augment_list = transform_list  # Define this list with your augment functions

    def __call__(self, img):
        # random.seed(0)
        ops = random.sample(self.augment_list, k=self.n)

        # assert self.n == 3
        # img_1 = ops[0](img)
        # img_2 = ops[1](img_1)
        # img_3 = ops[2](img_2)
        #
        # return img_3

        for i, op in enumerate(ops):
            img = op(img)

            # # Save each iteration's image
            # if isinstance(img, Image.Image):  # Check if it's a PIL Image
            #     img.save(f"pil_iter_{i}.png")
            # else:  # Assume it's a tensor and convert to PIL before saving
            #     VF.to_pil_image(img).save(f"tensor_iter_{i}.png")

        return img

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort



color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
# # image cifar augmentation
# image_aug_cifar = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.RandomResizedCrop(size=64),
#             transforms.Compose(color_transform),
#             transforms.RandomRotation(10),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#         ])
# image_no_aug_cifar = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#         ])

# image_waug_cifar_freeMatch = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
#             transforms.RandomApply([transforms.GaussianBlur(kernel_size=64//3*2+1, sigma=(4., 10.))], p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
#         ])

class AnisotropicDiffusion3D:
    def __init__(self, niter=1, kappa=50, gamma=0.1, step=(1., 1., 1.), option=1, ploton=False):
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.step = step
        self.option = option
        self.ploton = ploton

    def __call__(self, stack):
        """
        Apply the 3D anisotropic diffusion on the given stack.
        """
        stack = np.array(stack)
        if stack.ndim == 4:
            warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
            stack = stack.mean(3)

        stack = stack.astype('float32')
        stackout = stack.copy()

        deltaS = np.zeros_like(stackout)
        deltaE = deltaS.copy()
        deltaD = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        UD = deltaS.copy()
        gS = np.ones_like(stackout)
        gE = gS.copy()
        gD = gS.copy()

        if self.ploton:
            showplane = stack.shape[0] // 2
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5.5), num="Anisotropic diffusion")
            ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
            ih = ax2.imshow(stackout[showplane, ...].squeeze(), interpolation='nearest', animated=True)
            ax1.set_title(f"Original stack (Z = {showplane})")
            ax2.set_title("Iteration 0")
            fig.canvas.draw()

        for ii in range(self.niter):
            deltaD[:-1, :, :] = np.diff(stackout, axis=0)
            deltaS[:, :-1, :] = np.diff(stackout, axis=1)
            deltaE[:, :, :-1] = np.diff(stackout, axis=2)

            if self.option == 1:
                gD = np.exp(-(deltaD / self.kappa) ** 2.) / self.step[0]
                gS = np.exp(-(deltaS / self.kappa) ** 2.) / self.step[1]
                gE = np.exp(-(deltaE / self.kappa) ** 2.) / self.step[2]
            elif self.option == 2:
                gD = 1. / (1. + (deltaD / self.kappa) ** 2.) / self.step[0]
                gS = 1. / (1. + (deltaS / self.kappa) ** 2.) / self.step[1]
                gE = 1. / (1. + (deltaE / self.kappa) ** 2.) / self.step[2]

            D = gD * deltaD
            E = gE * deltaE
            S = gS * deltaS

            UD[:] = D
            NS[:] = S
            EW[:] = E
            UD[1:, :, :] -= D[:-1, :, :]
            NS[:, 1:, :] -= S[:, :-1, :]
            EW[:, :, 1:] -= E[:, :, :-1]

            stackout += self.gamma * (UD + NS + EW)

            if self.ploton:
                iterstring = f"Iteration {ii + 1}"
                ih.set_data(stackout[showplane, ...].squeeze())
                ax2.set_title(iterstring)
                fig.canvas.draw()

        return Image.fromarray(stackout.astype(np.uint8))


class AddGaussianNoisePIL(object):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        """
        Initialize the Gaussian noise transform with a range for std and a probability of applying the noise.

        Args:
        - mean (float): Mean of the Gaussian noise.
        - std (float): The maximum standard deviation for the Gaussian noise.
                       The actual std will be chosen uniformly from [0, std].
        - p (float): The probability of applying the noise. Default is 0.5 (50%).
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        """
        Apply Gaussian noise to a PIL image with a given probability.

        Args:
        - img (PIL.Image): Input image.

        Returns:
        - img (PIL.Image): Image with Gaussian noise added (if applied).
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL Image")

        # Apply noise with the given probability
        if np.random.rand() < self.p:
            # Convert image to NumPy array
            img_array = np.array(img).astype(np.float32)

            # Randomly select std from uniform distribution [0, self.std]
            current_std = np.random.uniform(0, self.std)

            # Generate Gaussian noise with the randomly chosen std
            noise = np.random.normal(self.mean, current_std, img_array.shape)

            # Add noise to the image
            img_array += noise

            # Clip values to ensure they remain in valid range [0, 255]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            # Convert NumPy array back to PIL Image
            img = Image.fromarray(img_array)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, max_std={self.std}, p={self.p})'


class AddWhiteNoisePIL(object):
    def __init__(self, noise_range=25.0):
        """
        Initialize the white noise transform with a range for noise intensity.

        Args:
        - noise_range (float): The maximum range for the uniform noise.
                               The actual noise will be chosen uniformly from [-noise_range, noise_range].
        """
        self.noise_range = noise_range

    def __call__(self, img):
        """
        Apply white noise to a PIL image.

        Args:
        - img (PIL.Image): Input image.

        Returns:
        - img (PIL.Image): Image with white noise added.
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL Image")

        # Convert image to NumPy array
        img_array = np.array(img).astype(np.float32)

        # Generate white noise with uniform distribution in the range [-noise_range, noise_range]
        noise = np.random.uniform(-self.noise_range, self.noise_range, img_array.shape)

        # Add noise to the image
        img_array += noise

        # Clip values to ensure they remain in valid range [0, 255]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert NumPy array back to PIL Image
        img_noise = Image.fromarray(img_array)

        return img_noise

    def __repr__(self):
        return f'{self.__class__.__name__}(noise_range={self.noise_range})'


class EdgePreservingBlur(object):
    def __init__(self, diameter=9, sigma_color=75, sigma_space=75):
        """
        Initialize the EdgePreservingBlur with bilateral filter parameters.
        :param diameter: Diameter of each pixel neighborhood.
        :param sigma_color: Filter sigma in the color space.
        :param sigma_space: Filter sigma in the coordinate space.
        """
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img):
        """
        Apply the edge-preserving blur to the image.
        :param img: Input PIL image.
        :return: Blurred image as PIL image.
        """
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Apply bilateral filter
        img_blurred = bilateralFilter(img_np, self.diameter, self.sigma_color, self.sigma_space)

        # Convert back to PIL image
        img_blurred = Image.fromarray(img_blurred)

        return img_blurred

image_no_aug_cifar_freeMatch_tensor = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=10.)], p=0.5),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.5),
            transforms.ToTensor(),
        ])

image_no_aug_cifar_freeMatch = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=10.)], p=0.5),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

image_waug_cifar_freeMatch_orig = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

image_waug_cifar_freeMatch = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=10.)], p=0.9),
            # AddGaussianNoisePIL(mean=0.0, std=45., p=0.9),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.9),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=45.)], p=0.5),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.5),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
            # transforms.RandomResizedCrop(64, scale=(0.6, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])


image_saug_cifar_freeMatch = copy.deepcopy(image_waug_cifar_freeMatch)
image_saug_cifar_freeMatch.transforms.insert(-2, RandAugment(3))



image_no_aug_cifar_freeMatch_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=10.)], p=0.5),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])


image_waug_cifar_freeMatch_224 = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.Resize((224, 224)),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=10.)], p=0.9),
            # AddGaussianNoisePIL(mean=0.0, std=45., p=0.9),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.9),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=45.)], p=0.5),
            # transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=150, sigma_space=150)], p=0.5),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
            # transforms.RandomResizedCrop(64, scale=(0.6, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])


image_saug_cifar_freeMatch_224 = copy.deepcopy(image_waug_cifar_freeMatch_224)
image_saug_cifar_freeMatch_224.transforms.insert(-2, RandAugment(3))


image_no_aug_cifar_freeMatch_blto = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

image_waug_cifar_freeMatch_blto = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([AddWhiteNoisePIL(noise_range=15.)], p=0.5),
            transforms.RandomApply([EdgePreservingBlur(diameter=10, sigma_color=75, sigma_space=75)], p=0.5),
            # transforms.RandomApply([AddWhiteNoisePIL(noise_range=1.)], p=0.9),
            # transforms.RandomApply([EdgePreservingBlur(diameter=5, sigma_color=75, sigma_space=150)], p=0.9),
            # AddWhiteNoisePIL(noise_range=1.),
            # EdgePreservingBlur(diameter=5, sigma_color=75, sigma_space=75),
            # transforms.RandomApply([EdgePreservingBlur(diameter=5, sigma_color=100, sigma_space=75)], p=1),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.9),
            # AddGaussianNoisePIL(mean=0.0, std=100, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

# image_waug_cifar_freeMatch_blto2 = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
#             # AddWhiteNoisePIL(noise_range=1.),
#             # EdgePreservingBlur(diameter=5, sigma_color=75, sigma_space=75),
#             # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.9),
#             # AddGaussianNoisePIL(mean=0.0, std=100, p=0.9),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
#         ])
image_saug_cifar_freeMatch_blto = copy.deepcopy(image_waug_cifar_freeMatch_blto)
image_saug_cifar_freeMatch_blto.transforms.insert(-2, RandAugment(3))
# print(image_waug_cifar_freeMatch_blto)
# print(image_saug_cifar_freeMatch_blto)



image_waug_imagenet_freeMatch = transforms.Compose([
        # transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

image_saug_imagenet_freeMatch = copy.deepcopy(image_waug_imagenet_freeMatch)
image_saug_imagenet_freeMatch.transforms.insert(1, RandAugment(3))

image_no_aug_imagenet_freeMatch = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
        ])

# audio augmentation
time_aug_audio = Compose([
            # LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=4000, p=0.5),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.5),
            AddGaussianSNR(min_snr_db=1, max_snr_db=5, p=0.),
        ])

spec_aug_audio = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=int(552.0/4)),
            T.TimeMasking(time_mask_param=int(100.0/4)),
        )


# Assuming `stack` is a 3D numpy array representing the image stack
# transformed_stack = transform(stack)





if __name__ == '__main__':

    # image_path = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/poisonTrain_pr_0.05_t_9/truck/truck_17639.png'  # Replace with the correct image path
    # image = Image.open(image_path).convert('RGB')
    # image.save('original.png')
    # # Apply the transformations
    # transformed_image = image_waug_cifar_freeMatch(image)
    #
    # # Function to unnormalize the tensor
    # def unnormalize(tensor):
    #     unnormalize_transform = transforms.Normalize(
    #         mean=[-x / 255 for x in [125.3, 123.0, 113.9]],
    #         std=[1 / (x / 255) for x in [63.0, 62.1, 66.7]]
    #     )
    #     return unnormalize_transform(tensor)
    #
    #
    # # Convert the tensor back to a PIL image
    # unnormalized_image = unnormalize(transformed_image)
    # unnormalized_image = unnormalized_image.permute(1, 2, 0)  # C x H x W to H x W x C
    # unnormalized_image = unnormalized_image.clamp(0, 1)  # Clipping to be in the valid range
    # unnormalized_image = (unnormalized_image * 255).byte()  # Convert to byte type
    # pil_image = Image.fromarray(unnormalized_image.numpy())
    #
    # # Save the PIL image
    # output_path = 'transformed_image.png'  # Specify the path to save the transformed image
    # pil_image.save(output_path)
    #
    #
    trans_img = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    trans_img.transforms.insert(1, RandAugment(3))
    trans_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    trans_tensor.transforms.insert(2, RandAugment(3))

    # normalization = transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])

    img_path = '/storageA/david_projects/DefTimeSeries/poisonDataset/adaptiveattack/Train/airplane/10008.png'

    img_pil = Image.open(img_path).convert('RGB')
    # img_pil.save('aug_original.png')
    # for transform in transform_list:
    #     # Apply transform
    #     transformed_img = transform(img_pil)
    #
    #     # Save the transformed image with the transform name
    #     transform_name = transform.__class__.__name__
    #     output_path = f"aug_{transform_name}.png"
    #     transformed_img.save(output_path)
    #     print(f"Saved transformed image: {output_path}")

    #
    aug_img = trans_img(img_pil)
    aug_img.save('aug_img.png')
    aug_tensor = trans_tensor(img_pil)
    img_pil = VF.to_pil_image(aug_tensor)

    # Step 2: Save the PIL image
    img_pil.save("aug_tensor.png")
    # # img_norm = normalization(img_aug)
