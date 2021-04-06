import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import cv2
import os

from .base_dataset import BaseImageDataset, BaseMaskDataset


class Dilate(object):
    """
    Dilate mask with given kernel.

    :param: kernel_size(int): size of dilatation
    """

    def __init__(self, kernel_size: tuple):
        assert isinstance(kernel_size, tuple)
        self.kernel_size = random.randint(kernel_size[0], kernel_size[1])

    def __call__(self, mask):
        dilatation_type = cv2.MORPH_ELLIPSE

        element = cv2.getStructuringElement(dilatation_type, (2 * self.kernel_size + 1, 2 * self.kernel_size + 1),
                                            (self.kernel_size, self.kernel_size))
        dilatation_dst = Image.fromarray(cv2.dilate(np.asarray(mask).astype(np.uint8), element))

        return dilatation_dst


class CelebaHQDataset(BaseImageDataset):
    def __init__(self, im_size, mode, image_dir, image_list, normalization="tanh"):
        super(CelebaHQDataset, self).__init__(im_size, normalization)
        self.image_dir = image_dir
        self.image_list = image_list
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def load_sample(self, image_name):
        image_path = os.path.join(self.image_dir,
                                image_name)
        image = Image.open(image_path)
        return image

    def __getitem__(self, idx):
        normalize = transforms.Normalize(mean=self.mean,
                                         std=self.std)
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(self.im_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
                # transforms.RandomApply([transforms.ColorJitter(contrast=0.9)], p=0.5),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=0.5),
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.im_size),
                transforms.ToTensor(),
                normalize])
        }
        transform = data_transforms[self.mode]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.load_sample(self.image_list[idx])
        image = transform(image)
        return image


class NvidiaMaskDataset(Dataset):
    def __init__(self, im_size, mode, multichannel=True):
        super(NvidiaMaskDataset, self).__init__(im_size, image_dir, multichannel)
        self.images_list = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.mode = mode
        self.im_size = im_size

    def __len__(self):
        return len(self.image_list)

    def load_sample(self, mask_name):
        mask_path = os.path.join(self.image_dir,
                                mask_name)
        mask = np.asarray(Image.open(mask_path))
        if self.mode == 'train':
            mask = (mask < 153).astype(float)
        elif self.mode == 'test':
            mask = (mask > 153).astype(float)
        mask = Image.fromarray(mask * 255)
        return mask

    def __getitem__(self, idx):
        data_transforms = {
            'train': transforms.Compose([
                Dilate(kernel_size=(9, 49)),
                transforms.RandomRotation(degrees=(-180, 180)),
                # transforms.RandomCrop(512),
                transforms.RandomResizedCrop(self.im_size, scale=(0.6, 1.0))
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.im_size)
            ])
        }
        transform = data_transforms[self.mode]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.load_sample(self.image_list[idx])
        mask = transform(mask)
        mask = transforms.ToTensor()(np.array(mask))
        mask = (mask < 0.6).float()
        if self.multichannel:
            mask = mask.repeat(3, 1, 1)
        return mask