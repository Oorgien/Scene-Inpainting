import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from data.base_dataset import BaseImageDataset, BaseMaskDataset


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


class NvidiaMaskDataset(BaseMaskDataset):
    def __init__(self, im_size, image_dir, mode, multichannel, expand):
        super(NvidiaMaskDataset, self).__init__(im_size, multichannel)
        self.image_list = sorted(os.listdir(image_dir)) * expand
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
