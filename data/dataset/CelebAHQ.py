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


class CelebaHQDataset(BaseImageDataset):
    def __init__(self, im_size, mode, image_dir, image_list, normalization="tanh", expand=1):
        super(CelebaHQDataset, self).__init__(im_size, normalization)
        self.image_dir = image_dir
        self.image_list = image_list * expand if expand > 1 else image_list[:round(len(image_list) * expand)]
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
                # transforms.RandomResizedCrop(size=self.im_size, scale=(0.5, 1.0)),
                transforms.Resize(self.im_size),
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
