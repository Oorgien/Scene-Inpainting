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


class ImageNetDataset(BaseImageDataset):
    def __init__(self,
                 im_size, txt_files_dir,
                 train_dir, val_dir, test_dir, mode,
                 train_file='train.txt', val_file='val.txt',
                 test_file='test.txt', normalization="tanh"):
        """
        :param train_dir: path to train data
        :param val_dir: path to val data
        :param test_dir: path to test data
        :param txt_files_dir: path to txt files

        :param im_size: imahe size to feed forward to network
        :param mode: train or test

        :param train_file: txt file with listed training images
        :param val_file: txt file with listed validation images
        :param test_file: txt file with listed testing images

        :param normalization: define way to normalize data (mean and std)
        """
        super(ImageNetDataset, self).__init__(im_size, normalization)
        self.txt_files_dir = txt_files_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.mode = mode
        if mode not in ["train", "test"]:
            raise NameError('Ivalid mode name (please select train or test).')

        self.txt_files = dict()
        self.read_txt()

    def read_txt(self):
        filenames = [self.train_file, self.val_file, self.test_file]
        for filename in filenames:
            with open(os.path.join(self.txt_files_dir, filename)) as f:
                file_txt = f.readlines()
                file_txt = [line.strip().split(" ")[0] for line in file_txt]
                self.txt_files.update({filename: file_txt})
        # self.txt_files[self.train_file] += self.txt_files[self.val_file]

    def __len__(self):
        if self.mode == "train":
            return len(self.txt_files[self.train_file])
        elif self.mode == "test":
            return len(self.txt_files[self.test_file])

    def load_sample(self, image_name):
        if self.mode == "train":
            image_path = os.path.join(self.train_dir,
                                      image_name + '.JPEG')
        elif self.mode == "test":
            image_path = os.path.join(self.test_dir,
                                      image_name + '.JPEG')
        image = Image.open(image_path).convert("RGB")
        # print (image_name, transforms.ToTensor()(image.convert("RGB")).shape)
        return image

    def __getitem__(self, idx):
        normalize = transforms.Normalize(mean=self.mean,
                                         std=self.std)
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(
                    size=(self.im_size[0], self.im_size[1]),
                    pad_if_needed=True,
                    padding_mode="reflect"
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
                # transforms.RandomApply([transforms.ColorJitter(contrast=0.9)], p=0.5),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=0.5),
            ]),
            'test': transforms.Compose([
                transforms.RandomCrop(
                    size=(self.im_size[0], self.im_size[1]),
                    pad_if_needed=True,
                    padding_mode="reflect"
                ),
                transforms.ToTensor(),
                normalize
            ])
        }
        transform = data_transforms[self.mode]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == "train":
            image = self.load_sample(self.txt_files[self.train_file][idx])
        elif self.mode == "test":
            image = self.load_sample(self.txt_files[self.test_file][idx])
        image = transform(image)
        return image
