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


class NvidiaMaskDataset(BaseMaskDataset):
    def __init__(self, im_size, image_dir, mode, multichannel):
        super(NvidiaMaskDataset, self).__init__(im_size, multichannel)
        self.image_list = sorted(os.listdir(image_dir))
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
