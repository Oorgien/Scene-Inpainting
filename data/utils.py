import os
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset import CelebaHQDataset, ImageNetDataset, NvidiaMaskDataset


def subset_inds(dataset, ratio: float):
    return random.choices(range(dataset), k=int(ratio * len(dataset)))


def prepare_data(args):
    # Image dataset

    train_images = sorted(os.listdir(args.data_train))
    test_images = sorted(os.listdir(args.data_test))

    im_size = tuple(args.im_size[1:])
    expand_train = 1
    expand_test = 1
    if args.dataset == 'celeba-hq':
        train_data_dataset = CelebaHQDataset(im_size, args.data_train,
                                             train_images, 'train', args.dataset)
        test_data_dataset = CelebaHQDataset(im_size, args.data_test,
                                            test_images, 'test', args.dataset)
    elif args.dataset == 'imagenet':
        train_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='train',
            normalization=args.data_normalization)
        test_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='test',
            normalization=args.data_normalization)
        expand_train = 17
        expand_test = 20

    # Mask dataset
    if args.mask_dataset == 'nvidia':
        train_mask_dataset = NvidiaMaskDataset(im_size, args.mask_train,
                                               'train', multichannel=False,
                                               expand=expand_train)
        test_mask_dataset = NvidiaMaskDataset(im_size, args.mask_test,
                                              'test', multichannel=False,
                                              expand=expand_test)

    return train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset
