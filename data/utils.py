import numpy as np
import random
import cv2
import os
from data.dataset import NvidiaMaskDataset, CelebaHQDataset
from sklearn.model_selection import train_test_split


def subset_inds(dataset, ratio: float):
    return random.choices(range(dataset), k=int(ratio*len(dataset)))


def prepare_data(args):
    # Image dataset

    train_images = sorted(os.listdir(args.data_train))
    test_images = sorted(os.listdir(args.data_test))

    im_size = tuple(args.im_size[1:])
    if args.dataset == 'celeba-hq':
        train_data_dataset = CelebaHQDataset(im_size, args.data_train,
                                      train_images, 'train', args.dataset)
        test_data_dataset = CelebaHQDataset(im_size, args.data_test,
                                     test_images, 'test', args.dataset)

    # Mask dataset
    if args.mask_dataset == 'nvidia':
        train_mask_dataset = NvidiaMaskDataset(im_size, args.mask_train,
                                     'train', args.mask_dataset, multichannel=False)
        test_mask_dataset = NvidiaMaskDataset(im_size, args.mask_test,
                                    'test', args.mask_dataset, multichannel=False)

    return train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset