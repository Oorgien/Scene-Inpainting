import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def get_config(config):
    with open(config, 'r') as stream:
        return edict(yaml.load(stream, Loader=yaml.FullLoader))


def same_padding(
        w: int, h: int, ksizes: list,
        strides: list, dilations: list):
    """
    Function to get padding for saving initial image size

    :param w: width of Tensor
    :param h: height of Tensor
    :param ksizes: kernel sizes [ks1, ks2]
    :param strides: strides [s1, s2]
    :param dilations: dilations [d1, d2]
    :return: padding [pad_w, pad_h]
    """
    pad_w = math.floor(((w - 1) * strides[0] + dilations[0] * ksizes[0] - w) / 2)
    pad_h = math.floor(((h - 1) * strides[1] + dilations[1] * ksizes[1] - h) / 2)

    padding = (pad_w, pad_h)

    return padding


def get_pad(w: int, h: int, ksizes: list,
            strides: list, dilations: list):
    """
    Function to get padding to match out shape of image as input shape / stride

    :param w: width of Tensor
    :param h: height of Tensor
    :param ksizes: kernel sizes [ks1, ks2]
    :param strides: strides [s1, s2]
    :param dilations: dilations [d1, d2]
    :return: padding [pad_w, pad_h]
    """
    out_w = np.ceil(float(w) / strides[0])
    out_h = np.ceil(float(w) / strides[1])

    pad_w = int(((out_w - 1) * strides[0] + dilations[0] * (ksizes[0] - 1) + 1 - w) / 2)
    pad_h = int(((out_h - 1) * strides[1] + dilations[1] * (ksizes[1] - 1) + 1 - h) / 2)

    padding = (pad_w, pad_h)

    return padding


def get_pad_tp(w: int, h: int, output_padding: list,
               ksizes: list, strides: list, dilations: list):
    """
    Function to get padding to match out shape of
    ConvTranspose2d as input shape * stride

    :param w: width of Tensor
    :param h: height of Tensor
    :param ksizes: kernel sizes [ks1, ks2]
    :param strides: strides [s1, s2]
    :param dilations: dilations [d1, d2]
    :return: padding [pad_w, pad_h]
    """
    out_w = np.ceil(float(w) * strides[0])
    out_h = np.ceil(float(w) * strides[1])

    pad_w = int(((w - 1) * strides[0] + dilations[0] * (ksizes[0] - 1) + output_padding[0] + 1 - out_w) / 2)
    pad_h = int(((h - 1) * strides[1] + dilations[1] * (ksizes[1] - 1) + output_padding[1] + 1 - out_h) / 2)

    padding = (pad_w, pad_h)

    return padding


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
