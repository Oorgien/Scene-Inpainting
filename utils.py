import yaml
import torch
import torchvision.utils as vutils
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def get_config(config):
    with open(config, 'r') as stream:
        return edict(yaml.load(stream, Loader=yaml.FullLoader))

