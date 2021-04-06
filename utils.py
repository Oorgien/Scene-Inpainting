import os

import cv2
import numpy as np
import torch
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
