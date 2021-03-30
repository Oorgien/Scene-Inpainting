import yaml
import torch
import torchvision.utils as vutils
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict

from models.RaGAN import train as ragan_train  # test as ragan_test
from models.FineEdgeGAN import trainer as fine_train
from models.EdgeGAN import trainer as edge_train


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def get_config(config):
    with open(config, 'r') as stream:
        return edict(yaml.load(stream, Loader=yaml.FullLoader))

def train(args,
          train_data_dataset,
          train_mask_dataset,
          test_data_dataset,
          test_mask_dataset):

    if args.model_name == "RaGAN":

        ragan_train.train(
            args,
            train_data_dataset,
            train_mask_dataset,
            test_data_dataset,
            test_mask_dataset)

    elif args.model_name == "FineEdgeGAN":

        trainer = fine_train.FineEdgeGanTrainer(
            args,
            train_data_dataset,
            train_mask_dataset,
            test_data_dataset,
            test_mask_dataset)

        trainer.train()

    elif args.model_name == "EdgeGAN":

        trainer = edge_train.EdgeGanTrainer(
            args,
            train_data_dataset,
            train_mask_dataset,
            test_data_dataset,
            test_mask_dataset)

        trainer.train()

