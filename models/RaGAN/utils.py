import os
import random
import shutil

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image


def show(img):
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.show()


def adjust_learning_rate(epoch, args):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr_D = args.learning_rate_D * (0.5 ** ((epoch + 1) // args.lr_interval_D))
    lr_G = args.learning_rate_G * (0.5 ** ((epoch + 1) // args.lr_interval_G))
    for param_group in args.optimizer_G.param_groups:
        param_group['lr'] = lr_G
    for param_group in args.optimizer_D.param_groups:
        param_group['lr'] = lr_D


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.checkpoint_dir + args.model_log_name, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.checkpoint_dir + args.model_log_name, filename), os.path.join(args.checkpoint_dir + args.model_log_name, 'model_best.pth.tar'))


def local_crop(target_batch, predicted_batch):
    target_cropped = []
    predicted_cropped = []
    for target, predicted in zip(target_batch, predicted_batch):
        im_size = target.shape[1:]
        transform = transforms.Resize(size=im_size, interpolation=Image.BICUBIC)
        scale = random.uniform(0.2, 0.5)
        x = random.randint(0, int((1 - scale) * im_size[0]))
        y = random.randint(0, int((1 - scale) * im_size[1]))

        target = target[:, y:y + int(scale * im_size[1]), x:x + int(scale * im_size[0])]
        predicted = predicted[:, y:y + int(scale * im_size[1]), x:x + int(scale * im_size[0])]
        target_cropped.append(transform(target).unsqueeze(0))
        predicted_cropped.append(transform(predicted).unsqueeze(0))
    return torch.cat(target_cropped), torch.cat(predicted_cropped)
