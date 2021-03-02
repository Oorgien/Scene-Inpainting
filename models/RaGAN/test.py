import argparse
import numpy as np
import random
import os
import cv2
import shutil
import time

from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.PartialUnet import PartialUnet
from models.partialconv2d import PartialConv2d
from models.loss import VGG16PartialLoss

parser = argparse.ArgumentParser(description='PyTorch Inpainting test')
parser.add_argument('--mode', default=0, type=int,
                    help='dir=0 or file=1')
parser.add_argument('--data_dir', metavar='TESTDATA',
                    default='', type=str,
                    help='path to test data if mode=0')
parser.add_argument('--mask_dir', metavar='TESTMASK',
                    default='', type=str,
                    help='path to test masks if mode=0')
parser.add_argument('--image_name', metavar='IMFILE',
                    help='path to test image if mode=1')
parser.add_argument('--mask_name', metavar='MASKFILE',
                    help='path to test mask if mode=1')
parser.add_argument('--res_dir', metavar='RESDATA',
                    help='path to result dir')
parser.add_argument('--model_path', metavar='MODELPATH',
                    help='path to saved model')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed.')
args = parser.parse_args()
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

def loader(image_path, mask_path):
    mask = Image.open(mask_path)
    image = Image.open(image_path)
    im_size = image.size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                normalize])
    mask_transform = transforms.Compose([
                transforms.Resize((512, 512))
            ])
    mask = mask_transform(mask)
    mask = transforms.ToTensor()(np.array(mask))
    mask = (mask < 0.6).float()
    image = image_transform(image)
    return image, mask, im_size

def unloader(image, im_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(im_size),
        transforms.ToTensor()])
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.squeeze(0).cpu() * std + mean
    image = transform(torch.clamp(image, 0, 1))
    return image


def rid_of_nonimages(images_names):
    res = []
    for name in images_names:
        head_tail = os.path.split(name)
        if head_tail[1].split('.')[-1] in ['jpg', 'bmp', 'png', 'jpeg']:
            res.append(name)
    return res


def eval(images_names, masks_names, model, epoch, device):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(desc="Step", total=len(images_names)) as progress:
            for i, (input_path, mask_path) in enumerate(zip(images_names, masks_names)):
                input, mask, im_size = loader(os.path.join(args.data_dir, input_path), os.path.join(args.mask_dir, mask_path))
                im_size = (im_size[1], im_size[0])
                input = input.to(device)

                mask = mask.to(device)
                if mask.shape[0] == 1:
                    mask = mask.repeat(3, 1, 1)

                # compute output
                output, _ = model(input.unsqueeze(0), mask.unsqueeze(0), epoch)

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(im_size),
                    transforms.ToTensor()
                ])

                mask = transform(mask.cpu())
                # mask = np.array(mask, dtype=np.uint8).reshape(-1, im_size[0], im_size[1])
                progress.update(1)

                if not os.path.isdir(f'{args.res_dir}'):
                    os.makedirs(f'{args.res_dir}')

                output = unloader(output, im_size)
                input = unloader(input, im_size)
                ind = (mask == 0)
                masked_input = input.clone()
                masked_input[ind] = 1

                img = torch.Tensor(np.concatenate((input, output, masked_input, mask), axis=2))
                transform = transforms.ToPILImage()
                transform(img).save(f'{args.res_dir}/{i}_vis.jpg')


def main():
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = PartialUnet(freeze_epoch= 1000).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        args.epoch = checkpoint['epoch']
        args.best_test_loss = checkpoint['best_test_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))
        assert False

    model.eval()

    if args.mode == 0:
        images_names, masks_names = sorted(os.listdir(args.data_dir)), sorted(os.listdir(args.mask_dir))
    else:
        images_names, masks_names = list([args.image_name]), list([args.mask_name])
    images_names, masks_names = rid_of_nonimages(images_names), rid_of_nonimages(masks_names)

    eval(images_names, masks_names, model, args.epoch, device)


if __name__ == "__main__":
    main()