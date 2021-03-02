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

from models.PartialUnet import PartialUnet
from models.partialconv2d import PartialConv2d
from models.loss import VGG16PartialLoss

parser = argparse.ArgumentParser(description='PyTorch Inpainting train')
parser.add_argument('--data_dir', metavar='DIRDATA',
                    help='path to image dataset')
parser.add_argument('--mask_train', metavar='MASKTRAIN',
                    help='path to training mask dataset')
parser.add_argument('--mask_test', metavar='MASKTEST',
                    help='path to testing mask dataset')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-w', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 24)')
parser.add_argument('-b', '--batch_size', default=6, type=int,
                    metavar='N', help='mini-batch size (default: 6)')
parser.add_argument('-chk', '--checkpoint_dir', default='checkpoints',
                    metavar='CHK', help='Checkpoint dir (default: checkpoints)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run (default 150)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--freeze_epoch', default=100, type=int,
                    help='Epoch to freeze bn after (default 100).')
parser.add_argument('--eval_dir', default='eval_epochs', type=str,
                    help='Directory to store evaluation results')

args = parser.parse_args()
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

def subset_inds(dataset, ratio: float):
    return random.choices(range(dataset), k=int(ratio*len(dataset)))

class Dilate(object):
    """Dilate mask with given kernel.

        Args:
            kernel_size(int): size of dilatation
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

class ImageDataset(Dataset):
    def __init__(self, image_dir, images_list, mode):
        self.image_dir = image_dir
        self.mode = mode
        self.image_list = images_list

    def __len__(self):
        return len(self.image_list)

    def load_sample(self, image_name):
        image_path = os.path.join(self.image_dir,
                                image_name)
        image = Image.open(image_path)
        # image.load()
        return image

    def __getitem__(self, idx):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(512),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.RandomApply([transforms.ColorJitter(contrast=0.9)], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=0.5),
                normalize,
            ]),
            'test': transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                normalize])
        }
        transform = data_transforms[self.mode]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.load_sample(self.image_list[idx])
        image = transform(image)
        return image


class MaskDataset(Dataset):
    def __init__(self, image_dir, mode, multichannel=True):
        self.image_dir = image_dir
        self.mode = mode
        self.image_list = sorted(os.listdir(self.image_dir))
        self.multichannel = multichannel

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
                transforms.RandomResizedCrop(512, scale=(0.6, 1.0))
            ]),
            'test': transforms.Compose([
                transforms.Resize(512)
            ])
        }
        transform = data_transforms[self.mode]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.load_sample(self.image_list[idx])
        mask = transform(mask)
        mask = transforms.ToTensor()(np.array(mask))
        mask = (mask < 0.6).float()
        # check mask
        # t = transforms.ToPILImage()
        # t(mask).save(f'masks/img_{idx}.jpg')
        if self.multichannel:
            mask = mask.repeat(3, 1, 1)
        return mask


def train_epoch(train_data_loader, train_mask_loader, model, criterion, optimizer, epoch, device, writer):
    # switch to train mode
    train_losses = []

    model.train()
    with tqdm(desc="Batch", total=len(train_data_loader)) as progress:
        for i, (input, mask) in enumerate(zip(train_data_loader, train_mask_loader)):
            # mask_ids = random.sample(range(len(train_mask_loader)), 6)
            # mask = torch.as_tensor(train_mask_loader[mask_ids])

            optimizer.zero_grad()
            # measure data loading time
            input = input.to(device)
            mask = mask.to(device)
            target = input.clone().detach()

            # compute output
            output, _ = model(input, mask, epoch)
            loss, vgg_loss, style_loss = criterion(output, target)

            # record loss
            train_losses.append(loss.item())

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            progress.update(1)
            writer.add_scalar('Train total loss', loss, epoch)
            writer.add_scalar('Train vgg loss', vgg_loss, epoch)
            writer.add_scalar('Train style loss', style_loss, epoch)

    return train_losses


def eval_epoch(test_data_loader, test_mask_loader, model, criterion, epoch, device, writer):
    # switch to evaluate mode
    model.eval()
    test_losses = []

    with torch.no_grad():
        with tqdm(desc="Batch", total=len(test_data_loader)) as progress:
            for i, (input, mask) in enumerate(zip(test_data_loader, test_mask_loader)):
                # mask_ids = random.sample(range(len(test_data_loader)), 6)
                # mask = torch.as_tensor(test_data_loader[mask_ids])

                input = input.to(device)

                mask = mask.to(device)
                target = input.clone().detach()

                # compute output
                output, _ = model(input, mask, epoch)
                loss, vgg_loss, style_loss = criterion(output, target)

                # measure accuracy and record loss
                test_losses.append(loss.item())

                progress.update(1)
                writer.add_scalar('Test total loss', loss, epoch)
                writer.add_scalar('Test vgg loss', vgg_loss, epoch)
                writer.add_scalar('Test style loss', style_loss, epoch)

                if (i % 100 == 0):
                    if not os.path.isdir(f'eval_{epoch}'):
                        os.makedirs(f'eval_{epoch}')
                    t = transforms.ToPILImage()
                    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
                    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)

                    im = (output[0, :, :, :].cpu() * std + mean)
                    t(im).save(f'{args.eval_dir}eval_{epoch}/{i}_img.jpg')
                    t(mask[0, :, :, :].cpu()).save(f'eval_{epoch}/{i}_mask.jpg')

    return test_losses


def adjust_learning_rate(optimizer, epoch):
    if epoch >= args.freeze_epoch:
        lr = 0.00005
    else:
        lr = 0.0002
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, foldername=args.checkpoint_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(foldername, filename))
    if is_best:
        shutil.copyfile(os.path.join(foldername, filename), os.path.join(foldername, 'model_best.pth.tar'))


def train(args,
          train_data_dataset,
          train_mask_dataset,
          test_data_dataset,
          test_mask_dataset,
          model,
          criterion,
          optimizer,
          device):
    end = time.time()

    train_data_loader = DataLoader(
        train_data_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_data_loader = DataLoader(
        test_data_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    writer = SummaryWriter()

    with tqdm(desc="Epoch", total=args.epochs) as progress:
        for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
            train_mask_loader = DataLoader(
                train_mask_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            test_mask_loader = DataLoader(
                test_mask_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)


            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_losses = train_epoch(train_data_loader, train_mask_loader,
                                       model, criterion, optimizer, epoch+1, device, writer)
            avg_train_loss = np.mean(train_losses)

            # evaluate on validation set
            test_losses = eval_epoch(test_data_loader, test_mask_loader,
                                     model, criterion, epoch+1, device, writer)
            avg_test_loss = np.mean(test_losses)

            # remember best avg_test_loss and save checkpoint
            is_best = avg_test_loss < args.best_test_loss
            args.best_test_loss = min(args.best_test_loss, avg_test_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_test_loss': args.best_test_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, foldername=args.checkpoint_dir, filename='checkpoint.pth.tar')

            progress.update(1)


            # logging
            with open(args.logger_fname, "a") as log_file:
                log_file.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {time:.3f} \t'
                    'Loss: train - {train_loss:.4f} test - {test_loss:.4f}\n'.format(
                    epoch+1, i+1, args.epochs, time=time.time()-end,
                    train_loss=avg_train_loss, test_loss=avg_test_loss))

            tqdm.write('Epoch: [{0}][{1}/{2}]\t'
                       'Time {time:.3f}\t'
                       'Loss: train - {train_loss:.4f} test - {test_loss:.4f}'.format(
                        epoch+1, i+1, args.epochs, time=time.time() - end,
                        train_loss=avg_train_loss, test_loss=avg_test_loss))
            # measure time
            end = time.time()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, PartialConv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, 0, 0.01)
            nn.init.constant(m.bias, 0)


def main():
    args.best_test_loss = np.inf
    args.logger_fname = os.path.join(args.checkpoint_dir, 'loss.txt')
    with open(args.logger_fname, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)


    # prepare image dataset
    images_list = sorted(os.listdir(args.data_dir))
    train_images, test_images = train_test_split(images_list, test_size=0.1)

    train_data_dataset = ImageDataset(args.data_dir, train_images, 'train')
    test_data_dataset = ImageDataset(args.data_dir, test_images, 'test')

    train_mask_dataset = MaskDataset(args.mask_train, 'train')
    test_mask_dataset = MaskDataset(args.mask_test, 'test')

    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('training/val dataset created\n'
                       f'train data size: {len(train_data_dataset)}\t'
                       f'test data size: {len(test_data_dataset)}\n'
                       f'train mask size: {len(train_mask_dataset)}\t'
                       f'test mask size: {len(test_mask_dataset)}\n')

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = PartialUnet(freeze_epoch=args.freeze_epoch).to(device)
    criterion = VGG16PartialLoss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_test_loss = checkpoint['best_test_loss']

            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write(f'model created on device {device}\n')
        log_file.write('started training\n')

    train(args,
          train_data_dataset,
          train_mask_dataset,
          test_data_dataset,
          test_mask_dataset,
          model,
          criterion,
          optimizer,
          device)

if __name__ == "__main__":
    main()
