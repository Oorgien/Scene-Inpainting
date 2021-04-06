import argparse
import os
import random
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class trainer():
    def __init__(self, args,
                 train_data_dataset,
                 train_mask_dataset,
                 test_data_dataset,
                 test_mask_dataset):
        """
        :param args: args from config
        :type args: EasyDict
        """
        self.__dict__ = args
        self.start = time.time()
        self.train_data_dataset = train_data_dataset
        self.train_mask_dataset = train_mask_dataset
        self.test_data_dataset = test_data_dataset
        self.test_mask_dataset = test_mask_dataset

    def init_model(self):
        pass

    def init_mask_loader(self):
        self.train_mask_loader = DataLoader(
            self.train_mask_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)
        self.test_mask_loader = DataLoader(
            self.test_mask_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)

    def init_data_loader(self):
        self.train_data_loader = DataLoader(
            self.train_data_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)
        self.test_data_loader = DataLoader(
            self.test_data_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)

    def init_logger(self):
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logdir, self.model_log_name) if self.logdir != '' else None)

    def set_to_train(self):
        self.model_G.train()
        self.model_D.train()

    def set_to_eval(self):
        self.model_G.eval()
        self.model_D.eval()

    def initialize_weights(self, model, mode):
        """
        :param model:
        :param mode:
        :return:
        """
        if mode == "kaiming":
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def adjust_learning_rate(self, epoch):
        """
        Sets the learning rate to the initial LR decayed by 2 every self.lr_interval_{model} epochs

        :param epoch: current epoch
        """
        lr_D = self.learning_rate_D * (0.5 ** ((epoch + 1) // self.lr_interval_D))
        lr_G = self.learning_rate_G * (0.5 ** ((epoch + 1) // self.lr_interval_G))
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr_D

    def resume_training(self):
        if os.path.isfile(self.resume):
            print("=> loading checkpoint '{}'".format(self.resume))
            checkpoint = torch.load(self.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_test_loss = checkpoint['best_test_loss']
            self.model_G.load_state_dict(checkpoint['state_dict_G'])
            self.model_D.load_state_dict(checkpoint['state_dict_D'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.resume))
            assert False

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(self.checkpoint_dir + self.model_log_name, filename))
        if is_best:
            shutil.copyfile(os.path.join(self.checkpoint_dir + self.model_log_name, filename),
                            os.path.join(self.checkpoint_dir + self.model_log_name, 'model_best.pth.tar'))

    def save_state(self, epoch, is_best):
        """
        :param epoch: training epoch
        :param is_best: bool indicator of whether current state is the best or not
        """
        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_G': self.model_G.state_dict(),
                'state_dict_D': self.model_D.state_dict(),
                'best_test_loss': self.best_test_loss,
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
            }, is_best, filename='checkpoint.pth.tar')

    def log_after_epoch(self, epoch, i,
                        generator_loss_train,
                        generator_loss_test,
                        discriminator_loss_train,
                        discriminator_loss_test):

        with open(self.logger_fname, "a") as log_file:
            log_file.write('Epoch: [{0}][{1}/{2}]\t'
                           'Time {time:.3f} \n'
                           'Loss: Train generator - {train_loss_G:.4f} '
                           'Train discriminator - {train_loss_D:.4f} '
                           'Test generator - {test_loss_G:.4f} '
                           'Test discriminator - {test_loss_D:.4f}\n'.format(
                               epoch + 1, i + 1, self.epochs, time=time.time() - self.start,
                               train_loss_G=np.mean(generator_loss_train),
                               train_loss_D=np.mean(discriminator_loss_train),
                               test_loss_G=np.mean(generator_loss_test),
                               test_loss_D=np.mean(discriminator_loss_test)))

        tqdm.write('Epoch: [{0}][{1}/{2}]\t'
                   'Time {time:.3f}\t'
                   'Loss: Train generator - {train_loss_G:.4f} '
                   'Train discriminator - {train_loss_D:.4f} '
                   'Test generator - {test_loss_G:.4f} '
                   'Test discriminator - {test_loss_D:.4f}\n'.format(
                       epoch + 1, i + 1, self.epochs, time=time.time() - self.start,
                       train_loss_G=np.mean(generator_loss_train),
                       train_loss_D=np.mean(discriminator_loss_train),
                       test_loss_G=np.mean(generator_loss_test),
                       test_loss_D=np.mean(discriminator_loss_test)))

    def train(self):
        self.init_model()
        self.init_data_loader()
        self.init_mask_loader()
        self.init_logger()

        if self.resume:
            self.resume_training()
        with tqdm(desc="Epoch", total=self.epochs) as progress:
            for i, epoch in enumerate(range(self.start_epoch, self.start_epoch + self.epochs)):
                if self.mask_dataset == 'nvidia':
                    self.init_mask_loader()
                self.adjust_learning_rate(epoch)

                # train for one epoch
                generator_loss_train, discriminator_loss_train = self.train_epoch(epoch)

                # evaluate on validation set
                generator_loss_test, discriminator_loss_test = self.eval_epoch(epoch)

                # remember best avg_test_loss and save checkpoint
                is_best = np.mean(generator_loss_train) < self.best_test_loss
                self.best_test_loss = min(self.best_test_loss, np.mean(generator_loss_train))

                self.save_state(epoch, is_best)

                progress.update(1)

                # logging
                self.log_after_epoch(epoch, i,
                                     generator_loss_train,
                                     generator_loss_test,
                                     discriminator_loss_train,
                                     discriminator_loss_test)

    def test(self):
        pass

    def train_epoch(self, epoch):
        # switch to train mode
        generator_loss = []
        discriminator_loss = []
        self.set_to_train()
        with tqdm(desc="Batch", total=len(self.train_data_loader)) as progress:
            for i, (image, mask) in enumerate(zip(self.train_data_loader, self.train_mask_loader)):
                self.train_batch(i, epoch, image, mask, generator_loss, discriminator_loss)
                progress.update(1)

        return generator_loss, discriminator_loss

    def eval_epoch(self, epoch):
        generator_loss = []
        discriminator_loss = []
        self.set_to_eval()
        with torch.no_grad():
            with tqdm(desc="Batch", total=len(self.test_data_loader)) as progress:
                for i, (image, mask) in enumerate(zip(self.test_data_loader, self.test_mask_loader)):

                    self.eval_batch(i, epoch, image, mask, generator_loss, discriminator_loss)
                    progress.update(1)

        return generator_loss, discriminator_loss

    def train_batch(self, i, epoch, image, mask, generator_loss, discriminator_loss):
        pass

    def eval_batch(self, i, epoch, image, mask, generator_loss, discriminator_loss):
        pass
