import argparse
import os
import random
import shutil
import time

import cv2
import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from base_model import RelativisticAdvLoss, trainer
from models.EdgeGAN.model import FineEdgeLoss, FmLoss

from .model import InpaintingGenerator, InpaintingDiscriminator


class BestModelTrainer(trainer):
    def __init__(self, args,
                 train_data_dataset,
                 train_mask_dataset,
                 test_data_dataset,
                 test_mask_dataset):
        super(BestModelTrainer, self).__init__(args,
                                               train_data_dataset,
                                               train_mask_dataset,
                                               test_data_dataset,
                                               test_mask_dataset)

    def init_model(self):
        self.model_G = DistributedDataParallel(
            InpaintingGenerator(
                in_nc=4, c_num=48,
                out_nc=3, res_blocks_num=8,
                dmfb_blocks_num=8,
                norm='in', activation='relu',
                device=self.device).to(self.device),
            device_ids=[self.gpu]
        )

        self.optimizer_G = torch.optim.Adam(
            self.model_G.parameters(),
            betas=self.adam_betas_G,
            lr=self.learning_rate_G
        )
        self.model_D = DistributedDataParallel(
            InpaintingDiscriminator(device=self.device, im_size=self.im_size[-1]).to(self.device),
            device_ids=[self.gpu]
        )

        self.optimizer_D = torch.optim.Adam(
            self.model_D.parameters(),
            betas=self.adam_betas_D,
            lr=self.learning_rate_D
        )

        self.loss = FineEdgeLoss(
            in_nc=3, kernel_size=5, sigma=1, device=self.device,
            vgg_path=self.vgg_path, num_layers=self.num_layers_vgg
        ).to(self.device)

        self.fm_loss = FmLoss(num_layers=self.num_layers_fm).to(self.device)

        self.adv_loss = RelativisticAdvLoss(mode=self.adv_mode, device=self.device).to(self.device)

        if self.init_weights:
            self.initialize_weights(self.model_G, self.init_weights)
            self.initialize_weights(self.model_D, self.init_weights)

    @staticmethod
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

    def train_batch(self, i, epoch, image, mask, generator_loss, discriminator_loss):
        if (image.shape[0] != mask.shape[0]):
            mask = mask[:image.shape[0]]
        # ------------------
        #  Train Generator
        # ------------------
        self.optimizer_G.zero_grad()

        # Input images and masks
        image = image.to(self.device)
        mask = mask.to(self.device)

        mask_3x = torch.cat((mask, mask, mask), dim=1)
        masked_image = torch.mul(image, mask_3x)

        target = image.clone()

        # Prediction
        # print(summary(self.model_G, [masked_image, mask_3x, mask]))
        predicted_coarse, predicted_fine = self.model_G(masked_image, mask_3x, mask)

        target_cropped, predicted_cropped = self.local_crop(target, predicted_fine)
        # print(summary(self.model_D, [target, target_cropped]))

        if self.debug and self.gpu == 0:
            if not os.path.isdir(f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_epoch_{epoch}_batch_{i}'):
                os.makedirs(f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_epoch_{epoch}_batch_{i}')
            for k, img in enumerate(predicted_fine.detach().cpu().numpy()):
                mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
                std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
                img = (img * std + mean)
                test_img = Image.fromarray((np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8))
                test_img.save(
                    os.path.join(os.path.join(self.eval_dir, self.model_log_name), f'eval_epoch_{epoch}_batch_{i}',
                                 f'debug_image_{k}.jpg'))

        # Compute loss
        l1_loss_coarse, low_pass, _, content_loss_coarse, style_loss_coarse = self.loss(predicted_coarse, target)
        l1_loss_fine, _, high_pass, content_loss_fine, style_loss_fine = self.loss(predicted_fine, target)
        dis_out_pred, patch_features_pred = self.model_D(predicted_fine, predicted_cropped)
        dis_out_target_, patch_features_target_ = self.model_D(target, target_cropped)

        patch_features_target = []
        for layer in patch_features_target_:
            patch_features_target.append(layer.detach())
        dis_out_target = dis_out_target_.detach()

        adv_loss_G = self.adv_loss(dis_out_pred, dis_out_target, "generator")
        fm_loss_G = self.fm_loss(patch_features_pred, patch_features_target)

        l1_loss = self.l1_coef * (l1_loss_coarse + l1_loss_fine)
        freq_loss = self.freq_coef * (low_pass + high_pass)
        content_loss = self.content_coef * (content_loss_fine + content_loss_coarse)
        style_loss = self.style_coef * (style_loss_coarse + style_loss_fine)
        adv_loss_G = self.adv_G_coef * adv_loss_G
        fm_loss = self.fm_coef * fm_loss_G

        loss_G = l1_loss + freq_loss + content_loss + style_loss + adv_loss_G + fm_loss

        # Record loss
        generator_loss.append(loss_G.item())

        # Compute gradient and do gradient step

        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        # Logging
        counter = epoch * len(self.train_data_loader) + i
        if self.gpu == 0:
            self.writer.add_scalar('Generator train loss', loss_G, counter)
            self.writer.add_scalar('Generator train l1 loss', l1_loss, counter)
            self.writer.add_scalar('Generator train frequency loss', freq_loss, counter)
            self.writer.add_scalar('Generator train content loss', content_loss, counter)
            self.writer.add_scalar('Generator train style loss', style_loss, counter)
            self.writer.add_scalar('Generator train adv loss', adv_loss_G, counter)
            self.writer.add_scalar('Generator train feature matching loss', fm_loss, counter)

        if counter > self.warmup_batches and epoch % self.train_interval_D == 0:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Compute loss
            dis_out_pred, _ = self.model_D(predicted_fine.detach(), predicted_cropped.detach())
            dis_out_target, _ = self.model_D(target, target_cropped)
            loss_D = self.adv_loss(dis_out_pred, dis_out_target, "discriminator")

            # Record loss
            discriminator_loss.append(loss_D.item())

            # Compute gradient and do gradient step
            loss_D.backward()
            self.optimizer_D.step()

            # Logging
            if self.gpu == 0:
                self.writer.add_scalar('Discriminator train loss', loss_D, counter)

            discriminator_loss.append(loss_D.item())

    def eval_batch(self, i, epoch, image, mask, generator_loss, discriminator_loss):
        if (image.shape[0] != mask.shape[0]):
            mask = mask[:image.shape[0]]
        # ------------------
        #  Test Generator
        # ------------------

        # Input images and masks
        image = image.to(self.device)
        mask = mask.to(self.device)

        mask_3x = torch.cat((mask, mask, mask), dim=1)
        masked_image = torch.mul(image, mask_3x)

        target = image.clone()

        # Prediction
        predicted_coarse, predicted_fine = self.model_G(masked_image, mask_3x, mask)

        target_cropped, predicted_cropped = self.local_crop(target, predicted_fine)

        # Compute loss
        l1_loss_coarse, low_pass, _, content_loss_coarse, style_loss_coarse = self.loss(predicted_coarse, target)
        l1_loss_fine, _, high_pass, content_loss_fine, style_loss_fine = self.loss(predicted_fine, target)
        dis_out_pred, patch_features_pred = self.model_D(predicted_fine, predicted_cropped)
        dis_out_target, patch_features_target = self.model_D(target, target_cropped)

        adv_loss_G = self.adv_loss(dis_out_pred, dis_out_target, "generator")
        fm_loss_G = self.fm_loss(patch_features_pred, patch_features_target)

        l1_loss = self.l1_coef * (l1_loss_coarse + l1_loss_fine)
        freq_loss = self.freq_coef * (low_pass + high_pass)
        content_loss = self.content_coef * (content_loss_fine + content_loss_coarse)
        style_loss = self.style_coef * (style_loss_coarse + style_loss_fine)
        adv_loss_G = self.adv_G_coef * adv_loss_G
        fm_loss = self.fm_coef * fm_loss_G

        loss_G = l1_loss + freq_loss + content_loss + style_loss + adv_loss_G + fm_loss

        # Record loss
        generator_loss.append(loss_G.item())

        # Logging
        counter = epoch * len(self.train_data_loader) + i
        mask_groups = ['(0.01, 0.1]', '(0.1, 0.2]', '(0.2, 0.3]', '(0.3, 0.4]', '(0.4, 0.5]', '(0.5, 0.6]']
        mask_grop_id = (i * self.batch_size * (self.gpus if self.parallel else 1)) // 2000
        if mask_grop_id > 5:
            mask_grop_id = 5
        mask_group = mask_groups[mask_grop_id]
        if self.gpu == 0:
            self.writer.add_scalar(f'Generator test {mask_group} loss', loss_G, counter)
            self.writer.add_scalar(f'Generator test {mask_group} l1 loss', l1_loss, counter)
            self.writer.add_scalar(f'Generator test {mask_group} frequency loss', freq_loss, counter)
            self.writer.add_scalar(f'Generator test {mask_group} content loss', content_loss, counter)
            self.writer.add_scalar(f'Generator test {mask_group} style loss', style_loss, counter)
            self.writer.add_scalar(f'Generator test {mask_group} adv loss', adv_loss_G, counter)
            self.writer.add_scalar(f'Generator test {mask_group} feature matching loss', fm_loss, counter)

        # ---------------------
        #  Test Discriminator
        # ---------------------

        # Compute loss
        dis_out_pred, _ = self.model_D(predicted_fine, predicted_cropped)
        dis_out_target, _ = self.model_D(target, target_cropped)
        loss_D = self.adv_loss(dis_out_pred, dis_out_target, "discriminator")

        # Record loss
        discriminator_loss.append(loss_D.item())

        # Logging
        if self.gpu == 0:
            self.writer.add_scalar(f'Discriminator test {mask_group} loss', loss_D, counter)

        if (i % self.sample_interval == 0 and self.gpu == 0):
            if not os.path.isdir(f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_{epoch}'):
                os.makedirs(f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_{epoch}')
            t = transforms.ToPILImage()
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

            img = (predicted_fine.cpu() * std + mean)

            for k, (im, m) in enumerate(zip(img, mask)):
                t(im).save(f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_{epoch}/batch{i}_{k}_img.jpg')
                t(m.cpu()).save(
                    f'{os.path.join(self.eval_dir, self.model_log_name)}/eval_{epoch}/batch{i}_{k}_mask.jpg')

    def test(self):
        super(BestModelTrainer, self).test()

        ssim = np.zeros((6, 1))
        psnr = np.zeros((6, 1))
        l1 = np.zeros((6, 1))
        l2 = np.zeros((6, 1))

        mask_group = 0
        counter = 0
        group_size = (len(self.test_data_loader) * self.batch_size) // 6

        with torch.no_grad():
            with tqdm(desc="Batch", total=len(self.test_data_loader)) as progress:
                for i, (image, mask) in enumerate(zip(self.test_data_loader, self.test_mask_loader)):

                    if (image.shape[0] != mask.shape[0]):
                        mask = mask[:image.shape[0]]

                    if counter >= 2000:
                        mask_group += 1
                        counter = 0

                    # Input images and masks
                    image = image.to(self.device)
                    mask = mask.to(self.device)

                    mask_3x = torch.cat((mask, mask, mask), dim=1)
                    masked_image = torch.mul(image, mask_3x)
                    target = image.clone()

                    # Prediction
                    _, predicted_fine = self.model_G(masked_image, mask_3x, mask)

                    # output_images = tf.convert_to_tensor(predicted_fine.cpu().detach().numpy())
                    # ground_truths = tf.convert_to_tensor(target.cpu().detach().numpy())
                    output_images = predicted_fine.cpu().detach().numpy()
                    ground_truths = target.cpu().detach().numpy()

                    for output_image, ground_truth in zip(output_images, ground_truths):
                        # [0, 1]

                        output_image = tf.convert_to_tensor(np.array(
                            Image.fromarray(
                                np.transpose(output_image, axes=[1, 2, 0]) * 127.5 + 127.5, 'RGB')
                        ) / 255.0, dtype=tf.float32
                        )
                        ground_truth = tf.convert_to_tensor(np.array(
                            Image.fromarray(
                                np.transpose(ground_truth, axes=[1, 2, 0]) * 127.5 + 127.5, 'RGB')
                        ) / 255.0, dtype=tf.float32
                        )

                        ssim[mask_group] += tf.image.ssim_multiscale(output_image, ground_truth, max_val=1.0)
                        psnr[mask_group] += tf.image.psnr(output_image, ground_truth, max_val=1.0)
                        l1[mask_group] += tf.math.reduce_mean(tf.math.abs(output_image - ground_truth))
                        l2[mask_group] += tf.reduce_mean((output_image - ground_truth) ** 2)

                    for idx, (predicted_, target_, mask_, masked_image_) in enumerate(zip(predicted_fine, target, mask_3x, masked_image)):
                        if not os.path.isdir(f'{os.path.join(self.eval_dir, self.model_log_name)}/result'):
                            os.makedirs(f'{os.path.join(self.eval_dir, self.model_log_name)}/result')

                        t = transforms.ToPILImage()

                        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

                        predicted_ = (predicted_.cpu() * std + mean)
                        target_ = (target_.cpu() * std + mean)
                        mask_ = mask_.cpu()
                        masked_image_ = (masked_image_.cpu() * std + mean)

                        result = torch.cat([predicted_, target_, mask_, masked_image_], dim=2)
                        t(result.squeeze(0)).save(
                            f'{os.path.join(self.eval_dir, self.model_log_name)}'
                            f'/result/{mask_group+1}_{counter + idx + 1}_result.jpg')

                    progress.update(1)
                    counter += self.batch_size
        ssim = ssim / group_size
        psnr = psnr / group_size
        l1 = l1 / group_size
        l2 = l2 / group_size

        print(ssim)
        print(psnr)
        print(l1)
        print(l2)

        # Logging
        with open(self.logger_fname, "a") as log_file:
            log_file.write(f'(0.01, 0.1]: \n\t ssim: {ssim[0]} psnr: {psnr[0]} l1: {l1[0]} l2: {l2[0]}\n')
            log_file.write(f'(0.1, 0.2]: \n\t ssim: {ssim[1]} psnr: {psnr[1]} l1: {l1[1]} l2: {l2[1]}\n')
            log_file.write(f'(0.2, 0.3]: \n\t ssim: {ssim[2]} psnr: {psnr[2]} l1: {l1[2]} l2: {l2[2]}\n')
            log_file.write(f'(0.3, 0.4]: \n\t ssim: {ssim[3]} psnr: {psnr[3]} l1: {l1[3]} l2: {l2[3]}\n')
            log_file.write(f'(0.4, 0.5]: \n\t ssim: {ssim[4]} psnr: {psnr[4]} l1: {l1[4]} l2: {l2[4]}\n')
            log_file.write(f'(0.5, 0.6]: \n\t ssim: {ssim[5]} psnr: {psnr[5]} l1: {l1[5]} l2: {l2[5]}\n')