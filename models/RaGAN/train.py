import argparse
import numpy as np
import random
import os
import cv2
import shutil
import time
from easydict import EasyDict as edict

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

from models.RaGAN.utils import adjust_learning_rate, save_checkpoint, show, local_crop
from models.RaGAN.model.model import InpaintingDiscriminator, InpaintingGenerator
from models.RaGAN.model.loss import AdversarialLoss, RaGANLoss, FeatureMatchingLoss


def train_epoch(args, train_data_loader, train_mask_loader, epoch, writer):
    # switch to train mode
    generator_loss = []
    discriminator_loss = []

    args.model_G.train()
    args.model_D.train()

    with tqdm(desc="Batch", total=len(train_data_loader)) as progress:
        for i, (image, mask) in enumerate(zip(train_data_loader, train_mask_loader)):
            if (image.shape[0] != mask.shape[0]):
                mask = mask[:image.shape[0]]

            adjust_learning_rate(epoch, args)
            # ------------------
            #  Train Generator
            # ------------------

            args.optimizer_G.zero_grad()

            # Input images and masks
            image = image.to(args.device)
            mask = mask.to(args.device)

            mask_3x = torch.cat((mask, mask, mask), dim=1)
            masked_image = torch.mul(image, mask_3x)

            input = torch.cat((masked_image, mask), dim=1)
            target = image.clone()

            # Prediction
            predicted = args.model_G(input)
            predicted = masked_image + torch.mul(predicted, (torch.ones(mask_3x.shape).to(args.device) - mask_3x))

            # Track images while training
            if args.debug:
                if not os.path.isdir(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_epoch_{epoch}_batch_{i}'):
                    os.makedirs(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_epoch_{epoch}_batch_{i}')
                for k, img in enumerate(predicted.detach().cpu().numpy()):
                    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
                    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
                    img = (img * std + mean)
                    test_img = Image.fromarray((np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8))
                    test_img.save(os.path.join(os.path.join(args.eval_dir, args.model_log_name), f'eval_epoch_{epoch}_batch_{i}', f'debug_image_{k}.jpg'))

            # Local and global discriminators
            target_local, predicted_local = local_crop(target, predicted)
            # if args.debug:
            #     if not os.path.isdir(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_epoch_{epoch}_batch_{i}'):
            #         os.makedirs(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_epoch_{epoch}_batch_{i}')
            #     for k, (tg, pred) in enumerate(zip(target_local.cpu().detach().numpy(), predicted_local.cpu().detach().numpy())):
            #         mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            #         std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            #         pred = (pred * std + mean)
            #         tg = (tg * std + mean)
            #         pred = Image.fromarray((np.transpose(pred,  (1, 2, 0)) * 255).astype(np.uint8))
            #         tg = Image.fromarray((np.transpose(tg, (1, 2, 0)) * 255).astype(np.uint8))
            #         pred.save(os.path.join(os.path.join(args.eval_dir, args.model_log_name), f'eval_epoch_{epoch}_batch_{i}', f'debug_pred_{k}.jpg'))
            #         tg.save(os.path.join(os.path.join(args.eval_dir, args.model_log_name), f'eval_epoch_{epoch}_batch_{i}', f'debug_tg_{k}.jpg'))

            _, local_real_raw, real_dis_raw = args.model_D(target, target_local)

            local_real = []
            for layer in local_real_raw:
                local_real.append(layer.detach())
            real_dis = real_dis_raw.detach()

            _, local_fake, fake_dis = args.model_D(predicted, predicted_local)

            # Compute loss
            l1_loss, self_guided_loss, align_loss = args.ragan_loss(predicted, target)
            adv_loss_G = args.adv_loss_G(real_dis, fake_dis)
            fm_vgg_loss, fm_dis_loss = args.fm_loss(predicted, target, local_real, local_fake)

            loss_G = args.l1_coef * l1_loss + \
                     args.guided_coef * self_guided_loss + \
                     args.align_coef * align_loss + \
                     args.adv_G_coef * adv_loss_G + \
                     args.fm_vgg_coef * fm_vgg_loss + \
                     args.fm_dis_coef * fm_dis_loss

            # Record loss
            generator_loss.append(loss_G.item())

            # Compute gradient and do gradient step
            loss_G.backward(retain_graph=True)
            args.optimizer_G.step()

            # Logging
            counter = epoch * len(train_data_loader) + i
            writer.add_scalar('Generator loss', loss_G, counter)
            writer.add_scalar('Generator train l1 loss', args.l1_coef * l1_loss, counter)
            writer.add_scalar('Generator train self guided loss', args.guided_coef * self_guided_loss, counter)
            writer.add_scalar('Generator train geometrical alignment loss', args.align_coef * align_loss, counter)
            writer.add_scalar('Generator train adversarial loss', args.adv_G_coef * adv_loss_G, counter)
            writer.add_scalar('Generator train feature matching vgg loss', args.fm_vgg_coef * fm_vgg_loss, counter)
            writer.add_scalar('Generator train feature matching discriminator loss', args.fm_dis_coef * fm_dis_loss, counter)

            if counter > args.warmup_batches and epoch % args.train_interval_D == 0:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                args.optimizer_D.zero_grad()

                # Prediction
                _, _, real_dis = args.model_D(target, target_local)
                _, _, fake_dis = args.model_D(predicted.detach(), predicted_local.detach())

                # Compute loss
                loss_D = args.adv_D_coef * args.adv_loss_D(real_dis, fake_dis)
                # Record loss
                discriminator_loss.append(loss_D.item())

                # Compute gradient and do gradient step
                loss_D.backward()
                args.optimizer_D.step()

                # Logging
                writer.add_scalar('Discriminator train loss', loss_D, counter)
            progress.update(1)

    return generator_loss, discriminator_loss


def eval_epoch(args, test_data_loader, test_mask_loader, epoch, writer):
    # switch to evaluate mode
    args.model_D.eval()
    args.model_G.eval()
    generator_loss = []
    discriminator_loss = []

    with torch.no_grad():
        with tqdm(desc="Batch", total=len(test_data_loader)) as progress:
            for i, (image, mask) in enumerate(zip(test_data_loader, test_mask_loader)):
                if (image.shape[0] != mask.shape[0]):
                    mask = mask[:image.shape[0]]
                # ------------------
                #  Test Generator
                # ------------------

                # Input images and masks
                image = image.to(args.device)
                mask = mask.to(args.device)

                mask_3x = torch.cat((mask, mask, mask), dim=1)
                masked_image = torch.mul(image, mask_3x)

                input = torch.cat((masked_image, mask), dim=1)
                target = image.clone()

                # Prediction
                predicted = args.model_G(input)
                predicted = masked_image + torch.mul(predicted, (torch.ones(mask_3x.shape).to(args.device) - mask_3x))

                # Local and global discriminators
                target_local, predicted_local = local_crop(target, predicted)

                _, local_real, real_dis = args.model_D(target, target_local)
                _, local_fake, fake_dis = args.model_D(predicted, predicted_local)

                # Compute loss
                l1_loss, self_guided_loss, align_loss = args.ragan_loss(predicted, target)
                adv_loss_G = args.adv_loss_G(real_dis, fake_dis)
                fm_vgg_loss, fm_dis_loss = args.fm_loss(predicted, target, local_real, local_fake)

                loss_G = args.l1_coef * l1_loss + \
                     args.guided_coef * self_guided_loss + \
                     args.align_coef * align_loss + \
                     args.adv_G_coef * adv_loss_G + \
                     args.fm_vgg_coef * fm_vgg_loss + \
                     args.fm_dis_coef * fm_dis_loss

                # Record loss
                generator_loss.append(loss_G.item())

                progress.update(1)

                # Logging
                counter = epoch * len(test_data_loader) + i
                writer.add_scalar('Generator test loss', loss_G, counter)
                writer.add_scalar('Generator test l1 loss', args.l1_coef * l1_loss, counter)
                writer.add_scalar('Generator test self guided loss', args.guided_coef * self_guided_loss, counter)
                writer.add_scalar('Generator test geometrical alignment loss', args.align_coef * align_loss, counter)
                writer.add_scalar('Generator test adversarial loss', args.adv_G_coef * adv_loss_G, counter)
                writer.add_scalar('Generator test feature matching vgg loss', args.fm_vgg_coef * fm_vgg_loss, counter)
                writer.add_scalar('Generator test feature matching discriminator loss', args.fm_dis_coef * fm_dis_loss, counter)

                # ---------------------
                #  Test Discriminator
                # ---------------------

                # Prediction
                _, _, real_dis = args.model_D(target, target_local)
                _, _, fake_dis = args.model_D(predicted, predicted_local)

                # Compute loss
                loss_D = args.adv_D_coef * args.adv_loss_D(real_dis, fake_dis)

                # Record loss
                discriminator_loss.append(loss_D.item())

                # Logging
                writer.add_scalar('Discriminator test loss', loss_D, counter)

                if (i % args.sample_interval == 0):
                    if not os.path.isdir(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_{epoch}'):
                        os.makedirs(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_{epoch}')
                    t = transforms.ToPILImage()
                    mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
                    std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

                    img = (predicted.cpu() * std + mean)
                    for k, (im, m) in enumerate(zip(img, mask)):
                        t(im).save(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_{epoch}/batch{i}_{k}_img.jpg')
                        t(m.cpu()).save(f'{os.path.join(args.eval_dir, args.model_log_name)}/eval_{epoch}/batch{i}_{k}_mask.jpg')

    return generator_loss, discriminator_loss

def train(args,
          train_data_dataset,
          train_mask_dataset,
          test_data_dataset,
          test_mask_dataset):

    args.model_D = InpaintingDiscriminator(in_nc=3, nf=64).to(args.device)
    args.model_G = InpaintingGenerator(in_nc=4, out_nc=3, nf=64, n_blocks=8).to(args.device)
    args.ragan_loss = RaGANLoss(args.device).to(args.device)
    args.adv_loss_G = AdversarialLoss(args.adv_mode, 'generator', args.device).to(args.device)
    args.adv_loss_D = AdversarialLoss(args.adv_mode, 'discriminator', args.device).to(args.device)
    args.fm_loss = FeatureMatchingLoss(vgg_path=args.vgg_path).to(args.device)
    args.optimizer_D = torch.optim.Adam(args.model_D.parameters(),
                                        betas=args.adam_betas_D,
                                        lr=args.learning_rate_D)
    args.optimizer_G = torch.optim.Adam(args.model_G.parameters(),
                                        betas=args.adam_betas_G,
                                        lr=args.learning_rate_G)

    start = time.time()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_test_loss = checkpoint['best_test_loss']

            args.model_G.load_state_dict(checkpoint['state_dict_G'])
            args.model_D.load_state_dict(checkpoint['state_dict_D'])

            args.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            args.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    train_data_loader = DataLoader(
        train_data_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_data_loader = DataLoader(
        test_data_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, args.model_log_name) if args.logdir != '' else None)

    if not args.mask_dataset == 'nvidia':
        train_mask_loader = DataLoader(
            train_mask_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        test_mask_loader = DataLoader(
            test_mask_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    with tqdm(desc="Epoch", total=args.epochs) as progress:
        for i, epoch in enumerate(range(args.start_epoch, args.start_epoch + args.epochs)):
            if args.mask_dataset == 'nvidia':
                train_mask_loader = DataLoader(
                    train_mask_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
                test_mask_loader = DataLoader(
                    test_mask_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)


            # train for one epoch
            generator_loss_train, discriminator_loss_train = train_epoch(args, train_data_loader,
                                                                        train_mask_loader, epoch, writer)

            # evaluate on validation set
            generator_loss_test, discriminator_loss_test = eval_epoch(args, test_data_loader,
                                                                        test_mask_loader, epoch, writer)

            # remember best avg_test_loss and save checkpoint
            is_best = np.mean(generator_loss_train) < args.best_test_loss
            args.best_test_loss = min(args.best_test_loss, np.mean(generator_loss_train))

            if epoch % args.checkpoint_interval == 0:
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict_G': args.model_G.state_dict(),
                    'state_dict_D': args.model_D.state_dict(),
                    'best_test_loss': args.best_test_loss,
                    'optimizer_G': args.optimizer_G.state_dict(),
                    'optimizer_D': args.optimizer_D.state_dict(),
                }, is_best, filename='checkpoint.pth.tar')

            progress.update(1)

            # logging
            with open(args.logger_fname, "a") as log_file:
                log_file.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {time:.3f} \n'
                    'Loss: Train generator - {train_loss_G:.4f} '
                    'Train discriminator - {train_loss_D:.4f} '
                    'Test generator - {test_loss_G:.4f} '
                    'Test discriminator - {test_loss_D:.4f}\n'.format(
                    epoch+1, i+1, args.epochs, time=time.time()-start,
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
                        epoch+1, i+1, args.epochs, time=time.time()-start,
                        train_loss_G=np.mean(generator_loss_train),
                        train_loss_D=np.mean(discriminator_loss_train),
                        test_loss_G=np.mean(generator_loss_test),
                        test_loss_D=np.mean(discriminator_loss_test)))