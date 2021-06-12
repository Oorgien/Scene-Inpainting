import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import random
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from data.utils import prepare_data
from models.BestModel import trainer as best_train
from models.EdgeGAN import trainer as edge_train
from models.FineEdgeGAN import trainer as fine_train
from models.MadfGAN import trainer as madf_train
from models.RaGAN import train as ragan_train  # test as ragan_test
from utils import get_config


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    args.logger_fname = os.path.join(args.checkpoint_dir + args.model_log_name, args.logger)
    args.eval_dir = os.path.join(args.eval_dir, args.eval_dir + "_" + args.model_name)

    if not os.path.isdir(args.checkpoint_dir + args.model_log_name):
        os.makedirs(args.checkpoint_dir + args.model_log_name)

    # Prepare image dataset
    train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset = prepare_data(args)

    if not args.parallel:
        args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        args.gpu = 0
    elif args.parallel:
        print(f"Multiple GPU devices found: {torch.cuda.device_count()}")
        args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        args.gpu_ids = list(map(int, args.gpu_id.split(',')))
        args.gpus = len(args.gpu_ids)
        args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = str(args.MASTER_ADDR) if args.MASTER_ADDR is not None else 'localhost'  # 10.241.24.185
    os.environ['MASTER_PORT'] = str(args.MASTER_PORT) if args.MASTER_PORT is not None else '8888'

    fn = None

    if args.mode == 'train':
        # torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        args.best_test_loss = np.inf

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training started (%s) ================\n' % now)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('training/val dataset created\n'
                           f'train data size: {len(train_data_dataset)}\t'
                           f'test data size: {len(test_data_dataset)}\n'
                           f'train mask size: {len(train_mask_dataset)}\t'
                           f'test mask size: {len(test_mask_dataset)}\n')

    fn = None

    if args.mode == 'train':
        # torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        args.best_test_loss = np.inf

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training started (%s) ================\n' % now)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('training/val dataset created\n'
                           f'train data size: {len(train_data_dataset)}\t'
                           f'test data size: {len(test_data_dataset)}\n'
                           f'train mask size: {len(train_mask_dataset)}\t'
                           f'test mask size: {len(test_mask_dataset)}\n')

    fn = None

    if args.mode == 'train':
        # torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        args.best_test_loss = np.inf

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training started (%s) ================\n' % now)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('training/val dataset created\n'
                           f'train data size: {len(train_data_dataset)}\t'
                           f'test data size: {len(test_data_dataset)}\n'
                           f'train mask size: {len(train_mask_dataset)}\t'
                           f'test mask size: {len(test_mask_dataset)}\n')

    fn = None

    if args.mode == 'train':
        # torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        args.best_test_loss = np.inf

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training started (%s) ================\n' % now)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('training/val dataset created\n'
                           f'train data size: {len(train_data_dataset)}\t'
                           f'test data size: {len(test_data_dataset)}\n'
                           f'train mask size: {len(train_mask_dataset)}\t'
                           f'test mask size: {len(test_mask_dataset)}\n')

        # logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write(f'model created on device {args.device}\n')
            log_file.write('started training\n')

        fn = train

    elif args.mode == 'test':
        if not args.resume:
            raise NotImplemented('Please select model to test.')

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Testing started (%s) ================\n' % now)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('testing dataset created\n'
                           f'test data size: {len(test_data_dataset)}\n'
                           f'test mask size: {len(test_mask_dataset)}\n')

            log_file.write(f'model created on device {args.device}\n')
            log_file.write('started testing\n')

        fn = test
    else:
        raise NameError('Running mode is not chosen.')

    if args.parallel:
        mp.spawn(
            fn, nprocs=args.gpus,
            args=(args,
                  train_data_dataset,
                  train_mask_dataset,
                  test_data_dataset,
                  test_mask_dataset))
    else:
        fn(args.gpu_id, args,
           train_data_dataset,
           train_mask_dataset,
           test_data_dataset,
           test_mask_dataset)


def test(gpu, args,
         train_data_dataset,
         train_mask_dataset,
         test_data_dataset,
         test_mask_dataset):
    args.gpu = gpu
    if args.model_name == "BestModel":
        trainer = best_train.BestModelTrainer(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
        trainer.test()


def train(gpu, args,
          train_data_dataset,
          train_mask_dataset,
          test_data_dataset,
          test_mask_dataset):
    args.gpu = gpu
    if args.model_name == "RaGAN":
        ragan_train.train(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
    elif args.model_name == "FineEdgeGAN":
        trainer = fine_train.FineEdgeGanTrainer(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
        trainer.train()
    elif args.model_name == "EdgeGAN":
        trainer = edge_train.EdgeGanTrainer(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
        trainer.train()
    elif args.model_name == "MadfGAN":
        trainer = madf_train.MADFGanTrainer(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
        trainer.train()
    elif args.model_name == "BestModel":
        trainer = best_train.BestModelTrainer(
            args, train_data_dataset, train_mask_dataset, test_data_dataset, test_mask_dataset
        )
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Inpainting scrypt')
    parser.add_argument('-m', '--mode', default='train', type=str,
                        help='Mode to run script in')
    parser.add_argument('-cfg', '--config', type=str,
                        help='Config name with parameters')
    parsed_args = parser.parse_args()
    args = get_config(parsed_args.config)

    args.mode = parsed_args.mode
    args.cfg = parsed_args.config
    main(args)
