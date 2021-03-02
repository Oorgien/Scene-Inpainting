import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import sys

from data.utils import prepare_data
from utils import get_config, train

def main(args):
    if args.mode == 'train':
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        args.best_test_loss = np.inf
        args.logger_fname = os.path.join(args.checkpoint_dir + args.model_log_name, args.logger)

        if not os.path.isdir(args.checkpoint_dir + args.model_log_name):
            os.makedirs(args.checkpoint_dir + args.model_log_name)

        args.eval_dir = os.path.join(args.eval_dir, args.eval_dir + "_" + args.model_name)

        with open(args.logger_fname, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training started (%s) ================\n' % now)

        # Prepare image dataset
        train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset = prepare_data(args)

        # Logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write('training/val dataset created\n'
                           f'train data size: {len(train_data_dataset)}\t' 
                           f'test data size: {len(test_data_dataset)}\n'
                           f'train mask size: {len(train_mask_dataset)}\t' 
                           f'test mask size: {len(test_mask_dataset)}\n')

        if not args.parallel:
            args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        elif args.parallel:
            print (f"Multiple GPU devices found: {torch.cuda.device_count()}")
            args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # logging
        with open(args.logger_fname, "a") as log_file:
            log_file.write(f'model created on device {args.device}\n')
            log_file.write('started training\n')

        train(args,
            train_data_dataset,
            train_mask_dataset,
            test_data_dataset,
            test_mask_dataset)

    elif args.mode == 'test':
        pass
    else:
        raise NameError('Running mode is not chosen.')


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