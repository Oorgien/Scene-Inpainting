import os
import random

from data.dataset import CelebaHQDataset, ImageNetDataset, NvidiaMaskDataset


def subset_inds(dataset, ratio: float):
    return random.choices(range(dataset), k=int(ratio * len(dataset)))


def prepare_data(args):
    # Image dataset

    train_images = sorted(os.listdir(args.data_train))
    test_images = sorted(os.listdir(args.data_test))

    im_size = tuple(args.im_size[1:])
    expand_train_mask = 1
    expand_test_mask = 1
    expand_train_data = 1
    expand_test_data = 1
    if args.dataset == 'celeba-hq':
        if args.mask_dataset == 'nvidia':
            expand_train_data = 1
            expand_test_data = 4
        train_data_dataset = CelebaHQDataset(
            im_size=im_size, image_dir=args.data_train,
            image_list=train_images, mode='train',
            normalization=args.data_normalization,
            expand=expand_train_data)

        test_data_dataset = CelebaHQDataset(
            im_size=im_size, image_dir=args.data_test,
            image_list=test_images, mode='test',
            normalization=args.data_normalization,
            expand=expand_test_data)

    elif args.dataset == 'imagenet':
        train_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='train',
            normalization=args.data_normalization)
        test_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='test',
            normalization=args.data_normalization)
        expand_train_mask = 9
        expand_test_mask = 5
    else:
        raise NameError(f"Unsupported dataset {args.dataset} name")

    # Mask dataset
    if args.mask_dataset == 'nvidia':
        train_mask_dataset = NvidiaMaskDataset(im_size, args.mask_train,
                                               'train', multichannel=False,
                                               expand=expand_train_mask)
        test_mask_dataset = NvidiaMaskDataset(im_size, args.mask_test,
                                              'test', multichannel=False,
                                              expand=expand_test_mask)
    else:
        raise NameError(f"Unsupported mask dataset {args.mask_dataset} name")
    return train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset
