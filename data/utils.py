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
    expand_train = 1
    expand_test = 1
    if args.dataset == 'celeba-hq':
        train_data_dataset = CelebaHQDataset(
            im_size=im_size, image_dir=args.data_train,
            image_list=train_images, mode='train',
            normalization=args.data_normalization)

        test_data_dataset = CelebaHQDataset(
            im_size=im_size, image_dir=args.data_test,
            image_list=test_images, mode='test',
            normalization=args.data_normalization)

    elif args.dataset == 'imagenet':
        train_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='train',
            normalization=args.data_normalization)
        test_data_dataset = ImageNetDataset(
            im_size, txt_files_dir=args.txt_files_dir, train_dir=args.data_train,
            val_dir=args.data_val, test_dir=args.data_test, mode='test',
            normalization=args.data_normalization)
        expand_train = 17
        expand_test = 20
    else:
        raise NameError(f"Unsupported dataset {args.dataset} name")

    # Mask dataset
    if args.mask_dataset == 'nvidia':
        train_mask_dataset = NvidiaMaskDataset(im_size, args.mask_train,
                                               'train', multichannel=False,
                                               expand=expand_train)
        test_mask_dataset = NvidiaMaskDataset(im_size, args.mask_test,
                                              'test', multichannel=False,
                                              expand=expand_test)
    else:
        raise NameError(f"Unsupported mask dataset {args.mask_dataset} name")

    return train_data_dataset, test_data_dataset, train_mask_dataset, test_mask_dataset
