from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    def __init__(self, im_size, normalization):
        self.im_size = im_size
        if normalization == "tanh":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif normalization == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]


class BaseMaskDataset(Dataset):
    def __init__(self, im_size, multichannel):
        self.multichannel = multichannel
        self.im_size = im_size
