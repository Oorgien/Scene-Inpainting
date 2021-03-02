import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo


def make_vgg16_layers(style_avg_pool = False):
    """
    make_vgg16_layers
    Return a custom vgg16 feature module with avg pooling
    """
    vgg16_cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        512, 512, 512, 'M', 512, 512, 512, 'M'
    ]

    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            if style_avg_pool:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Features(nn.Module):
    """
    VGG16 partial model
    """
    def __init__(self, vgg_path, layer_num=3):
        """
        Init
        :param layer_num: number of layers
        """
        super(VGG16Features, self).__init__()
        vgg_model = models.vgg16()
        vgg_model.features = make_vgg16_layers()
        try:
            vgg_model.load_state_dict(
                torch.load(vgg_path, map_location='cpu')
            )
            print (f"Model loaded from {vgg_path}")
        except:
            vgg_model.load_state_dict(
                model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
            )
            print(f"Model loaded from 'https://download.pytorch.org/models/vgg16-397923af.pth' url")

        vgg_pretrained_features = vgg_model.features

        assert layer_num > 0
        assert isinstance(layer_num, int)
        self.layer_num = layer_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # 4
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # (5, 9)
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):  # (10, 16)
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):  # (17, 23)
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 4:
            self.slice5 = torch.nn.Sequential()
            for x in range(24, 31):  # (24, 30)
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        """
        Forward, get features used for perceptual loss
        :param x: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: list of self.layer_num feature maps used to compute the
         perceptual loss
        """
        h = self.slice1(x)
        h1 = h

        output = []

        if self.layer_num == 1:
            output = [h1]
        elif self.layer_num == 2:
            h = self.slice2(h)
            h2 = h
            output = [h1, h2]
        elif self.layer_num == 3:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            output = [h1, h2, h3]
        elif self.layer_num == 4:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            output = [h1, h2, h3, h4]
        elif self.layer_num >= 5:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            h = self.slice5(h)
            h5 = h
            output = [h1, h2, h3, h4, h5]
        return output


class RelativisticAdvLoss(nn.Module):
    def __init__(self, mode, device):
        """
        :param mode: mode to compute adversarial loss: bce or l1
        :param device: device
        """
        super(RelativisticAdvLoss, self).__init__()
        self.mode = mode
        self.device = device
        self.BCEloss = nn.BCEWithLogitsLoss().to(device)

    def forward(self, fake_dis, real_dis, model):
        """
        :param fake_dis: predicted image from generator output
        :param real_dis: real image
        :param model: generator or discriminator mode
        :return: adversarial loss value
        """

        mean_fake = torch.mean(fake_dis, dim=0, keepdim=True)
        mean_real = torch.mean(real_dis, dim=0, keepdim=True)

        D_real = real_dis - mean_fake
        D_fake = fake_dis - mean_real

        # -> max for discriminator
        # -> min for generator

        zeros = torch.zeros(real_dis.shape[0], 1).to(self.device)
        ones = torch.ones(real_dis.shape[0], 1).to(self.device)

        if model == 'discriminator':
            if self.mode == "bce":
                L_adv = self.BCEloss(D_real, ones) + self.BCEloss(D_fake, zeros)
            elif self.mode == "l1":
                L_adv = torch.mean(torch.abs(D_real - ones) + torch.abs(D_fake))

        elif model == 'generator':
            if self.mode == "bce":
                L_adv = self.BCEloss(D_fake, ones) + self.BCEloss(D_real, zeros)
            elif self.mode == "l1":
                L_adv = torch.mean(torch.abs(D_fake - ones) + torch.abs(D_real))
        else:
            raise NotImplementedError(f'{self.mode} part of model is not defined.')

        return L_adv


def gram_matrix(input_tensor):
    """
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1. / (ch * h * w), out=None)

    # naive way to avoid underflow for mixed precision training
    # features = features / (ch * h)
    # gram = features.bmm(features_t) / w

    # for fp32 training, it is also safe to use the following:
    # gram = features.bmm(features_t) / (ch * h * w)

    return gram
