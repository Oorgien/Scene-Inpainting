import torch
from torch import nn
from torch.nn import functional as F

import math
from base_model import VGG16Features, gram_matrix
from base_model.base_blocks import get_pad


class L1FrequencyLoss(nn.Module):
    def __init__(self, in_nc, kernel_size, sigma, device):
        """
        :param kernel_size: kernel size for Sobel filter
        :param sigma: standard deviation of normal distribution
        """
        super(L1FrequencyLoss, self).__init__()

        self.in_nc = in_nc
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.low_pass = self.calc_gauss(kernel_size, sigma)
        self.high_pass = torch.ones(self.low_pass.shape, device=device) - self.low_pass

    def calc_gauss(self, kernel_size, sigma):
        X, Y = torch.meshgrid(torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1),
                                           torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1))
        gaussian_filter = torch.exp(-0.5 * (torch.pow(X, 2) + torch.pow(Y, 2))/math.pow(sigma, 2))\
                          / (math.sqrt(2) * math.pow(sigma, 2) * math.pi)
        gaussian_filter /= torch.sum(gaussian_filter)
        return gaussian_filter.reshape(1,1,kernel_size,kernel_size).repeat(1,self.in_nc,1,1).to(self.device)

    def forward(self, predicted, target):
        im_size = predicted.shape[2:]
        coarse_pred = F.conv2d(predicted, self.low_pass, stride=1, padding=get_pad(im_size[0], self.kernel_size, 1))
        coarse_real = F.conv2d(target, self.low_pass, stride=1, padding=get_pad(im_size[0], self.kernel_size, 1))

        fine_pred = F.conv2d(predicted, self.high_pass, stride=1, padding=get_pad(im_size[0], self.kernel_size, 1))
        fine_real = F.conv2d(target, self.high_pass, stride=1, padding=get_pad(im_size[0], self.kernel_size, 1))

        l1_coarse = self.l1_loss(coarse_pred, coarse_real)
        l1_fine = self.l1_loss(fine_pred, fine_real)
        return l1_coarse, l1_fine
    
    
class VggLoss(nn.Module):
    def __init__(self, vgg_path, num_layers):
        super(VggLoss, self).__init__()
        self.vgg_features = VGG16Features(vgg_path=vgg_path, layer_num=5)
        self.num_layers = num_layers

    def compute_vgg(self, x, num_layers):
        """
        :param x: input image
        :param num_layers: number of feature maps
        :type num_layers: list

        :return: list of vgg features of {num_layers} length
        """
        return self.vgg_features(x)[:num_layers]

    def compute_perceptual(self, predicted, target):
        """
        :param predicted: predicted image
        :param target: target image
        :return: Content loss & style loss
        """
        vgg_pred = self.compute_vgg(predicted, self.num_layers)
        with torch.no_grad():
            vgg_gt = self.compute_vgg(target, self.num_layers)

        # Content loss
        Content_loss = 0
        for i in range(len(vgg_gt)):
            num_elements = vgg_gt[i].shape[1] * vgg_gt[i].shape[2] * vgg_gt[i].shape[3]
            w = 1e3 / (vgg_gt[i].shape[1] ** 2)

            loss = (w/num_elements) * F.l1_loss(vgg_pred[i], vgg_gt[i], reduction="sum")
            Content_loss += loss

        # Style loss
        Style_loss = 0
        for i in range(len(vgg_gt)):
            num_elements = vgg_gt[i].shape[1] * vgg_gt[i].shape[2] * vgg_gt[i].shape[3]
            w = 1e3 / (vgg_gt[i].shape[1] ** 2)

            loss = (w/num_elements) * F.l1_loss(gram_matrix(vgg_pred[i]), gram_matrix(vgg_gt[i]), reduction="sum")
            Style_loss += loss

        return Content_loss, Style_loss

    def forward(self, predicted, target):
        return self.compute_perceptual(predicted, target)


class FmLoss(nn.Module):
    def __init__(self, num_layers):
        """
        :param num_layers: num layers of features to compute loss with
        """
        super(FmLoss, self).__init__()
        self.num_layers = num_layers

    def forward(self, predicted, target):
        """
        :param predicted: predicted images
        :type predicted: list

        :param target: target image
        :type target: list

        :return: loss value
        """
        assert self.num_layers <= len(target) and self.num_layers >=0

        Loss_fm = 0
        for i in range(self.num_layers):
            num_elements = target[i].shape[1] * target[i].shape[2] * target[i].shape[3]
            w = 1e3 / (target[i].shape[1] ** 2)
            Loss_fm += (w / num_elements) * F.l1_loss(predicted[i], target[i], reduction='sum')
        return Loss_fm

class FineEdgeLoss(nn.Module):
    def __init__(self, in_nc, kernel_size, sigma, device, vgg_path, num_layers):
        """
        :param in_nc: input number of channels for Sobel filter Gaussian kernel
        :param kernel_size: kernel size for Sobel filter
        :param sigma: sigma for gaussian distribution
        :param device: training device
        :param vgg_path: path for vgg model
        :param num_layers: number of vgg layers to match
        """
        super(FineEdgeLoss, self).__init__()
        self.num_layers = num_layers
        self.l1loss = torch.nn.L1Loss(reduction='mean').to(device)
        self.frequency_loss = L1FrequencyLoss(in_nc, kernel_size, sigma, device)
        self.vgg_loss = VggLoss(vgg_path, num_layers)

    def forward(self, predicted, target):
        l1_loss = self.l1loss(predicted, target)
        low_pass, high_pass = self.frequency_loss(predicted, target)
        content_loss, style_loss = self.vgg_loss(predicted, target)
        return l1_loss, low_pass, high_pass, content_loss, style_loss




