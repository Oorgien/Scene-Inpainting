import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F

from base_model.base_blocks import _activation, _norm, _padding, conv_block


class ResConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True, padding=0, norm='in', activation='relu', pad_type='zero'):
        super(ResConv, self).__init__()
        self.activation = _activation(activation)
        self.norm = _norm(norm, channels)
        self.pad = _padding(pad_type, padding)

        self.conv = nn.Sequential(
            self.pad,
            nn.Conv2d(channels, channels, kernel_size, stride, 0, dilation, groups, bias),
            self.norm,
            self.activation,
            self.pad,
            nn.Conv2d(channels, channels, kernel_size, stride, 0, dilation, groups, bias),
            self.norm
        )

    def forward(self, x):
        return x + self.conv(x)
