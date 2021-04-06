import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from .base_blocks import _activation, _norm


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):

    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class DMFB(nn.Module):
    def __init__(self, in_channels):
        super(DMFB, self).__init__()
        self.c1 = conv_layer(in_channels, in_channels // 4, 3, 1)
        self.d1 = conv_layer(in_channels // 4, in_channels // 4, 3, 1, 1)  # rate = 1
        self.d2 = conv_layer(in_channels // 4, in_channels // 4, 3, 1, 2)  # rate = 2
        self.d3 = conv_layer(in_channels // 4, in_channels // 4, 3, 1, 4)  # rate = 4
        self.d4 = conv_layer(in_channels // 4, in_channels // 4, 3, 1, 8)  # rate = 8
        self.act = _activation('relu')
        self.norm = _norm('in', in_channels)
        self.c2 = conv_layer(in_channels, in_channels, 1, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output
