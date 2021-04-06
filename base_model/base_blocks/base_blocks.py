import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Parameter


def get_pad(in_size, kernel_size, stride, dilation=1):
    out_size = np.ceil(float(in_size) / stride)
    return int(((out_size - 1) * stride + dilation * (kernel_size - 1) + 1 - in_size) / 2)


def _norm(norm_type, channels):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(channels, affine=True)
    elif norm_type == 'in':
        layer = nn.InstanceNorm2d(channels, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def _activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'elu':
        layer = nn.ELU()
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer


def _padding(pad_type, padding):
    if pad_type == 'zero':
        layer = nn.ZeroPad2d(padding)
    elif pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    return layer


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True, padding=0, norm='in', activation='relu', pad_type='zero'):
        super(conv_block, self).__init__()

        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported norm type: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class upconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upconv_stride=2, padding=0,
                 bias=True, pad_type='zero', norm='none', activation='relu'):
        super(upconv_block, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=upconv_stride, padding=1)
        self.act = _activation('relu')
        self.norm = _norm('in', out_channels)

        self.conv = conv_block(out_channels, out_channels, kernel_size, stride, bias=bias,
                               padding=padding, pad_type=pad_type, norm=norm, activation=activation)

    def forward(self, x):
        x = self.act(self.norm(self.deconv(x)))
        x = self.conv(x)
        return x
