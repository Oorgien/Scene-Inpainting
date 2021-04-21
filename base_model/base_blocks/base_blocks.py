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
    elif norm_type == 'none':
        layer = None
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
    elif act_type == 'tanh':
        layer = nn.Tanh()
    elif act_type == 'none':
        layer = None
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
        raise NotImplementedError("Unsupported padding type: {}".format(pad_type))

    return layer


class conv_block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride=1,
                 dilation=1, groups=1,
                 bias=True, padding=0,
                 norm='in', activation='relu', pad_type='zero'):
        super(conv_block, self).__init__()

        self.pad = _padding(pad_type, padding)
        self.norm = _norm(norm, out_channels)
        self.activation = _activation(activation, neg_slope=0.2, n_prelu=1)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class upconv_block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, output_padding=0,
                 groups=1, bias=True,
                 dilation=1, padding_mode='zeros',
                 norm='none', activation='relu'):
        """
        Input -> conv transpose -> norm -> activation -> output

        learn more :
            https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
            https://github.com/vdumoulin/conv_arithmetic

        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size:  Size of the convolving kernel
        :param stride: controls the stride for the cross-correlation.
        :param padding: controls the amount of implicit zero padding on both sides
            for dilation * (kernel_size - 1) - padding number of points. See note below for details.
        :param output_padding:  Additional size added to one side of each dimension in the output shape.
        :param groups: controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. For example,
        :param bias:
        :param dilation: controls the spacing between the kernel points; also known as the à trous algorithm.
            It is harder to describe, but this link has a nice visualization of what dilation does.
        :param padding_mode: mode of padding
        :param norm: normalization after transposed convolution layer
        :param activation: activation after transposed convolution layer
        """
        super(upconv_block, self).__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )
        self.norm = _norm(norm, out_channels)
        self.activation = _activation(activation, neg_slope=0.2, n_prelu=1)

    def forward(self, x):
        x = self.deconv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
