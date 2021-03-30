import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F

from base_model.base_blocks import _norm, _activation, _padding, conv_block

class RConv(nn.Module):
    def __init__(self, layer, buffer, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True, padding=0, norm='in', activation='elu', pad_type='zero'):
        super(RConv, self).__init__()
        self.buffer = buffer
        self.layer = layer
        self.conv = conv_block(in_channels, out_channels, kernel_size, stride, dilation, groups,
                                 bias, padding, "none", "none", pad_type)
        self.fusion = conv_block(out_channels*2, out_channels, kernel_size=1, norm=norm,
                                 activation=activation)
        self.bn_act = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.ELU()
        )

    def forward(self, x, mode):
        if mode == "coarse":
            x = self.conv(x)
            x = self.bn_act(x)
            self.buffer[self.layer] = x
        elif mode == "fine":
            c1 = self.conv(x)
            c2 = self.buffer[self.layer]
            x = torch.cat([c1, c2], dim=1)
            x = self.fusion(x)
        else:
            raise NotImplementedError('Generator mode [{:s}] is not found'.format(mode))
        return x


class ResConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True, padding=0, norm='in', activation='relu', pad_type='zero'):
        super(ResConv, self).__init__()
        self.activation = _activation(activation)
        self.norm = _norm(norm, channels)
        self.pad = _padding(pad_type, padding)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, 0, dilation, groups, bias),
            self.norm,
            self.activation,
            nn.Conv2d(channels, channels, kernel_size, stride, 1, 1, groups, bias),
            self.norm
        )

    def forward(self, x):
        return x + self.conv(self.pad(x))
    

class RDeConv(nn.Module):
    def __init__(self, layer, buffer, in_channels, out_channels, kernel_size=3, stride=2, dilation=1, groups=1,
                 bias=True, padding=1, output_padding=1, norm='in', activation='elu', pad_type='zero'):
        """

        :param layer:
        :param buffer:
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param groups:
        :param bias:
        :param padding:
        :param output_padding:
        :param norm:
        :param activation:
        :param pad_type:
        """
        super(RDeConv, self).__init__()

        self.buffer = buffer
        self.layer = layer
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.act = _activation(activation)
        self.norm = _norm(norm, out_channels)
        self.conv = conv_block(out_channels, out_channels, kernel_size, stride=1, bias=bias,
                               padding=1, pad_type=pad_type, norm=norm, activation=activation)
        self.fusion = conv_block(out_channels * 2, out_channels, kernel_size=1, norm=norm,
                                 activation=activation)

    def forward(self, x, mode):
        x = self.act(self.norm(self.deconv(x)))
        if mode == "coarse":
            x = self.conv(x)
            x = self.act(self.norm(x))
            self.buffer[self.layer] = x
        elif mode == "fine":
            c1 = self.conv(x)
            c2 = self.conv(self.buffer[self.layer])
            x = torch.cat([c1, c2], dim=1)
            x = self.fusion(x)
        return x


class SNBlock(conv_block):
    def __init__(self, *args, **kwargs):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param groups:
        :param bias:
        :param padding:
        :param norm:
        :param activation:
        :param pad_type:
        """
        super(SNBlock, self).__init__(*args, **kwargs)
        self.conv = torch.nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        return super(SNBlock, self).forward(x)


class SobelFilter(nn.Module):
    def __init__(self, device, in_nc=3, filter_c=1, stride=1, padding=0, dilation=1, groups=1, mode="scharr"):
        super(SobelFilter, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if mode == "scharr":
            self.X_filter = torch.tensor([
                [47, 0, -47],
                [162, 0, -162],
                [47, 0, -47]
            ], dtype=torch.float, device=device,
                requires_grad=False).reshape(1, 1, 3, 3).repeat(filter_c, in_nc, 1, 1)

            self.Y_filter = torch.tensor([
                [47, 162, 47],
                [0,  0,   0],
                [-47, -162, -47]
            ], dtype=torch.float, device=device,
                requires_grad=False).reshape(1, 1, 3, 3).repeat(filter_c, in_nc, 1, 1)

    def forward(self, x):
        X_grad = F.conv2d(x, self.X_filter, stride=self.stride, padding=self.padding, dilation=self.dilation)
        Y_grad = F.conv2d(x, self.Y_filter, stride=self.stride, padding=self.padding, dilation=self.dilation)
        out = torch.cat([X_grad, Y_grad], dim=1)
        return out








