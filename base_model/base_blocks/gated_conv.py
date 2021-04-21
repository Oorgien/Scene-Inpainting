import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _activation, _norm, _padding


class GatedConv(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride=1,
                 dilation=1, groups=1,
                 bias=True, padding=0,
                 norm='in', activation='relu', pad_type='zero'):
        super(GatedConv, self).__init__()

        self.pad = _padding(pad_type, padding)
        self.norm = _norm(norm, out_channels)
        self.activation = _activation(activation, neg_slope=0.2, n_prelu=1)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()

    def gated(self, x):
        return self.sigmoid(x)

    def forward(self, inp):
        x = self.conv(self.pad(inp))
        mask = self.mask(inp)
        if self.activation:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        if self.norm:
            x = self.norm(x)
        return x


class SNGatedConv(GatedConv):
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
        super(SNGatedConv, self).__init__(*args, **kwargs)
        self.conv = nn.utils.spectral_norm(self.conv)
        self.mask = nn.utils.spectral_norm(self.mask)

    def forward(self, x):
        return super(SNGatedConv, self).forward(x)
