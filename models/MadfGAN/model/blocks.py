import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F

from base_model.base_blocks import _norm, _activation, _padding, conv_block

class MADF(nn.Module):
    def __init__(self,
                 in_channels_m, out_channels_m,
                 in_channels_e, out_channels_e,
                 kernel_size, stride, padding):
        super(MADF, self).__init__()
        self.conv_m = conv_block(in_channels=in_channels_m,
                                 out_channels=out_channels_m,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 activation="relu")
        self.conv_filters = conv_block( in_channels=out_channels_m,
                                        out_channels=in_channels_e * kernel_size * \
                                                     out_channels_e * kernel_size,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        activation="none")

    def forward(self, x):
        pass