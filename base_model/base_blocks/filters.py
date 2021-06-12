import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F

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
                [0, 0, 0],
                [-47, -162, -47]
            ], dtype=torch.float, device=device,
                requires_grad=False).reshape(1, 1, 3, 3).repeat(filter_c, in_nc, 1, 1)

    def forward(self, x):
        X_grad = F.conv2d(x, self.X_filter, stride=self.stride, padding=self.padding, dilation=self.dilation)
        Y_grad = F.conv2d(x, self.Y_filter, stride=self.stride, padding=self.padding, dilation=self.dilation)
        out = torch.cat([X_grad, Y_grad], dim=1)
        return out