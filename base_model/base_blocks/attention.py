import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class SelfAttention(nn.Module):
    def __init__(self, channels, k=8):
        """
        :param channels: in and out number of channels
        :param k: param to reduce channel depth
        """

        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // k, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // k, kernel_size=1)
        self.value = nn.Conv2d(channels, channels // k, kernel_size=1)
        self.out = nn.Conv2d(channels // k, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, width, height = x.size()
        N = width * height
        query = self.query(x).view(batchsize, -1, N).permute(0, 2, 1)  # B * N * C
        key = self.key(x).view(batchsize, -1, N)  # B * C * N

        energy = torch.bmm(query, key)  # B * N * N
        attention = self.softmax(energy)  # B * N * N

        value = self.value(x).view(batchsize, -1, N)  # B * C * N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # out[i,j] = (value[i, :], attention[i, :])
        out = out.view(batchsize, -1, width, height)

        out = self.out(out)
        return out
