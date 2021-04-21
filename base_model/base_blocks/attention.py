import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1,
                 dilation=1, fuse_k=3,
                 padding=0, softmax_scale=10.,
                 training=True, fuse=True, device=None):
        """
        Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
             Generative Image Inpainting with Contextual Attention, Yu et al.

        :param ksize: Kernel size for contextual attention.
        :param stride: Stride for extracting patches from b.
        :param dilation: Dilation for matching.
        :param fuse_k:
        :param padding: padding for feature extraction.
        :param softmax_scale:  Scaled softmax for attention.
        :param training:
        :param fuse:
        """
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device = device
        self.training = training

        self.padding = padding

        self.extract_patches = nn.Unfold(
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, b, f, mask=None):
        """
        :param b: Input feature for match (background) - known region.
        :param f: Input feature to match (foreground) - missing region.
        :param mask: Input mask for b, indicating patches not available.
        :return:
        """

        # get shapes
        f_shape = list(f.size())  # batch_size * c * h * w
        b_shape = list(b.size())  # batch_size * c * h * w

        # extract patches from background with stride, padding and dilation
        # raw_w is extracted for reconstruction
        b_weights = self.extract_patches(b)  # [batch_size, C*k*k, L]

        b_weights = b_weights.view(b_shape[0], b_shape[1], self.ksize, self.ksize, -1)
        b_weights = b_weights.permute(0, 4, 1, 2, 3)  # b_weights shape: [batch_size, L, C, k, k]

        # tuple of tensors with size [L, C, k, k] with len = batch_size
        b_groups = torch.split(b_weights, 1, dim=0)

        # split tensors along the batch dimension
        # tuple of tensors with size [C, h, w] with len = batch_size
        f_groups = torch.split(f, 1, dim=0)

        if mask is None:
            mask = torch.zeros(f_shape[0], 1, f_shape[2], f_shape[3])
            if self.device is not None:
                mask = mask.to(self.device)
        else:
            mask_scale = mask.size()[3] // f_shape[3]

            # downscale to match f shape
            # mask = F.interpolate(mask, scale_factor=1 / mask_scale, mode='nearest')
            mask = F.avg_pool2d(mask, kernel_size=4, stride=mask_scale)

        m_shape = list(mask.size())  # c * h * w
        m = self.extract_patches(mask)  # [batch_size, k*k, L]

        m = m.view(m_shape[0], m_shape[1], self.ksize, self.ksize, -1)  # [batch_size, 1, k, k, L]
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [batch_size, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]

        # Hide unknown regions with mask multiplying
        b_groups = b_weights * m.unsqueeze(0)

        # Hide known regions with mask multiplying
        f_groups = f * (torch.ones(*mask.shape).to(self.device) - mask)

        y = []
        offsets = []
        EPS = 1e-4
        for f_i, b_i in zip(f_groups, b_groups):
            # Take l2 norm
            norm = torch.sqrt(torch.pow(b_i, 2).sum(-1, keepdim=True).sum(-1, keepdim=True).sum(-1, keepdim=True) + EPS)
            b_i_normed = b_i / norm
            y_i = F.conv2d(f_i.unsqueeze(0), b_i_normed, stride=self.stride, padding=self.padding)  # y shape: [L, h, w]

            attention = F.softmax(y_i * self.softmax_scale, dim=1)  # attention shape: [L, h, w]

            # Take argmax, meaning: number of patch
            offset = torch.argmax(y_i, dim=1)
            # Reduce dimension number
            attention_reduced = F.conv_transpose2d(input=attention, weight=b_i, padding=1)
            y.append(attention_reduced)
            offsets.append(offset)

        y = torch.cat(y, dim=0)
        offsets = torch.cat(offsets, dim=0).unsqueeze(1)
        return y, offsets


def test_attn(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    contextual = ContextualAttention(stride=1, padding=1, device=device)
    f = torch.rand(size=(10, 64, 64, 64)).to(device)
    mask = torch.rand(size=(10, 1, 256, 256)).to(device)
    attn = contextual(f, f, mask)
    assert attn[0].shape == (10, 64, 64, 64)
    assert attn[1].shape == (10, 1, 64, 64)
    return True
