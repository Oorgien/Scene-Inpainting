import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils import get_pad, reduce_mean, reduce_sum, same_padding

from .base_blocks import _padding


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
    def __init__(self, ksize=3, stride=1, rate=2,
                 dilation=1, fuse_k=3, softmax_scale=10.,
                 training=True, fuse=True, device=None):
        """
        Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
             Generative Image Inpainting with Contextual Attention, Yu et al.

        :param ksize: Kernel size for contextual attention.
        :param stride: Stride for extracting patches from b.
        :param dilation: Dilation for matching.
        :param fuse_k: kernel size for fusion step
        :param softmax_scale:  Scaled softmax for attention.
        :param training:
        :param fuse: True or False
        """
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.dilation = dilation
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device = device
        self.training = training

    def extract_patches(self, x,
                        kernel_size, stride,
                        dilation, padding='same'):

        if padding == 'same':
            pad_fn = same_padding
        elif padding == 'valid':
            pad_fn = get_pad
        else:
            raise NotImplementedError('Padding mode [{:s}] is not found'.format(padding))

        pad_size = pad_fn(x.shape[2], x.shape[3],
                          [kernel_size, kernel_size],
                          [stride, stride],
                          [dilation, dilation])

        padding_layer = _padding(pad_type='zero', padding=pad_size * 2)

        x = padding_layer(x)
        unfold = nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

        patches = unfold(x)
        return patches

    @staticmethod
    def make_color_wheel():
        RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])
        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY
        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG
        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC
        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB
        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += + BM
        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255
        return colorwheel

    def compute_color(self, u, v):
        h, w = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0
        # colorwheel = COLORWHEEL
        colorwheel = self.make_color_wheel()
        ncols = np.size(colorwheel, 0)
        rad = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1) + 1
        k0 = np.floor(fk).astype(int)
        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0
        for i in range(np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1
            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)
            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
        return img

    def flow_to_image(self, flow):
        """Transfer flow map to image.
        Part of code forked from flownet.
        """
        out = []
        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.
        maxrad = -1
        for i in range(flow.shape[0]):
            u = flow[i, :, :, 0]
            v = flow[i, :, :, 1]
            idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
            u[idxunknow] = 0
            v[idxunknow] = 0
            maxu = max(maxu, np.max(u))
            minu = min(minu, np.min(u))
            maxv = max(maxv, np.max(v))
            minv = min(minv, np.min(v))
            rad = np.sqrt(u ** 2 + v ** 2)
            maxrad = max(maxrad, np.max(rad))
            u = u / (maxrad + np.finfo(float).eps)
            v = v / (maxrad + np.finfo(float).eps)
            img = self.compute_color(u, v)
            out.append(img)
        return np.float32(np.uint8(out))

    def forward(self, b, f, mask=None):
        """
        :param b: Input feature for match (background) - known region.
        :param f: Input feature to match (foreground) - missing region.
        :param mask: Input mask for b, indicating patches not available.
        :return:
        """

        # get shapes
        f_shape_raw = list(f.size())  # batch_size * c * h * w
        b_shape_raw = list(b.size())  # batch_size * c * h * w

        kernel_size = 2 * self.rate

        # extract patches from background with stride, padding and dilation
        # raw_w is extracted for reconstruction
        raw_w = self.extract_patches(
            b, kernel_size,
            self.rate * self.stride,
            self.dilation, padding='valid')  # [batch_size, C*k*k, L]

        raw_w = raw_w.view(b_shape_raw[0], b_shape_raw[1], kernel_size, kernel_size, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # b_weights shape: [batch_size, L, C, k, k]

        # tuple of tensors with size [L, C, k, k] with len = batch_size
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')

        f_shape = list(f.size())  # b*c*h*w
        b_shape = list(b.size())

        # split tensors along the batch dimension
        # tuple of tensors with size [C, h, w] with len = batch_size
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension

        # w shape: [N, C*k*k, L]
        w = self.extract_patches(
            b, self.ksize,
            self.stride, 1,
            padding='same')

        # w shape: [N, C, k, k, L]
        w = w.view(b_shape[0], b_shape[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        if mask is None:
            mask = torch.zeros(f_shape[0], 1, f_shape[2], f_shape[3])
            if self.device is not None:
                mask = mask.to(self.device)
        else:
            mask_scale = mask.size()[3] // f_shape[3]

            # downscale to match f shape
            mask = F.interpolate(mask, scale_factor=1 / mask_scale, mode='nearest')
            # mask = F.avg_pool2d(mask, kernel_size=4, stride=mask_scale)

        m_shape = list(mask.size())  # c * h * w
        m = self.extract_patches(
            mask, self.ksize,
            self.stride, 1,
            padding='same')  # [batch_size, k*k, L]

        m = m.view(m_shape[0], m_shape[1], self.ksize, self.ksize, -1)  # [batch_size, 1, k, k, L]
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [batch_size, L, C, k, k]
        # m = m[0]  # m shape: [L, C, k, k]

        # 0 for patches where all values are 0
        # 1 for patches with non-zero mean
        # mm shape: [batch_size, L, 1, 1, 1]

        mm = (reduce_mean(m, axis=[2, 3, 4], keepdim=True) == 1.).to(torch.float32)
        # mm shape: [batch_size, 1, L, 1, 1]
        mm = mm.permute(0, 2, 1, 3, 4)

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        # Diagonal matrix with shape k * k
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.device:
            fuse_weight = fuse_weight.to(self.device)
        EPS = torch.FloatTensor([1e-4]).to(self.device)
        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm):
            """
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            """
            # Normalizing weight tensor

            wi = wi.squeeze(0)
            wi_norm = torch.sqrt(reduce_sum(torch.pow(wi, 2) + EPS, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / wi_norm

            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi_pad = same_padding(xi.shape[0], xi.shape[1],
                                  [self.ksize, self.ksize],
                                  [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1, padding=xi_pad)  # [1, L, H, W]

            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # Convolution with diagonal shaped kernel №1
                yi = yi.view(1, 1, b_shape[2] * b_shape[3], f_shape[2] * f_shape[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi_pad = same_padding(yi.shape[0], yi.shape[1], [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=yi_pad)  # (B=1, C=1, H=32*32, W=32*32)

                # Convolution with diagonal shaped kernel №2
                yi = yi.contiguous().view(1, b_shape[2], b_shape[3], f_shape[2], f_shape[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, b_shape[2] * b_shape[3], f_shape[2] * f_shape[3])
                yi_pad = same_padding(yi.shape[0], yi.shape[1], [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=yi_pad)

                yi = yi.contiguous().view(1, b_shape[3], b_shape[2], f_shape[3], f_shape[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            yi = yi.view(1, b_shape[2] * b_shape[3], f_shape[2], f_shape[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mi
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi  # [1, L, H, W]
            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W
            if b_shape != f_shape:
                # Normalize the offset value to match foreground dimension
                times = float(f_shape[2] * f_shape[3]) / float(b_shape[2] * b_shape[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // f_shape[3], offset % b_shape[3]], dim=1)  # 1*2*H*W
            # deconv for patch pasting
            wi_center = raw_wi[0]

            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(f_shape_raw)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(f_shape[0], 2, *f_shape[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(f_shape[2]).view([1, 1, f_shape[2], 1]).expand(f_shape[0], -1, -1, f_shape[3])
        w_add = torch.arange(f_shape[3]).view([1, 1, 1, f_shape[3]]).expand(f_shape[0], -1, f_shape[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        ref_coordinate = ref_coordinate.to(self.device)

        offsets = offsets - ref_coordinate
        flow = torch.from_numpy(self.flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        flow = flow.to(self.device)

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate * 4, mode='nearest')

        return y, flow


def test_attn(device_id):
    torch.set_printoptions(profile="full")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    contextual = ContextualAttention(stride=1, device=device, fuse=True)
    f = torch.rand(size=(10, 64, 64, 64)).to(device)
    mask = torch.ones(size=(10, 1, 256, 256)).to(device)
    attn = contextual(f, f, mask)
    assert attn[0].shape == (10, 64, 64, 64)
    assert attn[1].shape == (10, 3, 256, 256)
    return True
