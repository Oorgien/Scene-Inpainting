import os

import torch
import torch.nn as nn

from base_model.base_blocks import DMFB, SelfAttention, _activation, get_pad

from . import blocks as B


class FineBottleneck(nn.Module):
    def __init__(self, nf, res_blocks_num=8, dmfb_blocks_num=4, attention_mode="self"):
        """
        :param nf: number of channels
        :param res_blocks_num: number of Residual blocks
        :param dmfb_blocks_num: number of DMFB blocks
        :param attention_mode: Attention mode ("context" or "self")
        """

        super(FineBottleneck, self).__init__()
        self.attention = SelfAttention(nf, k=8)
        res_seq = []
        for _ in range(res_blocks_num):
            block = B.ResConv(nf, kernel_size=3, dilation=2,
                              padding=2)  # [192, 64, 64]
            res_seq.append(block)
        self.res_seq = nn.Sequential(*res_seq)

        dmfb_seq = []
        for _ in range(dmfb_blocks_num):
            block = DMFB(nf)  # [192, 64, 64]
            dmfb_seq.append(block)
        self.dmfb_seq = nn.Sequential(*dmfb_seq)

        self.out = nn.Sequential(
            B.conv_block(3 * nf, nf, kernel_size=1, stride=1, padding=0,
                         norm="in", activation="elu", pad_type="zero"),
            B.conv_block(nf, nf, kernel_size=3, stride=1, padding=1,
                         norm="in", activation="elu", pad_type="zero")
        )

    def forward(self, x):
        res_seq = self.res_seq(x)
        dmfb_seq = self.dmfb_seq(x)
        attn = self.attention(x)
        out = self.out(torch.cat([res_seq, dmfb_seq, attn], dim=1))
        return out


class InpaintingGenerator(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nf=48,
                 res_blocks_num=8,
                 dmfb_block_num=4):
        """
        :param in_nc: in channels number
        :param out_nc: out channels number
        :param nf: number of intermediete features
        :param res_blocks_num: number of Residual blocks in fine and coarse roots
        :param dmfb_block_num: nunumber of DMFB blocks in fine route
        :param init_weights: what weight initialization to use
        """

        super(InpaintingGenerator, self).__init__()
        self.buffer = dict()
        # [4, 256, 256]
        self.encoder = nn.ModuleList([
            # [48, 256, 256] -> decoder_6
            B.RConv("encoder_1", self.buffer, in_nc, nf, kernel_size=5, stride=1, padding=2),
            # [96, 128, 128]
            B.RConv("encoder_2", self.buffer, nf, nf * 2, kernel_size=3, stride=2, padding=1),
            # [96, 128, 128] -> decoder_4
            B.RConv("encoder_3", self.buffer, nf * 2, nf * 2, kernel_size=3, stride=1, padding=1),
            # [192, 64, 64]
            B.RConv("encoder_4", self.buffer, nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
            # [192, 64, 64]
            B.RConv("encoder_5", self.buffer, nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
            # [192, 64, 64]
            B.RConv("encoder_6", self.buffer, nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        ])

        blocks = []
        for _ in range(res_blocks_num):
            block = B.ResConv(4 * nf, kernel_size=3,
                              dilation=2, padding=2)  # [192, 64, 64]
            blocks.append(block)

        self.coarse = nn.Sequential(*blocks)
        self.fine = FineBottleneck(
            4 * nf, res_blocks_num, dmfb_block_num, attention_mode="self")

        self.decoder = nn.ModuleList([
            # [192, 64, 64]
            B.RConv("decoder_1", self.buffer, nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
            # [192, 64, 64]
            B.RConv("decoder_2", self.buffer, nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
            # [96, 128, 128]
            B.RDeConv("decoder_3", self.buffer, nf * 4, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            # [96, 128, 128]
            B.RConv("decoder_4", self.buffer, nf * 4, nf * 2, kernel_size=3, stride=1, padding=1),
            # [48, 256, 256]
            B.RDeConv("decoder_5", self.buffer, nf * 2, nf, kernel_size=3, stride=2, padding=1, output_padding=1),
            # [48, 256, 256]
            B.RConv("decoder_6", self.buffer, nf * 2, nf, kernel_size=3, stride=1, padding=1),
            # [24, 256, 256]
            B.RConv("decoder_7", self.buffer, nf, nf // 2, kernel_size=3, stride=1, padding=1)
        ])

        self.out = nn.Sequential(
            B.conv_block(nf // 2, out_nc, 3, stride=1, padding=1,
                         norm='none', activation='tanh')
        )

    def forward(self, x, mode):
        encoder_states = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x, mode)
            if i == 0 or i == 2:
                encoder_states[f"encoder_{i+1}"] = x

        if mode == "coarse":
            x = self.coarse(x)
        elif mode == "fine":
            x = self.fine(x)

        for i, layer in enumerate(self.decoder):
            if (i == 5 or i == 3) and 6 - i >= 0:
                x = torch.cat([encoder_states[f"encoder_{6-i}"], x], dim=1)
            x = layer(x, mode)
        x = self.out(x)
        return x


class InpaintingDiscriminator(nn.Module):
    def __init__(self, device, in_nc=3, kernel_size=4, nf=48):
        """
        :param in_nc: number of in channels
        :param kernel_size: kernel size
        :param nf: number of convolution filters after 1 layer
        """
        super(InpaintingDiscriminator, self).__init__()

        self.patch_dis = nn.ModuleList([
            B.SNBlock(in_channels=in_nc, out_channels=nf, kernel_size=kernel_size, stride=2,
                      padding=get_pad(256, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf, out_channels=nf * 2, kernel_size=kernel_size, stride=2,
                      padding=get_pad(128, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 2, out_channels=nf * 2, kernel_size=kernel_size, stride=2,
                      padding=get_pad(64, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 2, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(32, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 4, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(16, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 4, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(8, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            nn.Flatten(),
            nn.Linear(nf * 4 * 4 * 4, 512)
        ])
        self.flat = nn.Flatten()

        self.edge_dis = nn.Sequential(
            B.SobelFilter(device, in_nc=3, filter_c=1,
                          padding=get_pad(256, 3, 1), stride=1),
            B.SNBlock(in_channels=2, out_channels=nf // 2, kernel_size=kernel_size, stride=4,
                      padding=get_pad(256, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf // 2, out_channels=nf, kernel_size=kernel_size, stride=2,
                      padding=get_pad(64, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf, out_channels=nf * 2, kernel_size=kernel_size, stride=2,
                      padding=get_pad(32, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 2, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(16, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 4, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(8, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            B.SNBlock(in_channels=nf * 4, out_channels=nf * 4, kernel_size=kernel_size, stride=2,
                      padding=get_pad(4, kernel_size, 2), norm='in', activation='relu', pad_type='zero'),
            nn.Flatten(),
            nn.Linear(nf * 4 * 2 * 2, 512)
        )

        self.out = nn.Sequential(
            _activation('relu'),
            nn.Linear(1024, 1)
        )

    def forward(self, x, x_local):
        patch_features = []
        patch_features.append(x_local)
        for layer in self.patch_dis:
            patch_features.append(layer(patch_features[-1]))
        patch_out = patch_features[-1]
        edge_out = self.edge_dis(x)
        out = self.out(torch.cat([patch_out, edge_out], dim=-1))
        return out, patch_features[1:7]
