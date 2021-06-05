import torch
import torch.nn as nn
from torchsummary import summary

from base_model.base_blocks import (DMFB, ContextualAttention, HypergraphConv,
                                    SelfAttention, _activation, conv_block,
                                    get_pad, upconv_block)
from utils import get_pad_tp

from . import blocks as B


class CoarseGenerator(nn.Module):
    def __init__(self,
                 in_nc=4, c_num=48,
                 out_nc=3, res_blocks_num=8,
                 norm="in",
                 activation="relu"):
        super(CoarseGenerator, self).__init__()

        self.encoder_coarse = nn.ModuleList([
            # [4, 256, 256]
            conv_block(in_nc, c_num, kernel_size=5, stride=1, padding=2,
                       norm=norm, activation=activation),
            # [c_num, 256, 256] -> decoder_6
            conv_block(c_num, c_num * 2, kernel_size=3, stride=2, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128]
            conv_block(c_num * 2, c_num * 2, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128] -> decoder_4
            conv_block(c_num * 2, c_num * 4, kernel_size=3, stride=2, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            conv_block(c_num * 4, c_num * 4, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            conv_block(c_num * 4, c_num * 4, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation)
            # [c_num * 4, 64, 64]]
        ])

        blocks = []
        for _ in range(res_blocks_num):
            block = B.ResConv(4 * c_num, kernel_size=3,
                              dilation=2, padding=2)  # [192, 64, 64]
            blocks.append(block)

        self.coarse_bn = nn.Sequential(*blocks)

        self.decoder_coarse = nn.ModuleList([
            conv_block(c_num * 4, c_num * 4, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            conv_block(c_num * 4, c_num * 4, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            upconv_block(c_num * 4, c_num * 2, kernel_size=3, stride=2, output_padding=1,
                         padding=get_pad_tp(64, 64, [1, 1], [3, 3], [2, 2], [1, 1]),
                         norm=norm, activation=activation),
            # [c_num * 2, 128, 128] + skip [c_num * 2, 128, 128]
            conv_block(c_num * 4, c_num * 2, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128]
            upconv_block(c_num * 2, c_num, kernel_size=3, stride=2, output_padding=1,
                         padding=get_pad_tp(128, 128, [1, 1], [3, 3], [2, 2], [1, 1]),
                         norm=norm, activation=activation),
            # [c_num, 256, 256] + skip [c_num, 256, 256]
            conv_block(c_num * 2, c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num, 256, 256]
            conv_block(c_num, c_num // 2, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation)
            # [c_num // 2, 256, 256]
        ])

        self.out_coarse = nn.Sequential(
            conv_block(c_num // 2, out_nc, 3, stride=1, padding=1,
                       norm='none', activation='tanh')
        )

    def forward(self, x):
        # Coarse root
        encoder_states = {}
        for i, layer in enumerate(self.encoder_coarse):
            x = layer(x)
            if i == 0 or i == 2:
                encoder_states[f"encoder_{i + 1}"] = x

        x = self.coarse_bn(x)

        for i, layer in enumerate(self.decoder_coarse):
            if (i == 5 or i == 3) and 6 - i >= 0:
                x = torch.cat([encoder_states[f"encoder_{6 - i}"], x], dim=1)
            x = layer(x)
        x = self.out_coarse(x)
        return x


class FineBottleneck(nn.Module):
    def __init__(self, c_num, res_blocks_num=8, dmfb_blocks_num=8,
                 activateion="relu", norm="in", device=None):
        super(FineBottleneck, self).__init__()

        dmfb_blocks = []
        for i in range(dmfb_blocks_num):
            # [c_num, 64, 64]
            dmfb_blocks.append(DMFB(in_channels=c_num))

        self.dmfb_seq = nn.Sequential(*dmfb_blocks)

        res_blocks = []
        for i in range(res_blocks_num):
            # [c_num, 64, 64]
            block = B.ResConv(
                channels=c_num, kernel_size=3,
                dilation=2, padding=2
            )
            res_blocks.append(block)

        self.res_seq = nn.Sequential(*res_blocks)

        # Contextual attention
        self.contextual_attention = ContextualAttention(
            ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
            fuse=True, device=device
        )

        self.hypergraph = HypergraphConv(
            in_channels=c_num,
            out_channels=c_num,
            filters=256, edges=256,
            height=64, width=64)

        self.out_1 = nn.Sequential(
            conv_block(in_channels=3 * c_num, out_channels=c_num, kernel_size=1, stride=1, padding=0,
                       norm=norm, activation=activateion, pad_type="zero"),
            conv_block(in_channels=c_num, out_channels=c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activateion, pad_type="zero")
        )

        self.out = nn.Sequential(
            conv_block(in_channels=2 * c_num, out_channels=c_num, kernel_size=1, stride=1, padding=0,
                       norm=norm, activation=activateion, pad_type="zero"),
            conv_block(in_channels=c_num, out_channels=c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activateion, pad_type="zero")
        )

    def forward(self, x, mask):
        res_seq = self.res_seq(x)
        dmfb_seq = self.dmfb_seq(x)
        attn, flow = self.contextual_attention(x, x, mask)
        hypergraph = self.hypergraph(x)

        out_1 = self.out_1(torch.cat([attn, dmfb_seq, res_seq], dim=1))
        out = self.out(torch.cat([out_1, hypergraph], dim=1))
        return out


class FineGenerator(nn.Module):
    def __init__(self,
                 in_nc=4, out_nc=3,
                 res_blocks_num=8,
                 c_num=48, dmfb_blocks_num=8,
                 norm='in', activation='relu',
                 device=None):
        super(FineGenerator, self).__init__()

        self.encoder = nn.ModuleList([
            # [4, 256, 256]
            conv_block(in_channels=in_nc, out_channels=c_num, kernel_size=5, stride=1, padding=2,
                       norm=norm, activation=activation),
            # [c_num, 256, 256] -> decoder
            conv_block(in_channels=c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128]
            conv_block(in_channels=2 * c_num, out_channels=2 * c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128] -> decoder
            conv_block(in_channels=2 * c_num, out_channels=4 * c_num, kernel_size=3, stride=2, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
        ])

        self.fine_bn = FineBottleneck(
            c_num=4 * c_num, dmfb_blocks_num=dmfb_blocks_num,
            res_blocks_num=res_blocks_num, device=device)

        self.decoder = nn.ModuleList([
            # [c_num * 4, 64, 64]
            conv_block(in_channels=c_num * 4, out_channels=c_num * 4, kernel_size=3, stride=1, padding=1),
            # [c_num * 4, 64, 64]
            conv_block(in_channels=c_num * 4, out_channels=c_num * 4, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 4, 64, 64]
            upconv_block(in_channels=c_num * 4, out_channels=c_num * 2, kernel_size=3, stride=2, output_padding=1,
                         padding=get_pad_tp(64, 64, [1, 1], [3, 3], [2, 2], [1, 1]),
                         norm=norm, activation=activation),
            # encoder 3 + skip [c_num * 2, 128, 128] -> decoder 4
            # [c_num * 4, 128, 128]
            conv_block(in_channels=c_num * 4, out_channels=c_num * 2, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num * 2, 128, 128]
            upconv_block(in_channels=c_num * 2, out_channels=c_num, kernel_size=3, stride=2, output_padding=1,
                         padding=get_pad_tp(128, 128, [1, 1], [3, 3], [2, 2], [1, 1]),
                         norm=norm, activation=activation),
            # encoder 1 + skip [c_num, 256, 256] -> decoder 6
            # [c_num, 256, 256]
            conv_block(in_channels=c_num * 2, out_channels=c_num // 2, kernel_size=3, stride=1, padding=1,
                       norm=norm, activation=activation),
            # [c_num//2, 256, 256]
            conv_block(in_channels=c_num // 2, out_channels=out_nc, kernel_size=3, stride=1, padding=1,
                       norm='none', activation='tanh'),
            # [out_nc, 256, 256]
        ])

    def forward(self, x, mask):
        encoder_states = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 0 or i == 2:
                encoder_states[f"encoder_{i + 1}"] = x

        x = self.fine_bn(x, mask)

        for i, layer in enumerate(self.decoder):
            if (i == 3 or i == 5) and 5 - i >= 0:
                x = torch.cat([encoder_states[f"encoder_{5 - i + 1}"], x], dim=1)
            x = layer(x)

        return x


class InpaintingGenerator(nn.Module):
    def __init__(self,
                 in_nc=4, c_num=48,
                 out_nc=3,
                 dmfb_blocks_num=8,
                 res_blocks_num=8,
                 norm='in', activation='relu',
                 device=None):
        super(InpaintingGenerator, self).__init__()

        self.coarse_generator = CoarseGenerator(
            in_nc=in_nc,
            c_num=c_num,
            out_nc=out_nc,
            res_blocks_num=res_blocks_num,
            norm=norm,
            activation=activation)

        self.fine_generator = FineGenerator(
            in_nc=in_nc,
            c_num=c_num,
            out_nc=out_nc,
            res_blocks_num=res_blocks_num,
            dmfb_blocks_num=dmfb_blocks_num,
            norm=norm,
            activation=activation,
            device=device
        )
        self.device = device

    def forward(self, masked_image, mask_3x, mask):
        predicted_coarse = self.coarse_generator(torch.cat([masked_image, mask], dim=1))
        predicted_coarse = masked_image + torch.mul(predicted_coarse,
                                                    (torch.ones(mask_3x.shape).to(self.device) - mask_3x))

        predicted_fine = self.fine_generator(torch.cat((predicted_coarse, mask), dim=1), mask)
        predicted_fine = masked_image + torch.mul(predicted_fine, (torch.ones(mask_3x.shape).to(self.device) - mask_3x))
        return predicted_coarse, predicted_fine


def test_best_model(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    mask = torch.ones(size=(10, 1, 256, 256)).to(device)
    im = torch.randn(size=(10, 3, 256, 256)).to(device)

    coarse = CoarseGenerator()
    coarse_out = coarse(torch.cat([im, mask], dim=1))
    assert coarse_out.shape == (10, 3, 256, 256)

    fine = FineGenerator(device=device)
    fine_out = fine(torch.cat([coarse_out, mask], dim=1), mask)
    assert fine_out.shape == (10, 3, 256, 256)
    return True
