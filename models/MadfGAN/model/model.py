import os

import torch
import torch.nn as nn

from base_model.base_blocks import (ContextualAttention, conv_block,
                                    upconv_block)

from . import blocks as B


class CoarseEncoder(nn.Module):
    def __init__(self,
                 in_channels_m=1, in_channels_e=3, device=torch.device('cpu')):
        super(CoarseEncoder, self).__init__()

        self.madf_seq = nn.ModuleList([
            # im - [3, 256, 256]
            # mask - [1, 256, 256]
            B.SimpleMADF(in_channels_m=in_channels_m, out_channels_m=2, in_channels_e=in_channels_e,
                         out_channels_e=16, kernel_size=5, stride=2, padding=2),
            # im - [16, 128, 128]
            # mask - [2, 128, 128]
            B.SimpleMADF(in_channels_m=2, out_channels_m=4, in_channels_e=16,
                         out_channels_e=32, kernel_size=3, stride=2, padding=1),
            # im - [32, 64, 64]
            # mask - [4, 64, 64]
            B.SimpleMADF(in_channels_m=4, out_channels_m=8, in_channels_e=32,
                         out_channels_e=64, kernel_size=3, stride=2, padding=1),
            # im - [64, 32, 32]
            # mask - [8, 32, 32]
            B.SimpleMADF(in_channels_m=8, out_channels_m=16, in_channels_e=64,
                         out_channels_e=128, kernel_size=3, stride=2, padding=1),
            # im - [128, 16, 16]
            # mask - [16, 16, 16]
        ])

        self.up_seq = nn.ModuleList([
            conv_block(in_channels=3, out_channels=16, kernel_size=3,
                       stride=1, padding=1, activation='relu', norm='none'),
            conv_block(in_channels=16, out_channels=16, kernel_size=3,
                       stride=1, padding=1, activation='relu', norm='none'),
            conv_block(in_channels=32, out_channels=32, kernel_size=3,
                       stride=1, padding=1, activation='relu', norm='none'),
            conv_block(in_channels=64, out_channels=64, kernel_size=3,
                       stride=1, padding=1, activation='relu', norm='none'),
            conv_block(in_channels=128, out_channels=128, kernel_size=3,
                       stride=1, padding=1, activation='relu', norm='none')
        ])

    def forward(self, mask, image):
        """
        :param mask: input mask
        :param image: input image
        :return:
            array of m {l}, array of e {l}, array of u {l} for each l
        """
        masks = [mask]
        images = [image]
        for layer in self.madf_seq:
            mask, image = layer(mask, image)
            masks.append(mask)
            images.append(image)

        up_layers = []
        for e_l, up_layer in zip(images, self.up_seq):
            up_layers.append(up_layer(e_l))

        return masks, images, up_layers


class RefinementDecoder(nn.Module):
    def __init__(self):
        super(RefinementDecoder, self).__init__()

        self.refinement_seq = nn.ModuleList([
            B.RefinementBlock(in_channels_1=64, in_channels_2=128, out_channels=64,
                              kernel_size_1=3, kernel_size_2=3, stride_1=1, up_stride_2=2,
                              padding_1=1, up_padding_2=1, output_padding=1),
            B.RefinementBlock(in_channels_1=32, in_channels_2=64, out_channels=32,
                              kernel_size_1=3, kernel_size_2=3, stride_1=1, up_stride_2=2,
                              padding_1=1, up_padding_2=1, output_padding=1),
            B.RefinementBlock(in_channels_1=16, in_channels_2=32, out_channels=16,
                              kernel_size_1=3, kernel_size_2=3, stride_1=1, up_stride_2=2,
                              padding_1=1, up_padding_2=1, output_padding=1),
            B.RefinementBlock(in_channels_1=3, in_channels_2=16, out_channels=3,
                              kernel_size_1=3, kernel_size_2=3, stride_1=1, up_stride_2=2,
                              padding_1=1, up_padding_2=1, output_padding=1, activation_out='tanh')
        ])

    def forward(self, recovery_out: list, f_l_k):
        refinement_out = []
        for f_l_1_k_1, layer in zip(recovery_out, self.refinement_seq):
            f_l_k = layer(f_l_1_k_1, f_l_k)
            refinement_out.append(f_l_k)
        return refinement_out


class CoarseDecoder(nn.Module):
    def __init__(self):
        super(CoarseDecoder, self).__init__()

        self.recovery_seq = nn.ModuleList([
            B.RecovecyBlock(in_channels_r=128, out_channels_r=64, in_channels_u=64, out_channels=64,
                            kernel_size_in=3, kernel_size_out=3, up_stride_in=2, stride_out=1, up_padding_in=1,
                            padding_out=1, output_padding=1),
            B.RecovecyBlock(in_channels_r=64, out_channels_r=32, in_channels_u=32, out_channels=32,
                            kernel_size_in=3, kernel_size_out=3, up_stride_in=2, stride_out=1, up_padding_in=1,
                            padding_out=1, output_padding=1),
            B.RecovecyBlock(in_channels_r=32, out_channels_r=16, in_channels_u=16, out_channels=16,
                            kernel_size_in=3, kernel_size_out=3, up_stride_in=2, stride_out=1, up_padding_in=1,
                            padding_out=1, output_padding=1),
            B.RecovecyBlock(in_channels_r=16, out_channels_r=16, in_channels_u=16, out_channels=3,
                            kernel_size_in=3, kernel_size_out=3, up_stride_in=2, stride_out=1, up_padding_in=1,
                            padding_out=1, output_padding=1, activation_out='tanh'),
        ])

        self.refinement_seq = nn.ModuleList([
            RefinementDecoder(),
            RefinementDecoder(),
            RefinementDecoder(),
            RefinementDecoder()
        ])

    def forward(self, up_layers: list):
        r_l = up_layers[-1]
        recovery_out = []
        for u_l_1, layer in zip(up_layers[-2::-1], self.recovery_seq):
            r_l = layer(r_l, u_l_1)
            recovery_out.append(r_l)
        refined = recovery_out
        for layer in self.refinement_seq:
            refined = layer(refined, up_layers[-1])
        return refined


class CoarseGenerator(nn.Module):
    def __init__(self, img_in_c, mask_in_c, device):
        super(CoarseGenerator, self).__init__()

        self.coarse_encoder = CoarseEncoder(in_channels_m=mask_in_c, in_channels_e=img_in_c, device=device)
        self.coarse_decoder = CoarseDecoder()

    def forward(self, image, mask):
        masks, images, up_layers = self.coarse_encoder(mask, image)
        coarse = self.coarse_decoder.forward(up_layers)
        return coarse


class FineGenerator(nn.Module):
    def __init__(self, in_nc=4, c_num=32, device=None):
        super(FineGenerator, self).__init__()

        self.conv_seq = nn.Sequential(
            # [4, 256, 256]
            conv_block(in_channels=in_nc, out_channels=c_num, kernel_size=5, stride=1, padding=2),
            # [cnum, 256, 256]
            conv_block(in_channels=c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1),
            # [cnum * 2, 128, 128]
            conv_block(in_channels=2 * c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1),
            # [cnum * 2, 64, 64]
            conv_block(in_channels=2 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]

            # dilation
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=2, dilation=2),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=4, dilation=4),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=8, dilation=8),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=16, dilation=16)
            # [cnum * 4, 64, 64]
        )

        self.before_attn = nn.Sequential(
            # [4, 256, 256]
            conv_block(in_channels=in_nc, out_channels=c_num, kernel_size=5, stride=1, padding=2),
            # [cnum, 256, 256]
            conv_block(in_channels=c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1),
            # [cnum * 2, 128, 128]
            conv_block(in_channels=2 * c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1),
            # [cnum * 2, 64, 64]
            conv_block(in_channels=2 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
        )

        # Contextual attention
        self.contextual_attention = ContextualAttention(ksize=3, stride=1, padding=1, softmax_scale=10, device=device)
        # [cnum * 4, 64, 64]

        self.after_attn = nn.Sequential(
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
        )

        self.all_conv = nn.Sequential(
            # concatenated input [cnum * 8, 64, 64]
            conv_block(in_channels=8 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
            conv_block(in_channels=4 * c_num, out_channels=4 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 4, 64, 64]
            upconv_block(in_channels=4 * c_num, out_channels=2 * c_num, kernel_size=3, stride=2, padding=1,
                         norm='in', output_padding=1),
            # [cnum * 2, 128, 128]
            conv_block(in_channels=2 * c_num, out_channels=2 * c_num, kernel_size=3, stride=1, padding=1),
            # [cnum * 2, 128, 128]
            upconv_block(in_channels=2 * c_num, out_channels=c_num, kernel_size=3, stride=2, padding=1,
                         norm='in', output_padding=1),
            # [cnum, 256, 256]
            conv_block(in_channels=c_num, out_channels=c_num // 2, kernel_size=3, stride=1, padding=1),
            # [cnum * 8, 256, 256]
            conv_block(in_channels=c_num // 2, out_channels=3, kernel_size=3, stride=1, padding=1, activation='tanh'),
        )

    def forward(self, x, mask):
        conv_branch = self.conv_seq(x)
        attn_input = self.before_attn(x)
        attn_output, offset = self.contextual_attention(attn_input, attn_input, mask)
        attn_branch = self.after_attn(attn_output)

        x = torch.cat((conv_branch, attn_branch), dim=1)
        x = self.all_conv(x)
        return x, offset


class InpaintingGenerator(nn.Module):
    def __init__(self, image_inc_coarse=3, mask_inc_coarse=1, image_inc_fine=4, nc_fine=32, device=None):
        """
        :param image_inc_coarse: number of channels of input image in coarse generator.
        :param mask_inc_coarse: number of channels of input mask in coarse generator.
        :param image_inc_fine: number of channels of input image in fine generator.
        :param nc_fine: number of channels to propogate in fine generator.
        :param device: gpu: {gpu_id} or cpu
        """
        super(InpaintingGenerator, self).__init__()

        self.coarse_generator = CoarseGenerator(image_inc_coarse, mask_inc_coarse, device)
        self.fine_generator = FineGenerator(image_inc_fine, nc_fine, device)

    def forward_coarse(self, x, mask):
        coarse_output = self.coarse_generator(x, mask)
        return coarse_output

    def forward_fine(self, x, mask):
        fine_output, offset = self.fine_generator(x, mask)
        return fine_output, offset


def test_model(device_id):
    print("MADF Encoder test...")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    enc = CoarseEncoder(in_channels_e=3, in_channels_m=1, device=device).to(device)
    img = torch.rand(size=(10, 3, 256, 256)).requires_grad_().to(device)
    mask = torch.rand(size=(10, 1, 256, 256)).to(device)
    masks, images, up_layers = enc.forward(mask, img)
    print("MADF Encoder test --- OK")
    print("MADF Decoder test...")
    rec_dec = CoarseDecoder().to(device)
    rec_out = rec_dec.forward(up_layers)
    assert (rec_out[0].shape == (10, 64, 32, 32))
    assert (rec_out[1].shape == (10, 32, 64, 64))
    assert (rec_out[2].shape == (10, 16, 128, 128))
    assert (rec_out[3].shape == (10, 3, 256, 256))
    print("MADF Decoder test --- OK")
    print("Contextual attention fine generator testing...")
    fine_dec = FineGenerator(device=device).to(device)
    res = fine_dec(torch.cat((rec_out[3], mask), dim=1), mask)
    s = torch.sum(res[0])
    s.backward()
    assert res[0].shape == (10, 3, 256, 256)
    assert res[1].shape == (10, 1, 64, 64)
    print("Contextual attention fine generator test -- OK")
    return True
