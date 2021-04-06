import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torchsummary import summary

from .partialconv2d import PartialConv2d


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.activation = nn.ReLU()
        self.pconv_1 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                                     padding=3, return_mask=True, multi_channel=True)

        self.pconv_2 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,
                                     padding=2, return_mask=True, multi_channel=True)
        self.bn_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pconv_3 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,
                                     padding=2, return_mask=True, multi_channel=True)
        self.bn_3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pconv_4 = PartialConv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2,
                                     padding=2, return_mask=True, multi_channel=True)
        self.bn_4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pconv_5 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                     padding=1, return_mask=True, multi_channel=True)
        self.bn_5 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pconv_6 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                     padding=1, return_mask=True, multi_channel=True)
        self.bn_6 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pconv_7 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                     padding=1, return_mask=True, multi_channel=True)
        self.bn_7 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pconv_8 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                     padding=3, return_mask=True, multi_channel=True)
        self.bn_8 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, image, mask):
        im1, m1 = self.pconv_1(image, mask)
        im1 = self.activation(im1)

        im2, m2 = self.pconv_2(im1, m1)
        im2 = self.bn_2(im2)

        im3, m3 = self.pconv_3(im2, m2)
        im3 = self.bn_3(im3)

        im4, m4 = self.pconv_4(im3, m3)
        im4 = self.bn_4(im4)

        im5, m5 = self.pconv_5(im4, m4)
        im5 = self.bn_5(im5)

        im6, m6 = self.pconv_6(im5, m5)
        im6 = self.bn_6(im6)

        im7, m7 = self.pconv_7(im6, m6)
        im7 = self.bn_7(im7)

        im8, m8 = self.pconv_7(im7, m7)
        im8 = self.bn_8(im8)

        res = {
            'image': [image, im1, im2, im3, im4, im5, im6, im7, im8],
            'mask': [mask, m1, m2, m3, m4, m5, m6, m7, m8]
        }
        return res


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.pconv_1 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_1 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.pconv_2 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.pconv_3 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_3 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.pconv_4 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.pconv_5 = PartialConv2d(in_channels=512 + 256, out_channels=256, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_5 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.pconv_6 = PartialConv2d(in_channels=256 + 128, out_channels=128, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_6 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.pconv_7 = PartialConv2d(in_channels=128 + 64, out_channels=64, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)
        self.bn_7 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.pconv_8 = PartialConv2d(in_channels=64 + 3, out_channels=3, kernel_size=3,
                                     padding=1, stride=1, return_mask=True, multi_channel=True)

    def forward(self, encoder_res):
        enc_images = encoder_res['image']
        enc_masks = encoder_res['mask']

        im1_c, m1_c = self.up(enc_images[8]), self.up(enc_masks[8])
        im1_c = torch.cat((enc_images[7], im1_c), dim=1)
        m1_c = torch.cat((enc_masks[7], m1_c), dim=1)
        im1, m1 = self.pconv_1(im1_c, m1_c)
        im1 = self.bn_1(im1)

        im2_c, m2_c = self.up(im1), self.up(m1)
        im2_c = torch.cat((enc_images[6], im2_c), dim=1)
        m2_c = torch.cat((enc_masks[6], m2_c), dim=1)
        im2, m2 = self.pconv_2(im2_c, m2_c)
        im2 = self.bn_2(im2)

        im3_c, m3_c = self.up(im2), self.up(m2)
        im3_c = torch.cat((enc_images[5], im3_c), dim=1)
        m3_c = torch.cat((enc_masks[5], m3_c), dim=1)
        im3, m3 = self.pconv_3(im3_c, m3_c)
        im3 = self.bn_3(im3)

        im4_c, m4_c = self.up(im3), self.up(m3)
        im4_c = torch.cat((enc_images[4], im4_c), dim=1)
        m4_c = torch.cat((enc_masks[4], m4_c), dim=1)
        im4, m4 = self.pconv_4(im4_c, m4_c)
        im4 = self.bn_4(im4)

        im5_c, m5_c = self.up(im4), self.up(m4)
        im5_c = torch.cat((enc_images[3], im5_c), dim=1)
        m5_c = torch.cat((enc_masks[3], m5_c), dim=1)
        im5, m5 = self.pconv_5(im5_c, m5_c)
        im5 = self.bn_5(im5)

        im6_c, m6_c = self.up(im5), self.up(m5)
        im6_c = torch.cat((enc_images[2], im6_c), dim=1)
        m6_c = torch.cat((enc_masks[2], m6_c), dim=1)
        im6, m6 = self.pconv_6(im6_c, m6_c)
        im6 = self.bn_6(im6)

        im7_c, m7_c = self.up(im6), self.up(m6)
        im7_c = torch.cat((enc_images[1], im7_c), dim=1)
        m7_c = torch.cat((enc_masks[1], m7_c), dim=1)
        im7, m7 = self.pconv_7(im7_c, m7_c)
        im7 = self.bn_7(im7)

        im8_c, m8_c = self.up(im7), self.up(m7)
        im8_c = torch.cat((enc_images[0], im8_c), dim=1)
        m8_c = torch.cat((enc_masks[0], m8_c), dim=1)
        res_img, res_mask = self.pconv_8(im8_c, m8_c)

        return res_img, res_mask


class PartialUnet(nn.Module):
    def __init__(self, freeze_epoch, init_weights=True, ):
        super(PartialUnet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        if init_weights:
            self._initialize_weights()

        self.freeze_epoch = freeze_epoch

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_bn(self, m):
        classname = m.__class__.__name__
        if (classname.find('BatchNorm')) != -1:
            m.eval()

    def forward(self, image, mask, epoch):
        if epoch >= self.freeze_epoch:
            self.encoder.apply(self.freeze_bn)
        bottleneck = self.encoder(image, mask)
        res_image, res_mask = self.decoder(bottleneck)
        return res_image, res_mask
