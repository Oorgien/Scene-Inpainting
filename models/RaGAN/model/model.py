import torch
import torch.nn as nn
from . import blocks as B


class InpaintingGenerator(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nf=64, n_blocks=8, norm='in', activation='relu', init_weights=True):
        super(InpaintingGenerator, self).__init__()
        self.encoder = nn.Sequential(  # input: [4, 256, 256]
            B.conv_block(in_nc, nf, 5, stride=1, padding=2, norm='none', activation=activation),  # [64, 256, 256]
            B.conv_block(nf, 2 * nf, 3, stride=2, padding=1, norm=norm, activation=activation),  # [128, 128, 128]
            B.conv_block(2 * nf, 2 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [128, 128, 128]
            B.conv_block(2 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation)  # [256, 64, 64]
        )

        blocks = []
        for _ in range(n_blocks):
            block = B.DMFB(4 * nf)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            B.conv_block(4 * nf, 4 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [256, 64, 64]
            B.upconv_block(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),  # [128, 128, 128]
            B.upconv_block(2 * nf, nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),  # [64, 256, 256]
            B.conv_block(nf, out_nc, 3, stride=1, padding=1, norm='none', activation='tanh')  # [3, 256, 256]
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x
    

class InpaintingDiscriminator(nn.Module):
    def __init__(self, in_nc, nf, norm='bn', activation='lrelu', init_weights=True):
        super(InpaintingDiscriminator, self).__init__()
        self.global_branch = nn.ModuleList([
            # input: [3, 256, 256]
            B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation),  # [64, 128, 128]
            B.conv_block(nf, nf * 2, 5, stride=2, padding=2, norm=norm, activation=activation),  # [128, 64, 64]
            B.conv_block(nf * 2, nf * 4, 5, stride=2, padding=2, norm=norm, activation=activation),  # [256, 32, 32]
            B.conv_block(nf * 4, nf * 8, 5, stride=2, padding=2, norm=norm, activation=activation),  # [512, 16, 16]
            B.conv_block(nf * 8, nf * 8, 5, stride=2, padding=2, norm=norm, activation=activation),  # [512, 8, 8]
            B.conv_block(nf * 8, nf * 8, 5, stride=2, padding=2, norm=norm, activation=activation),  # [512, 4, 4]
            nn.Flatten(),
            nn.Linear(nf * 8 * 4 * 4, 512)  # [512]
        ])

        self.local_branch = nn.ModuleList([
            # input: [3, 256, 256]
            B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation),  # [64, 128, 128]
            B.conv_block(nf, nf * 2, 5, stride=2, padding=2, norm=norm, activation=activation),  # [128, 64, 64]
            B.conv_block(nf * 2, nf * 4, 5, stride=2, padding=2, norm=norm, activation=activation),  # [256, 32, 32]
            B.conv_block(nf * 4, nf * 8, 5, stride=2, padding=2, norm=norm, activation=activation),  # [512, 16, 16]
            B.conv_block(nf * 8, nf * 8, 5, stride=2, padding=2, norm=norm, activation=activation),  # [512, 8, 8]
            nn.Flatten(),
            nn.Linear(nf * 8 * 8 * 8, 512)  # [512]
        ])

        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, x_local):
        global_branch_features = []
        global_branch_features.append(x)
        for layer in self.global_branch:
            global_branch_features.append(layer(global_branch_features[-1]))

        local_branch_features = []
        local_branch_features.append(x_local)
        for layer in self.local_branch:
            local_branch_features.append(layer(local_branch_features[-1]))

        concat = torch.cat((local_branch_features[-1], global_branch_features[-1]), dim=-1)
        out = self.out(concat)

        return global_branch_features[1:], local_branch_features[1:], out   # also contains flatten layer