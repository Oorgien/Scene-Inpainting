import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo


def make_vgg16_layers(style_avg_pool = False):
    """
    make_vgg16_layers
    Return a custom vgg16 feature module with avg pooling
    """
    vgg16_cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        512, 512, 512, 'M', 512, 512, 512, 'M'
    ]

    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            if style_avg_pool:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Features(nn.Module):
    """
    VGG16 partial model
    """
    def __init__(self, vgg_path, layer_num=3):
        """
        Init
        :param layer_num: number of layers
        """
        super().__init__()
        vgg_model = models.vgg16()
        vgg_model.features = make_vgg16_layers()
        try:
            vgg_model.load_state_dict(
                torch.load(vgg_path, map_location='cpu')
            )
            print (f"Model loaded from {vgg_path}")
        except:
            vgg_model.load_state_dict(
                model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
            )
            print(f"Model loaded from 'https://download.pytorch.org/models/vgg16-397923af.pth' url")

        vgg_pretrained_features = vgg_model.features

        assert layer_num > 0
        assert isinstance(layer_num, int)
        self.layer_num = layer_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # 4
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # (5, 9)
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):  # (10, 16)
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):  # (17, 23)
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 4:
            self.slice5 = torch.nn.Sequential()
            for x in range(24, 31):  # (24, 30)
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        """
        Forward, get features used for perceptual loss
        :param x: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: list of self.layer_num feature maps used to compute the
         perceptual loss
        """
        h = self.slice1(x)
        h1 = h

        output = []

        if self.layer_num == 1:
            output = [h1]
        elif self.layer_num == 2:
            h = self.slice2(h)
            h2 = h
            output = [h1, h2]
        elif self.layer_num == 3:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            output = [h1, h2, h3]
        elif self.layer_num == 4:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            output = [h1, h2, h3, h4]
        elif self.layer_num >= 5:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            h = self.slice5(h)
            h5 = h
            output = [h1, h2, h3, h4, h5]
        return output


class SelfGuidedLoss(nn.Module):
    """
    Create Self-Guided Loss
    """
    def __init__(self, vgg_path, layer_num=2):
        super(SelfGuidedLoss, self).__init__()
        self.vgg_features = VGG16Features(vgg_path=vgg_path, layer_num=layer_num).eval()
        self.layer_num = layer_num
        self.AP = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, predicted, target):
        assert len(predicted.shape) == 4
        assert len(target.shape) == 4

        M_error = torch.mean(F.mse_loss(predicted, target, reduction='none'), dim=1)
        M_error = M_error.unsqueeze(1)
        M_max = M_error.max(2, True)[0].max(3, True)[0]
        M_min = M_error.min(2, True)[0].min(3, True)[0]
        M_guided = (M_error - M_min) / (M_max - M_min + 0.00001)
        assert M_error.shape == (predicted.shape[0], 1, predicted.shape[2], predicted.shape[3])

        # vgg features
        with torch.no_grad():
            vgg_gt = self.vgg_features(target)
        vgg_pred = self.vgg_features(predicted)

        Loss = 0
        for i in range(0, self.layer_num):
            num_elements = vgg_gt[i].shape[1] * vgg_gt[i].shape[2] * vgg_gt[i].shape[3]
            w = 1e3 / (vgg_gt[i].shape[1]) ** 2
            M_guided = self.AP(M_guided)
            Loss += (w/num_elements) * F.l1_loss(M_guided * vgg_pred[i], M_guided * vgg_gt[i], reduction='sum')

        return Loss


class GeometricalAlignLoss(nn.Module):
    def __init__(self, device, vgg_path, layer_num=4):
        super(GeometricalAlignLoss, self).__init__()
        self.vgg_features = VGG16Features(vgg_path=vgg_path, layer_num=layer_num).eval()
        self.device = device

    def forward(self, predicted, target):
        assert len(predicted.shape) == 4
        assert len(target.shape) == 4

        # vgg features
        with torch.no_grad():
            vgg_gt = self.vgg_features(target)
        vgg_pred = self.vgg_features(predicted)

        vgg_last_gt = vgg_gt[-1]
        vgg_last_pred = vgg_pred[-1]

        # compute center gt
        X_gt, Y_gt = self.compute_centers(vgg_last_gt)

        # compute center pred
        X_pred, Y_pred = self.compute_centers(vgg_last_pred)

        # b = torch.sub(Y_pred, Y_gt).unsqueeze(2)
        # a = torch.sub(X_pred, X_gt).unsqueeze(2)
        #
        # vectors = torch.cat([a, b], dim=2)
        # Loss = torch.sum(((vectors) ** 2).mean(2), dim=1)
        Loss = F.mse_loss(torch.stack([X_gt, Y_gt], -1), torch.stack([X_pred, Y_pred], -1), reduction='mean')
        return Loss


    def compute_centers(self, vgg_features):
        X, Y = torch.meshgrid(torch.linspace(-1, 1, vgg_features.shape[2]),
                              torch.linspace(-1, 1, vgg_features.shape[3]))
        X = X.unsqueeze(0).unsqueeze(1).to(self.device)
        Y = Y.unsqueeze(0).unsqueeze(1).to(self.device)

        C_x_gt = torch.div(vgg_features, torch.sum(vgg_features, dim=(2, 3)).unsqueeze(2).unsqueeze(3) + 0.0001)
        X_centers = torch.sum(X * C_x_gt, dim=(2, 3))

        C_y_gt = torch.div(vgg_features, torch.sum(vgg_features, dim=(2, 3)).unsqueeze(2).unsqueeze(3) + 0.0001)
        Y_centers = torch.sum(Y * C_y_gt, dim=(2, 3))
        return X_centers, Y_centers


    def compute_center_Y(self, vgg_features):
        Y = torch.arange(1, vgg_features.shape[3] + 1).view(1, 1, -1, 1).to(self.device)
        Y = torch.repeat_interleave(Y, vgg_features.shape[2], dim=2)
        Y = Y.view(Y.shape[0], Y.shape[1], vgg_features.shape[2], vgg_features.shape[3])
        C_y_gt = torch.div(vgg_features, torch.sum(vgg_features, dim=(2, 3)).unsqueeze(2).unsqueeze(3) + 0.0001)
        Y_centers = torch.sum(Y * C_y_gt, dim=(2, 3))
        return Y_centers


class FeatureMatchingLoss(nn.Module):
    def __init__(self, vgg_path, layer_num=5, dis_num=5):
        super(FeatureMatchingLoss, self).__init__()
        self.vgg_features = VGG16Features(vgg_path=vgg_path, layer_num=layer_num).eval()
        self.layer_num = layer_num
        self.dis_num = dis_num

    def forward(self, predicted, target, local_gt, local_pred):
        assert len(predicted.shape) == 4
        assert len(target.shape) == 4

        vgg_pred = self.vgg_features(predicted)
        with torch.no_grad():
            vgg_gt = self.vgg_features(target)


        Loss_vgg = 0
        for i in range(self.layer_num):
            num_elements = vgg_gt[i].shape[1] * vgg_gt[i].shape[2] * vgg_gt[i].shape[3]
            w = 1e3 / (vgg_gt[i].shape[1] ** 2)
            # print('vgg' , (w/num_elements) * F.l1_loss(local_gt[i], local_pred[i], reduction='sum'))
            Loss_vgg += (w/num_elements) * F.l1_loss(vgg_gt[i], vgg_pred[i], reduction='sum')

        Loss_dis = 0
        for i in range(min(self.dis_num, 5)):
            num_elements = local_gt[i].shape[1] * local_gt[i].shape[2] * local_gt[i].shape[3]
            w = 1e3 / (local_gt[i].shape[1] ** 2)
            Loss_dis += (w/num_elements) * F.l1_loss(local_gt[i], local_pred[i], reduction='sum')
        return Loss_vgg, Loss_dis


class AdversarialLoss(nn.Module):
    def __init__(self, mode, model, device):
        super(AdversarialLoss, self).__init__()
        self.model = model
        self.mode = mode
        self.device = device
        self.BCEloss = nn.BCEWithLogitsLoss().to(device)

    def forward(self, real_dis, fake_dis):
        mean_fake = torch.mean(fake_dis, dim=0, keepdim=True)
        mean_real = torch.mean(real_dis, dim=0, keepdim=True)

        D_real = real_dis - mean_fake
        D_fake = fake_dis - mean_real

        # -> max for discriminator
        # -> min for generator

        zeros = torch.zeros(real_dis.shape[0], 1).to(self.device)
        ones = torch.ones(real_dis.shape[0], 1).to(self.device)

        if self.model == 'discriminator':
            if self.mode == "bce":
                L_adv = self.BCEloss(D_real, ones) + self.BCEloss(D_fake, zeros)
            elif self.mode == "l1":
                L_adv = torch.mean(torch.abs(D_real - ones) + torch.abs(D_fake))

        elif self.model == 'generator':
            if self.mode == "bce":
                L_adv = self.BCEloss(D_fake, ones) + self.BCEloss(D_real, zeros)
            elif self.mode == "l1":
                L_adv = torch.mean(torch.abs(D_fake - ones) + torch.abs(D_real))
        else:
            raise NotImplementedError(f'{self.mode} part of model is not defined.')

        return L_adv


class RaGANLoss(nn.Module):
    def __init__(self, device,  vgg_path='downloaded_models/vgg16-397923af.pth', normalize=False):
        super(RaGANLoss, self).__init__()
        self.self_guided = SelfGuidedLoss(vgg_path=vgg_path)
        self.align_loss = GeometricalAlignLoss(device, vgg_path=vgg_path)
        self.normalize = normalize
        self.device = device
        self.l1loss = torch.nn.L1Loss(reduction='mean').to(device)

    @staticmethod
    def normalize_batch(batch, div_factor=255.):
        """
        Normalize batch

        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= mean
        batch = torch.div(batch, std)
        return batch

    def forward(self, x, y):
        """
        Forward

        :param: x - predicted, y - target
        :return: total loss without perceptual, discriminator
        feature matching and adversarial
        """

        if self.normalize:
            x = self.normalize_batch(x)
            y = self.normalize_batch(y)

        # L1 loss
        l1_loss = self.l1loss(x,y)

        # Self-guided loss
        self_guided_loss = self.self_guided(x, y)

        # Align loss
        align_loss = self.align_loss(x, y)

        return l1_loss, self_guided_loss, align_loss