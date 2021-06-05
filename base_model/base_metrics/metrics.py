import numpy as np
import torch
from scipy import linalg
from skimage.transform import resize
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3

from utils import reduce_mean


def covariance(x, y):
    x_centered = x - reduce_mean(x, axis=(1, 2, 3), keepdim=True)
    y_centered = y - reduce_mean(y, axis=(1, 2, 3), keepdim=True)

    x_centered = torch.flatten(x_centered, start_dim=1)
    y_centered = torch.flatten(y_centered, start_dim=1)
    cov = torch.mean(x_centered.mm(torch.transpose(y_centered, -1, -2)), dim=1)
    return cov


def SSIM(x, y, L=2 ** 255 - 1, k1=0.01, k2=0.03):
    """
    :param x: size (batch_size, channels, height, width)
    :param y: size (batch_size, channels, height, width)
    :return: SSIM metrics
    :param L:  the dynamic range of the pixel-values 2^{bits-1)
    :param k1: k_1=0.01 by default
    :param k2: k_2=0.03 by default
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    E_x = np.mean(x)
    E_y = np.mean(y)

    sigma_x = np.std(x)
    sigma_y = np.std(y)
    cov = np.mean((x - E_x) * (y - E_y))

    ssim = (2 * E_x * E_y + c1) * (2 * cov + c2)\
        / ((E_x ** 2 + E_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

    return ssim


def PSNR(x, y):
    mse = F.mse_loss(x, y).detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    psnr = 20 * np.log10(np.max(x)) - 10 * np.log10(mse)
    return psnr


class FrechetDistance:
    def __init__(self, in_size=(3, 256, 256)):
        self.inception_network = inception_v3(pretrained=True)

        self.in_size = in_size

    def resize_images(self, images):
        """
        :param images: np.ndarray of size [b_size, N, h, w]
        :return: resized image
        """

        for image in images:
            image = image.resize(3, 299, 299)

        return images

    def get_activations(self, x):
        """
        :param x: Torch tensor of size [b_size, N, h, w]
        :return: activations from Inception V3 model
        """

        x_resized = x.detach().cpu().numpy()
        x_resized = self.resize_images(x_resized)
        x_resized = torch.FloatTensor(x_resized)

        # Trigger output hook
        self.inception_network(x_resized)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x_resized.shape[0], 2048)
        return activations

    def calculate_fid(self, x, y):
        """
            Frechet Inception Distance
            https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

            Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            Stable version by Dougal J. Sutherland.
            Params:
            -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                     inception net ( like returned by the function 'get_predictions')
                     for generated samples.
            -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                       on an representive data set.
            -- sigma1: The covariance matrix over activations of the pool_3 layer for
                       generated samples.
            -- sigma2: The covariance matrix over activations of the pool_3 layer,
                       precalcualted on an representive data set.
            Returns:
            --   : The Frechet Distance.

            """
        x_activations, y_activations = self.get_activations(x), self.get_activations(y)

        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()

        mu1, mu2 = np.mean(x_numpy), np.mean(y_numpy)
        cov1, cov2 = np.cov(x_numpy), np.cov(y_numpy)

        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)


def PSNR(x, y):
    mse = F.mse_loss(x, y).detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    psnr = 20 * np.log10(np.max(x)) - 10 * np.log10(mse)
    return psnr


class FrechetDistance:
    def __init__(self, in_size=(3, 256, 256)):
        self.inception_network = inception_v3(pretrained=True)

        self.in_size = in_size

    def resize_images(self, images):
        """
        :param images: np.ndarray of size [b_size, N, h, w]
        :return: resized image
        """

        for image in images:
            image = image.resize(3, 299, 299)

        return images

    def get_activations(self, x):
        """
        :param x: Torch tensor of size [b_size, N, h, w]
        :return: activations from Inception V3 model
        """

        x_resized = x.detach().cpu().numpy()
        x_resized = self.resize_images(x_resized)
        x_resized = torch.FloatTensor(x_resized)

        # Trigger output hook
        self.inception_network(x_resized)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x_resized.shape[0], 2048)
        return activations

    def calculate_fid(self, x, y):
        """
            Frechet Inception Distance
            https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

            Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            Stable version by Dougal J. Sutherland.
            Params:
            -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                     inception net ( like returned by the function 'get_predictions')
                     for generated samples.
            -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                       on an representive data set.
            -- sigma1: The covariance matrix over activations of the pool_3 layer for
                       generated samples.
            -- sigma2: The covariance matrix over activations of the pool_3 layer,
                       precalcualted on an representive data set.
            Returns:
            --   : The Frechet Distance.

            """
        x_activations, y_activations = self.get_activations(x), self.get_activations(y)

        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()

        mu1, mu2 = np.mean(x_numpy), np.mean(y_numpy)
        cov1, cov2 = np.cov(x_numpy), np.cov(y_numpy)

        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)


def PSNR(x, y):
    mse = F.mse_loss(x, y).detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    psnr = 20 * np.log10(np.max(x)) - 10 * np.log10(mse)
    return psnr


class FrechetDistance:
    def __init__(self, in_size=(3, 256, 256)):
        self.inception_network = inception_v3(pretrained=True)

        self.in_size = in_size

    def resize_images(self, images):
        """
        :param images: np.ndarray of size [b_size, N, h, w]
        :return: resized image
        """

        for image in images:
            image = image.resize(3, 299, 299)

        return images

    def get_activations(self, x):
        """
        :param x: Torch tensor of size [b_size, N, h, w]
        :return: activations from Inception V3 model
        """

        x_resized = x.detach().cpu().numpy()
        x_resized = self.resize_images(x_resized)
        x_resized = torch.FloatTensor(x_resized)

        # Trigger output hook
        self.inception_network(x_resized)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x_resized.shape[0], 2048)
        return activations

    def calculate_fid(self, x, y):
        """
            Frechet Inception Distance
            https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

            Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            Stable version by Dougal J. Sutherland.
            Params:
            -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                     inception net ( like returned by the function 'get_predictions')
                     for generated samples.
            -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                       on an representive data set.
            -- sigma1: The covariance matrix over activations of the pool_3 layer for
                       generated samples.
            -- sigma2: The covariance matrix over activations of the pool_3 layer,
                       precalcualted on an representive data set.
            Returns:
            --   : The Frechet Distance.

            """
        x_activations, y_activations = self.get_activations(x), self.get_activations(y)

        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()

        mu1, mu2 = np.mean(x_numpy), np.mean(y_numpy)
        cov1, cov2 = np.cov(x_numpy), np.cov(y_numpy)

        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)


def PSNR(x, y):
    mse = F.mse_loss(x, y).detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    psnr = 20 * np.log10(np.max(x)) - 10 * np.log10(mse)
    return psnr


class FrechetDistance:
    def __init__(self, in_size=(3, 256, 256)):
        self.inception_network = inception_v3(pretrained=True)

        self.in_size = in_size

    def resize_images(self, images):
        """
        :param images: np.ndarray of size [b_size, N, h, w]
        :return: resized image
        """

        for image in images:
            image = image.resize(3, 299, 299)

        return images

    def get_activations(self, x):
        """
        :param x: Torch tensor of size [b_size, N, h, w]
        :return: activations from Inception V3 model
        """

        x_resized = x.detach().cpu().numpy()
        x_resized = self.resize_images(x_resized)
        x_resized = torch.FloatTensor(x_resized)

        # Trigger output hook
        self.inception_network(x_resized)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x_resized.shape[0], 2048)
        return activations

    def calculate_fid(self, x, y):
        """
            Frechet Inception Distance
            https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

            Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            Stable version by Dougal J. Sutherland.
            Params:
            -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                     inception net ( like returned by the function 'get_predictions')
                     for generated samples.
            -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                       on an representive data set.
            -- sigma1: The covariance matrix over activations of the pool_3 layer for
                       generated samples.
            -- sigma2: The covariance matrix over activations of the pool_3 layer,
                       precalcualted on an representive data set.
            Returns:
            --   : The Frechet Distance.

            """
        x_activations, y_activations = self.get_activations(x), self.get_activations(y)

        x_numpy = x.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy()

        mu1, mu2 = np.mean(x_numpy), np.mean(y_numpy)
        cov1, cov2 = np.cov(x_numpy), np.cov(y_numpy)

        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
