import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

import numpy as np
import math

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from tqdm import tqdm
from scipy.stats import norm

class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels, device):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        # 1, 2, 3, 4
        x_cord = torch.arange(kernel_size)
        # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              padding=0,
                              kernel_size=kernel_size,
                              groups=channels,
                              bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def forward(self, x):
        # x : [batch, 32, 25, 25]
        # pad : [batch, 32, 29, 29]
        # conv : [batch, 32, 25, 25]
        return self.conv(self.pad(x))

class AttributionBottleneck(nn.Module):

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + Variable(noise_std.data * eps)

    @staticmethod
    def _calc_capacity(mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())



class PerSampleBottleneck(AttributionBottleneck):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    """
    def __init__(self, mean=None, std=None,
                 sigma=0, device=None, relu=False,
                 fmap_size=25, channel=32):
        """
        :param mean: The empirical mean of the activations of the layer
        :param std: The empirical standard deviation of the activations of the layer
        :param sigma: The standard deviation of the gaussian kernel to smooth the mask, or None for no smoothing
        :param device: GPU/CPU
        :param relu: True if output should be clamped at 0, to imitate a post-ReLU distribution
        """
        super().__init__()
        self.device = device
        self.relu = relu
        self.initial_value = 5.0
        self.mean = nn.Conv2d(channel,channel,1,1)
        self.std = nn.Sequential(
            nn.Conv2d(channel,channel,1,1),)

        self.buffer_capacity = None
        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1
            channels = channel
            self.smooth = SpatialGaussianKernel(
                kernel_size, sigma, channels, device=self.device)
        else:
            self.smooth = None

        self.beta = 10.0 / (channel * fmap_size * fmap_size) # 1/k


    def forward(self, lamb, r_):
        """ Remove information from r by performing a sampling step,
        parametrized by the mask alpha """
        # Smoothen and expand a on batch dimension
        lamb = lamb.expand(r_.shape[0], r_.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        mean = self.mean(r_)
        std = F.softplus(self.std(r_))

        if True:
            r_norm = (r_ - mean) / std
        else:
            r_p = Normal(mean, std)
            r_norm = r_p.sample()

        # Get sampling parameters
        eps = 1e-8
        noise_var = (1-lamb + eps)**2
        scaled_signal = r_norm * lamb
        noise_log_var = torch.log(noise_var)

        # Sample new output values from p(z|r)
        z_norm = self._sample_z(scaled_signal, noise_log_var)

        self.buffer_capacity = self._calc_capacity(
            scaled_signal, noise_log_var) * self.beta

        # Denormalize z to match magnitude of r
        z = z_norm * std + mean

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z
