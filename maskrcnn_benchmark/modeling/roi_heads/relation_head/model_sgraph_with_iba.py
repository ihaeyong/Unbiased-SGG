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

import skimage.transform

class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels, device):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)

        self.f_type = 'gaussian'

        if self.f_type is 'gaussian':
            kernel_2d = self.gaussian_2d(sigma, kernel_size)

            # expand in channel dimension
            kernel_3d = kernel_2d.expand(channels, 1, -1, -1)
            self.conv = nn.Conv2d(in_channels=channels,
                                  out_channels=channels,
                                  padding=0,
                                  kernel_size=kernel_size,
                                  groups=channels,
                                  bias=False)
            self.conv.weight.data.copy_(kernel_3d)
            self.conv.weight.requires_grad = False

        elif self.f_type is 'multi-gaussian':
            kernel_2d_1s = self.gaussian_2d(sigma, kernel_size)
            kernel_2d_2s = self.gaussian_2d(sigma+1, kernel_size)

            # expand in channel dimension
            kernel_3d_1s = kernel_2d_1s.expand(channels // 2, 1, -1, -1)
            kernel_3d_2s = kernel_2d_2s.expand(channels // 2, 1, -1, -1)
            self.conv_s1 = nn.Conv2d(in_channels=channels,
                                     out_channels=channels // 2,
                                     padding=0,
                                     kernel_size=kernel_size,
                                     groups=channels // 2,
                                     bias=False)

            self.conv_s2 = nn.Conv2d(in_channels=channels,
                                     out_channels=channels // 2,
                                     padding=0,
                                     kernel_size=kernel_size,
                                     groups=channels // 2,
                                     bias=False)

            self.conv_s1.weight.data.copy_(kernel_3d_1s)
            self.conv_s2.weight.data.copy_(kernel_3d_2s)

            self.conv_s1.weight.requires_grad = False
            self.conv_s2.weight.requires_grad = False

        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def gaussian_2d(self, sigma, kernel_size):

        variance = sigma ** 2.

        # x_cord : [kernel_size, kernel_size]
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)

        # y_cord : [kernel_size, kernel_size]
        y_grid = x_grid.t()

        # xy_grid : [5,5,2]
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()

        return kernel_2d

    def gabor_2d(self, sigma, kernel_size):
        x, y = np.meshgrid(np.arange(-float(kernel_size/2), kernel_size/2),
                           np.arange(-float(kernel_size/2), kernel_size/2))

        y = skimage.transform.rotate(y, 35)
        x2 = skimage.transform.rotate(x, -35)
        sigma = 0.65 * np.pi
        lmbda = 1.5 * sigma
        gamma = 1.3
        gabor = np.exp(-(x**2 + gamma*y**2)/(2*sigma**2))*np.cos(2*np.pi*x2/lmbda)
        kernel_2d = torch.FloatTensor(gabor)

        return kernel_2d

    def forward(self, x):
        # x : [batch, 32, 25, 25]
        # pad : [batch, 32, 29, 29]
        # conv : [batch, 32, 25, 25]
        if self.f_type is 'gaussian':
            return self.conv(self.pad(x))
        else:
            out_s1 = self.conv_s1(self.pad(x))
            out_s2 = self.conv_s2(self.pad(x))

            return torch.cat((out_s1, out_s2), 1)


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
                 fmap_size=25, channel=32, pred_prop=None):
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

        e_type = 'inv_prop'
        # predicate proportion
        if e_type == 'prior':
            self.pred_prop = np.array(pred_prop)
            self.pred_prop = np.concatenate(([1], self.pred_prop), 0)
            self.pred_prop[0] = 1.0 - self.pred_prop[1:-1].sum()
        elif e_type == 'inv_prop':
            fg_rel = np.load('./datasets/vg/fg_matrix.npy')
            bg_rel = np.load('./datasets/vg/bg_matrix.npy')
            fg_rel[:,:,0] = bg_rel
            pred_freq = fg_rel.sum(0).sum(0)

            # pred prop
            self.pred_prop = pred_freq / pred_freq.max()

            # pred margin
            pred_margin = 1.0 / np.power(pred_freq, 1/2)
            max_m = 0.03
            self.pred_margin = pred_margin * (max_m / pred_margin.max())

        elif e_type == 'prop':

            fg_rel = np.load('./datasets/vg/fg_matrix.npy')
            bg_rel = np.load('./datasets/vg/bg_matrix.npy')
            fg_rel[:,:,0] = bg_rel
            pred_freq = fg_rel.sum(0).sum(0)

            # pred prop
            self.pred_prop = pred_freq / pred_freq.max()

            # pred margin
            pred_margin = np.power(pred_freq, 1/2)
            max_m = 0.01
            self.pred_margin = pred_margin * (max_m / pred_margin.max())

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

    def gaussian(self, ins, rel_labels, scale):
        batch_size = ins.size(0)
        bg_idx = np.where(rel_labels.cpu() == 0)[0]
        fg_idx = np.where(rel_labels.cpu() > 0)[0]

        rand = torch.rand_like(ins)
        eps = torch.zeros_like(ins)
        n_type = "pred_avg_margin_sigmoid"

        if n_type is 'normal':
            bg_stddev = len(bg_idx) / batch_size * rand[bg_idx,]
            fg_stddev = len(fg_idx) / batch_size * rand[fg_idx,]

            eps[bg_idx, ] = ins[bg_idx,] + bg_stddev * scale
            eps[fg_idx, ] = ins[fg_idx,] + fg_stddev * scale

        elif n_type is "inv":
            bg_stddev = len(fg_idx) / batch_size * rand[bg_idx,]
            fg_stddev = len(bg_idx) / batch_size * rand[fg_idx,]

            eps[bg_idx, ] = ins[bg_idx,] + bg_stddev * scale
            eps[fg_idx, ] = ins[fg_idx,] + fg_stddev * scale

        elif n_type is "prop":
            stddev = self.pred_prop[rel_labels.cpu()][:,None,None,None]
            stddev = torch.FloatTensor(stddev).cuda(rand.get_device())

            eps = ins + ins * stddev * scale

        elif n_type is "pred_margin":
            stddev = self.pred_margin[rel_labels.cpu()][:,None,None,None]
            stddev = torch.FloatTensor(stddev).cuda(rand.get_device())

            eps = rand * stddev

        elif n_type is "pred_avg_margin":
            stddev = self.pred_margin[rel_labels.cpu()][:,None,None,None]
            stddev = torch.FloatTensor(stddev).cuda(rand.get_device())

            eps = ins + rand * stddev

        elif n_type is "pred_avg_margin_sigmoid":
            stddev = self.pred_margin[rel_labels.cpu()][:,None,None,None]
            stddev = torch.FloatTensor(stddev).cuda(rand.get_device())

            eps = torch.sigmoid(ins + rand * stddev)

        return eps


    def forward(self, lamb, r_, rel_labels=None):
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
        if self.training and True:
            eps = self.gaussian(lamb, rel_labels, 1e-8)
        else:
            eps = 0.0

        if True:
            noise_var = (1-(lamb + eps)/2.0)**2
        else:
            noise_var = (1-lamb)**2

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
