import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, grad

from maskrcnn_benchmark.modeling.utils import cat
from .utils_relation import layer_init, seq_init

'''
Notes:
    Agent navigation for image classification. We propose
    an image classification task starting with a masked image
    where the agent starts at a random location on the image. It
    can unmask windows of the image by moving in one of 4 directions: 
    {UP, DOWN, LEFT, RIGHT}. At each timestep it
    also outputs a probability distribution over possible classes
    C. The episode ends when the agent correctly classifies the
    image or a maximum of 20 steps is reached. The agent receives a 
    -0.1 reward at each timestep that it misclassifies the
    image. The state received at each time step is the full image
    with unobserved parts masked out.
    -- for now, agent outputs direction of movement and class prediction (0-9)
    -- correct guess ends game
'''

class VGEnv(nn.Module):
    metadata = {'render.modes': ['human']}

    def __init__(self, type='train', seed=2069):

        super(VGEnv, self).__init__()

        if seed:
            np.random.seed(seed=seed)

        self.mask = np.zeros((4096))
        self.MAX_STEPS = 1

        # predicate inverse proportion
        fg_rel = np.load('./datasets/vg/fg_matrix.npy')
        bg_rel = np.load('./datasets/vg/bg_matrix.npy')
        fg_rel[:,:,0] = bg_rel
        pred_freq = fg_rel.sum(0).sum(0)

        # pred inverse proportion
        pred_inv_prop = 1.0 / np.power(pred_freq, 1/2)
        max_m = 1.0
        self.pred_inv_prop = pred_inv_prop * (max_m / pred_inv_prop.max())



        # transfer
        enc_transf = [
            nn.Linear(4096, 4096 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 2, 4096 // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 4, 4096 // 8, bias=True),
            nn.ReLU(inplace=True),
        ]

        dec_transf = [
            nn.Linear(4096 // 8, 4096 // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 4, 4096 // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 2, 4096, bias=True),
        ]

        self.enc_transf = nn.Sequential(*enc_transf)
        self.dec_transf = nn.Sequential(*dec_transf)
        self.enc_transf.apply(seq_init)
        self.dec_transf.apply(seq_init)

        self.mean = nn.Linear(4096 // 8, 4096 // 8, bias=True)
        self.std = nn.Linear(4096 // 8, 4096 // 8, bias=True)
        layer_init(self.mean, xavier=True)
        layer_init(self.std, xavier=True)

        self.mse_loss = nn.MSELoss() #nn.CrossEntropyLoss()

    def make_step(self, grad, attack='l2', step_size=0.1):

        if attack == 'l2':
            grad_norm = torch.norm(grad, dim=1).view(-1, 1)
            scaled_grad = grad / (grad_norm + 1e-10)
            step = step_size * scaled_grad

        elif attack == 'inf':
            step = step_size * torch.sign(grad)

        elif attack == 'none':
            step = step_size * grad

        return step

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = torch.randn_like(sd) # Sample from standard normal
        z = e.mul(sd).add_(mean)

        return z

    def rand_perturb(self, inputs, attack, eps=0.5):

        if attack == 'inf':
            r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
        elif attack == 'randn':
            r_inputs = torch.randn_like(inputs) * inputs.max(1)[0][:,None]
        elif attack == 'l2':
            r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
        elif attack == 'zero':
            r_inputs = torch.zeros_like(inputs)

        return r_inputs

    # Loss function
    def criterion(self, x_out, x_in, z_mu, z_logvar):

        mse_loss = self.mse_loss(x_out, x_in)
        kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        loss = (mse_loss + kld_loss) / x_out.size(0) # normalize by batch size

        return loss

    def _transform(self, x):

        enc_x = self.enc_transf(x)
        mean_x = self.mean(enc_x)
        logvar_x = self.std(enc_x)
        z = self.sample_normal(mean_x, logvar_x)
        dec_x = self.dec_transf(z)

        loss = self.criterion(dec_x, x, mean_x, logvar_x)
        grad, = torch.autograd.grad(loss, [z])

        noise = self.rand_perturb(logvar_x, 'l2', self.eps)
        logvar_x = logvar_x + noise
        z = self.sample_normal(mean_x, logvar_x)
        z = z - self.make_step(grad, 'l2', self.step_size)

        dec_x = self.dec_transf(z)

        return dec_x

    def torch_to_numpy(self, tensor):
        return tensor.data.numpy()

    def step(self, action, value, X):

        # action a consists of
        #   1. direction in {N, S, E, W}, determined by = a (mod 4)
        #   2. predicted class (0-9), determined by floor(a / 4)
        dir, Y_pred = action % 3, action // 3

        self.steps += 1

        move_map = {
            0:  -0.01,
            1:   0.00,
            2:   0.01,
        }
        self.eps += move_map[dir]

        obs = self._transform(X)

        # make move and reveal square
        #if self.Y[0] == 0:
        #    y = self.Y[1]
        #else:
        #    y = self.Y[0]
        y = self.Y[self.i]

        # -0.1 penalty for each additional timestep
        # +1.0 for correct guess
        if False:
            inv_freq = int(Y_pred == y) * self.pred_inv_prop[y] * 30.0
            reward = -self.pred_inv_prop.min() * 2 + inv_freq
        else:
            inv_freq = int(Y_pred == y)
            reward = -0.1 + inv_freq


        # game ends if prediction is correct or max steps is reached
        done = Y_pred == y or self.steps >= self.MAX_STEPS

        if Y_pred != y :
            if self.steps >= self.MAX_STEPS:
                y = 0

        # info is empty (for now)
        info = {}

        return obs, y, reward, done, info

    def reset(self, x, y=None, y_prob=None):
        # resets the environment and returns initial observation
        self.eps = 0.5
        self.step_size = 0.01
        self.Y = self.torch_to_numpy(y)

        #y_prob = 1 - self.torch_to_numpy(y_prob)
        y_prob = self.torch_to_numpy(y_prob)
        pi = y_prob / y_prob.sum()
        self.i = np.random.choice(np.arange(5), p=pi)

        self.idx = 1
        self.steps = 0
        self.MAX_STEPS = 9

        return self._transform(x[None])

