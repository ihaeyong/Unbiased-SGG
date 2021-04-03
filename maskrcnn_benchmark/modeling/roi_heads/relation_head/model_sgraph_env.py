import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

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

class VGEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, type='train', seed=2069):

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

    def torch_to_numpy(self, tensor):
        return tensor.data.numpy()

    def step(self, action, value, X):

        # action a consists of
        #   1. direction in {N, S, E, W}, determined by = a (mod 4)
        #   2. predicted class (0-9), determined by floor(a / 4)
        dir, Y_pred = action % 3, action // 3

        self.steps += 1
        self._update_obs(X)

        move_map = {
            0:  1,  # increase variance
            1:  0,  # increase variance
            2: -1,  # decrease variance
        }

        # make move and reveal square
        self.Y += move_map[dir]
        done = self.Y == 0 or self.Y > 50
        if self.Y > 50:
            self.Y = 0

        # state (observation) consists of masked image (h x w)
        obs,eps = self._get_obs()

        # -0.1 penalty for each additional timestep
        # +1.0 for correct guess
        if False:
            inv_freq = int(Y_pred == self.Y) * self.pred_inv_prop[self.Y]
            reward = -0.1 + int(Y_pred == self.Y) + inv_freq
        else:
            inv_freq = int(Y_pred == self.Y) * self.pred_inv_prop[self.Y]
            reward = -self.pred_inv_prop.min() * 0.1 + inv_freq

        # game ends if prediction is correct or max steps is reached
        done = Y_pred == self.Y or self.steps >= self.MAX_STEPS

        if self.steps >= self.MAX_STEPS:
            if Y_pred != self.Y:
                self.Y = 0

        # info is empty (for now)
        info = {}

        return obs, self.Y, reward, done, info

    def reset(self, x, y=None):
        # resets the environment and returns initial observation
        self.mu = 0.0
        self.var = 0.01
        self.var_eps = 0.01
        self.noise = np.zeros((4096))
        self.eps = np.zeros((512))

        self.X = self.torch_to_numpy(x)
        self.Y = self.torch_to_numpy(y)
        self.steps = 0
        self.MAX_STEPS = 9

        return self._get_obs()

    def _update_obs(self, x):
        self.X = self.torch_to_numpy(x[0].cpu())

    def _get_obs(self):
        obs = self.X + self.noise
        eps = self.eps
        #assert self.observation_space.contains(obs)
        return obs, eps

