import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from lib.sparse_targets import FrequencyBias

from scipy.stats import entropy

class RelWeight(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, temp=1):
        super(RelWeight, self).__init__()

        #self.freq_bias = FrequencyBias()
        self.temp = temp

    def softmax_with_temp(self,z, T=1):

        z = np.array(z)
        z = z / T
        max_z = np.max(z)
        exp_z = np.exp(z-max_z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        return y

    def forward(self, freq_bias):

        freq_dists = F.softmax(freq_bias, 1)
        batch_freq = freq_dists.sum(0).data.cpu().numpy()
        log_batch_freq = np.log(10.0 + batch_freq)

        # temp = [1, 1000]
        cls_num_list = self.softmax_with_temp(log_batch_freq, self.temp)

        # entropy * scale
        beta = 1.0 - entropy(cls_num_list, base=51) * 1.0

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        rel_weight = torch.FloatTensor(per_cls_weights).cuda()

        return  rel_weight


