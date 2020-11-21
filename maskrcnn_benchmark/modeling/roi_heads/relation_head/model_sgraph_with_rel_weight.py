import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.stats import entropy, skew

class RelWeight(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, predicate_proportion, temp=1):
        super(RelWeight, self).__init__()

        self.pred_prop = np.array(predicate_proportion)
        self.pred_prop[0] = 1.0 # set as backgrounds
        self.pred_idx = self.pred_prop.argsort()[::-1]

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

        #freq_dists = F.softmax(freq_bias, 1)
        batch_freq = freq_bias.sum(0).data.cpu().numpy()
        #log_batch_freq = np.log(1.0 + batch_freq)
        # temp = [1, 1000]
        #cls_num_list = self.softmax_with_temp(log_batch_freq, self.temp)
        cls_num_list = batch_freq

        # entropy * scale
        cls_order = cls_num_list[self.pred_idx]
        ent_v = entropy(cls_order, base=51)
        # skew_v > 0 : more weight in the left tail
        # skew_v < 0 : more weight in the right tail
        skew_v = skew(cls_order)
        if skew_v > 1.0 :
            beta = 1.0 - ent_v * 0.2
        else:
            beta = 0.0

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        rel_weight = torch.FloatTensor(per_cls_weights).cuda()

        return  rel_weight


