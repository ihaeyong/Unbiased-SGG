import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils_motifs import to_onehot

from scipy.stats import entropy, skew

torch.cuda.manual_seed(2021)


class LDAMLoss(nn.Module):

    def __init__(self, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        cls_num_list = np.load('./datasets/vg/obj_freq.npy')
        cls_num_list[0] = cls_num_list.max()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forwarzd(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class ObjWeight(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, cls_num_list=None):
        super(ObjWeight, self).__init__()

        '''
        we set the backgounds to the maximum number of samples
        since its sample size is zero
        '''
        self.cls_num_list = cls_num_list
        self.cls_num_list[0] = cls_num_list.max()

        obj_prop = cls_num_list / cls_num_list.sum()
        self.obj_prop = np.array(obj_prop)
        self.obj_idx = self.obj_prop.argsort()[::-1]

    def forward(self, obj_logits, obj_labels, gamma=0.01):

        with torch.no_grad():
            freq_bias = torch.sigmoid(obj_logits)

            # Entropy * scale
            topk_prob, topk_idx = F.softmax(obj_logits,1).topk(1)
            topk_true_mask = (topk_idx[:,0] == obj_labels).float().data.cpu().numpy()
            topk_false_mask = (topk_idx[:,0] != obj_labels).float().data.cpu().numpy()

            true_idx = np.where(topk_true_mask == 1)[0]
            false_idx = np.where(topk_false_mask == 1)[0]

            batch_freq = freq_bias.data.cpu().numpy()
            if False:
                cls_num_list = batch_freq.sum(0)
            else:
                cls_num_list = self.cls_num_list

            cls_order = batch_freq[:, self.obj_idx]

            w_type = 'avg'
            if w_type == 'full':
                ent_v = entropy(cls_order, base=151, axis=1).mean()
                skew_v = skew(cls_order, axis=1).mean()

            elif w_type == 'false':
                ent_v = entropy(cls_order, base=151, axis=1) * topk_false_mask
                ent_v = ent_v.sum() / (topk_false_mask.sum()+1)
                skew_v = skew(cls_order, axis=1) * topk_false_mask
                skew_v = skew_v.sum() / (topk_false_mask.sum()+1)

            elif w_type == 'avg':
                ent_v = entropy(cls_order, base=151, axis=1)
                skew_v = skew(cls_order, axis=1)

                ent_false_v = ent_v[false_idx].mean() if len(false_idx) > 0 else 0
                ent_true_v = ent_v[true_idx].mean() if len(true_idx) > 0 else 0
                skew_false_v = skew_v[false_idx].mean() if len(false_idx) > 0 else 0
                skew_true_v = skew_v[true_idx].mean() if len(true_idx) > 0 else 0

                alpha = topk_false_mask.sum() / topk_false_mask.shape[0]
                ent_v = ent_false_v * alpha + ent_true_v * (1-alpha)
                skew_v = skew_false_v * alpha + skew_true_v * (1-alpha)

            # skew_v > 0 : more weight in the left tail
            # skew_v < 0 : more weight in the right tail
            skew_th = 0.2 # default 2.2
            if skew_v > skew_th:
                beta = 1.0 - ent_v * 1.0
            elif skew_v < -skew_th:
                beta = 1.0 - ent_v * 1.0
            else:
                beta = 0.0

            beta = np.clip(beta, 0,1)

            effect_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effect_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            obj_weight = torch.FloatTensor(per_cls_weights).cuda()

            obj_margin = None
        return  obj_weight, obj_margin

class RelWeight(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, predicate_proportion, temp=1):
        super(RelWeight, self).__init__()

        self.pred_prop = np.array(predicate_proportion)
        self.pred_prop = np.concatenate(([1], self.pred_prop), 0)
        self.pred_prop[0] = 1.0 - self.pred_prop[1:-1].sum()
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

    def forward(self, rel_logits, freq_bias, rel_labels, gamma=0.01):

        # scaled mean of logits
        with torch.no_grad():
            # Entropy * scale
            # topk logits
            topk_prob, topk_idx = F.softmax(rel_logits,1).topk(1)
            topk_true_mask = (topk_idx[:,0] == rel_labels).float().data.cpu().numpy()
            topk_false_mask = (topk_idx[:,0] != rel_labels).float().data.cpu().numpy()

            true_idx = np.where(topk_true_mask == 1)[0]
            false_idx = np.where(topk_false_mask == 1)[0]

            batch_freq = freq_bias.data.cpu().numpy()
            cls_num_list = batch_freq.sum(0)
            cls_order = batch_freq[:, self.pred_idx]

            w_type = 'avg'
            if w_type == 'full':
                cls_num_list = batch_freq.sum(0)
                cls_order = batch_freq[:, self.pred_idx]
                ent_v = entropy(cls_order, base=51, axis=1).mean()
                skew_v = skew(cls_order, axis=1).mean()

            elif w_type == 'false':
                ent_v = entropy(cls_order, base=51, axis=1) * topk_false_mask
                skew_v = skew(cls_order, axis=1) * topk_false_mask
                ent_v = ent_v.sum() / (topk_false_mask.sum() + 1)
                skew_v = skew_v.sum() / (topk_false_mask.sum() + 1)

            elif w_type == 'avg':
                ent_v = entropy(cls_order, base=51, axis=1)
                skew_v = skew(cls_order, axis=1)

                ent_false_v = ent_v[false_idx].mean() if len(false_idx) > 0 else 0
                ent_true_v = ent_v[true_idx].mean() if len(true_idx) > 0 else 0
                skew_false_v = skew_v[false_idx].mean() if len(false_idx) > 0 else 0
                skew_true_v = skew_v[true_idx].mean() if len(true_idx) > 0 else 0

                alpha = topk_false_mask.sum() / topk_false_mask.shape[0]
                ent_v = ent_false_v * alpha + ent_true_v * (1-alpha)
                skew_v = skew_false_v * alpha + skew_true_v * (1-alpha)

            elif w_type == 'avg_smooth':
                ent_v = entropy(cls_order, base=51, axis=1)
                skew_v = skew(cls_order, axis=1)

                ent_false_v = ent_v * topk_false_mask
                ent_true_v = ent_v * topk_true_mask
                skew_false_v = skew_v * topk_false_mask
                skew_true_v = skew_v * topk_true_mask

                alpha = topk_false_mask.sum() / topk_false_mask.shape[0]
                ent_v = ent_false_v.mean() * alpha + ent_true_v.mean() * (1-alpha)
                skew_v = skew_false_v.mean() * alpha + skew_true_v.mean() * (1-alpha)

            skew_th = 0.9 # default 0.9
            if skew_v > skew_th :
                beta = 1.0 - ent_v * 0.05
            elif skew_v < -skew_th :
                beta = 1.0 - ent_v * 0.05
            else:
                beta = 0.0

            beta = np.clip(beta, 0,1)

            effect_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effect_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            rel_weight = torch.FloatTensor(per_cls_weights).cuda()

            rel_margin = None

        return  rel_weight, rel_margin

