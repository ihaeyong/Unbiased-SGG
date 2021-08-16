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

    def softmax(self, x, temp=1.0):
        x = x / temp

        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x

    # Moment with optional pre-computed mean, equal to a.mean(axis, keepdims=True)
    def _moment(self, a, moment, axis, mean=None):
        if np.abs(moment - np.round(moment)) > 0:
            raise ValueError("All moment parameters must be integers")

        if moment == 0 or moment == 1:
            # By definition the zeroth moment about the mean is 1, and the first
            # moment is 0.
            shape = list(a.shape)
            del shape[axis]
            dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

            if len(shape) == 0:
                return dtype(1.0 if moment == 0 else 0.0)
            else:
                return (np.ones(shape, dtype=dtype) if moment == 0
                        else np.zeros(shape, dtype=dtype))
        else:
            # Exponentiation by squares: form exponent sequence
            n_list = [moment]
            current_n = moment
            while current_n > 2:
                if current_n % 2:
                    current_n = (current_n - 1) / 2
                else:
                    current_n /= 2
                    n_list.append(current_n)

            # Starting point for exponentiation by squares
            mean = a.mean(axis, keepdims=True) if mean is None else mean
            a_zero_mean = a - mean
            if n_list[-1] == 1:
                s = a_zero_mean.copy()
            else:
                s = a_zero_mean**moment

            # Perform multiplications
            for n in n_list[-2::-1]:
                s = s**2
                if n % 2:
                    s *= a_zero_mean
            return np.mean(s, axis)

    def _skew(self, x, target=None, axis=1):

        '''
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        g_1=\frac{m_3}{m_2^{3/2}}
        '''
        if target is None:
            mean = x.mean(axis=1, keepdims=True)
        else:
            mean = target

        m2 = self._moment(x, moment=2, axis=axis, mean=mean)
        m3 = self._moment(x, moment=3, axis=axis, mean=mean)

        with np.errstate(all='ignore'):
            zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(1))**2)
            vals = np.where(zero, 0, m3 / m2**1.5)

        return vals

    def forward(self, obj_logits, obj_labels, gamma=0.01):

        batch_size = obj_logits.size(0)
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

            w_type = 'sample-target'
            if w_type == 'full':
                cls_order = batch_freq[:, self.obj_idx]
                ent_v = entropy(cls_order, base=151, axis=1).mean()
                skew_v = skew(cls_order, axis=1).mean()

            elif w_type == 'sample':
                cls_num_list = batch_freq.sum(0)
                cls_order = self.softmax(batch_freq[:, self.obj_idx], temp=1.0)
                ent_v = entropy(cls_order, base=151, axis=1)
                skew_v = skew(cls_order, axis=1)

            elif w_type == 'sample-target':
                cls_num_list = batch_freq.sum(0)
                cls_order_target = self.softmax(batch_freq)
                cls_order = cls_order_target[:, self.obj_idx]
                ent_v = entropy(cls_order, base=151, axis=1)
                target = np.take_along_axis(cls_order_target, obj_labels.data.cpu().numpy()[:,None], axis=1)
                skew_v = self._skew(cls_order, target, axis=1)

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
            if False:
                if skew_v > skew_th:
                    beta = 1.0 - ent_v * ent_w
                elif skew_v < -skew_th:
                    beta = 1.0 - ent_v * ent_w
                else:
                    beta = 0.0

                beta = np.clip(beta, 0,1)

                effect_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                obj_weight = torch.FloatTensor(per_cls_weights).cuda()
            elif False:
                skew_pos_th = 1.9 # default 2.2
                skew_neg_th = 1.9 # default 2.2
                ent_pos_w = 1.0
                ent_neg_w = 0.7
                #sample-ent
                pos_mask = (skew_v > skew_pos_th).astype(float)
                neg_mask = (skew_v < -skew_neg_th).astype(float)

                pos_beta = (1.0 - ent_v * ent_pos_w) * pos_mask
                neg_beta = (1.0 - ent_v * ent_neg_w) * neg_mask

                if False:
                    beta = pos_beta + neg_beta
                else:
                    beta = pos_beta

                effect_num = [1.0 - np.power(b, cls) for b, cls in zip(beta,cls_num_list[None,:].repeat(batch_size, 0))]
                per_cls_weights = (1.0 - beta[:,None]) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights,1)[:,None] * len(cls_num_list)
            elif True:
                skew_neg_th = skew_v.mean() - 0.7
                ent_neg_w = 1.0
                neg_mask = (skew_v > skew_neg_th).astype(float)
                neg_beta = (1.0 - ent_v * ent_neg_w) * neg_mask
                beta = neg_beta

                effect_num = [1.0 - np.power(b, cls) for b, cls in zip(beta,cls_num_list[None,:].repeat(batch_size, 0))]
                per_cls_weights = (1.0 - beta[:,None]) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights,1)[:,None] * len(cls_num_list)

            obj_margin = None
            obj_weight = torch.FloatTensor(per_cls_weights).cuda()
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

        self.register_buffer('mean', torch.zeros(51, 1))
        self.register_buffer('var', torch.zeros(51, 1))

    def moving_avg(self, holder, inputs, avg_ratio=0.0005):
        with torch.no_grad():
            holder = holder * (1 - avg_ratio) + avg_ratio * inputs
        return holder

    def update(self, margin, label):
        # get mean embedding
        self.mean[label] = self.moving_avg(self.mean[label], margin.cpu().detach())

        var = torch.pow(self.mean[label] - margin.cpu().detach(), 2)
        self.var[label] = self.moving_avg(self.var[label], var.detach())

    def margin(self, x, labels):

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        dists = F.softmax(x, dim=1)
        neg_labels = 1.0 - index_float
        neg_dists = dists * neg_labels

        min_pos_prob = torch.gather(dists, 1, labels[:,None]).data
        max_neg_prob = neg_dists.max(1)[0].data[:,None]

        # estimate the margin between dists and gt labels
        max_margin = min_pos_prob - max_neg_prob
        return max_margin


    def softmax(self, x, temp=1.0):
        x = x / temp

        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x

    # Moment with optional pre-computed mean, equal to a.mean(axis, keepdims=True)
    def _moment(self, a, moment, axis, mean=None):
        if np.abs(moment - np.round(moment)) > 0:
            raise ValueError("All moment parameters must be integers")

        if moment == 0 or moment == 1:
            # By definition the zeroth moment about the mean is 1, and the first
            # moment is 0.
            shape = list(a.shape)
            del shape[axis]
            dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

            if len(shape) == 0:
                return dtype(1.0 if moment == 0 else 0.0)
            else:
                return (np.ones(shape, dtype=dtype) if moment == 0
                        else np.zeros(shape, dtype=dtype))
        else:
            # Exponentiation by squares: form exponent sequence
            n_list = [moment]
            current_n = moment
            while current_n > 2:
                if current_n % 2:
                    current_n = (current_n - 1) / 2
                else:
                    current_n /= 2
                    n_list.append(current_n)

            # Starting point for exponentiation by squares
            mean = a.mean(axis, keepdims=True) if mean is None else mean
            a_zero_mean = a - mean
            if n_list[-1] == 1:
                s = a_zero_mean.copy()
            else:
                s = a_zero_mean**moment

            # Perform multiplications
            for n in n_list[-2::-1]:
                s = s**2
                if n % 2:
                    s *= a_zero_mean
            return np.mean(s, axis)

    def _skew(self, x, target=None, axis=1):

        '''
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.
        g_1=\frac{m_3}{m_2^{3/2}}
        '''

        if target is None:
            mean = x.mean(axis=1, keepdims=True)
        else:
            mean = target

        m2 = self._moment(x, moment=2, axis=axis, mean=mean)
        m3 = self._moment(x, moment=3, axis=axis, mean=mean)

        with np.errstate(all='ignore'):
            zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(1))**2)
            vals = np.where(zero, 0, m3 / m2**1.5)

        return vals

    def forward(self, rel_logits, freq_bias, rel_labels, gamma=0.01):

        # scaled mean of logits
        batch_size = rel_logits.size(0)
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

            w_type = 'sample-target'
            if w_type == 'full':
                cls_num_list = batch_freq.sum(0)
                cls_order = batch_freq[:, self.pred_idx]
                ent_v = entropy(cls_order, base=51, axis=1).mean()
                skew_v = skew(cls_order, axis=1).mean()

            elif w_type == 'sample':
                cls_num_list = batch_freq.sum(0)
                cls_order = self.softmax(batch_freq[:, self.pred_idx], temp=1.0)
                ent_v = entropy(cls_order, base=51, axis=1)
                skew_v = skew(cls_order, axis=1)

            elif w_type == 'sample-target':
                cls_num_list = batch_freq.sum(0)
                cls_order_target = self.softmax(batch_freq)
                cls_order = cls_order_target[:, self.pred_idx]
                ent_v = entropy(cls_order, base=51, axis=1)
                target = np.take_along_axis(cls_order_target, rel_labels.data.cpu().numpy()[:,None], axis=1)
                skew_v = self._skew(cls_order, target, axis=1)

            elif w_type == 'avg':
                cls_order = batch_freq[:, self.pred_idx]
                ent_v = entropy(cls_order, base=51, axis=1)
                skew_v = skew(cls_order, axis=1)

                ent_false_v = ent_v[false_idx].mean() if len(false_idx) > 0 else 0
                ent_true_v = ent_v[true_idx].mean() if len(true_idx) > 0 else 0
                skew_false_v = skew_v[false_idx].mean() if len(false_idx) > 0 else 0
                skew_true_v = skew_v[true_idx].mean() if len(true_idx) > 0 else 0

                alpha = topk_false_mask.sum() / topk_false_mask.shape[0]
                ent_v = ent_false_v * alpha + ent_true_v * (1-alpha)
                skew_v = skew_false_v * alpha + skew_true_v * (1-alpha)

            # todo : figure out how to set beta for scene graph classification
            skew_pos_th = 0.5 # default 0.9
            skew_neg_th = 0.1 # default 0.9
            ent_pos_w = 0.17 # default 0.05
            ent_neg_w = 0.17  # default 0.05
            if False:
                if skew_v > skew_th :
                    beta = 1.0 - ent_v * ent_pos_w
                elif skew_v < -skew_th :
                    beta = 1.0 - ent_v * ent_neg_w
                else:
                    beta = 0.0

                effect_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

            elif False:
                skew_pos_th = 0.9 # default 0.9
                skew_neg_th = 0.9 # default 0.9
                ent_pos_w = 0.17 # default 0.05
                ent_neg_w = 0.17  # default 0.05

                #Sample-ent
                pos_mask = (skew_v > skew_pos_th).astype(float)
                neg_mask = (skew_v < -skew_neg_th).astype(float)

                pos_beta = (1.0 - ent_v * ent_pos_w) * pos_mask
                neg_beta = (1.0 - ent_v * ent_neg_w) * neg_mask

                if False:
                    beta = pos_beta + neg_beta
                else:
                    beta = pos_beta

                effect_num = [1.0 - np.power(b, cls) for b, cls in zip(beta,cls_num_list[None,:].repeat(batch_size, 0))]
                per_cls_weights = (1.0 - beta[:,None]) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights,1)[:,None] * len(cls_num_list)

            elif True:
                skew_neg_th = skew_v.mean() - 0.7
                ent_neg_w = 0.16  # default 0.17

                neg_mask = (skew_v > skew_neg_th).astype(float)
                neg_beta = (1.0 - ent_v * ent_neg_w) * neg_mask

                beta = neg_beta
                effect_num = [1.0 - np.power(b, cls) for b, cls in zip(beta,cls_num_list[None,:].repeat(batch_size, 0))]
                per_cls_weights = (1.0 - beta[:,None]) / np.array(effect_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights,1)[:,None] * len(cls_num_list)


            rel_weight = torch.FloatTensor(per_cls_weights).cuda()

            rel_margin = None

        return  rel_weight, rel_margin

