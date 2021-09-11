"""Copyright (c) 5248's Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OOO and OOO projects which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2021, 5248's Aurthor
All rights reserved.
"""


"""
In this paper (5248), to alleviate the biased predicate predictions
caused by the long-tailed label distribution,
the Skew Class-balanced Re-weighting (SCR) loss function
is considered for the unbiased SGG models.

Leveraged by the skewness of biased predicate predictions,
firstly the SCR estimates the target predicate weight coefficient
and then re-weights more with respect to the biased predicates
for the better trading-off between the majority predicates and the minority ones.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.stats import entropy


class RelationLossComputation(nn.Module):
    """
    Step2: Computes the SKew Class-Balanced Re-Weighted loss for Unbiased SGG Models.
    """

    def __init__(self,):
        super(RelationLossComputation, self).__init__()

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def weighted_loss(self, relation_logits, rel_labels, rel_weight):

        # label mask
        index = torch.zeros_like(relation_logits, dtype=torch.uint8)
        index.scatter_(1, rel_labels.data.view(-1, 1), 1)
        target_weight = index.type(torch.cuda.FloatTensor) * rel_weight

        output = self.logSoftmax(relation_logits)
        loss = output * target_weight

        # weighted cross entropy
        loss_relation = -loss.sum() / target_weight.sum()

        return loss_relation

class SkewBalWeight(nn.Module):
    """
    Step1: leveraged by the skewness of biased predicate predictions,
    firstly the SCR estimates the target predicate weight coefficients.
    """

    def __init__(self):
        super(SkewBalWeight, self).__init__()

        # delta
        self.delta = 0.7

        # lambda_{skew}
        self.ent_neg_w = 0.06

    def softmax(self, x, temp=1.0):
        '''
        the softmax function
        '''
        x = x / temp
        max = np.max(x,axis=1,keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x,axis=1,keepdims=True)
        f_x = e_x / sum

        return f_x

    # Moment with optional pre-computed mean, equal to a.mean(axis, keepdims=True)
    def _moment(self, a, moment, axis, mean=None):
        '''
        the moment  measures comes from _moment function of scipy package
        '''
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
        of skewness, i.e. g_1=\frac{m_3}{m_2^{3/2}}.

        However, we use the target value of x instead of the mean value of x
        fairly for skew measures.
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

    def forward(self, freq_bias, rel_labels):
        '''
        Inputs :
        - skew_logits: freq_bias -- eq. (5)
        - ground-truth : rel_labels

        Outputs :
        - sample weight : rel_weight from eq. (7)
        '''

        # get mini-batch size
        batch_size, num_preds = freq_bias.size()

        # no gradient
        with torch.no_grad():
            # Entropy * scale
            batch_freq = freq_bias.data.cpu().numpy()

            # sample_estimates -- eq.(6)
            cls_num_list = batch_freq.sum(0)

            # target logit values
            cls_order_target = self.softmax(batch_freq)
            target = np.take_along_axis(cls_order_target, rel_labels.data.cpu().numpy()[:,None], axis=1)

            # skew measures --- eq.(9)
            skew_v = self._skew(cls_order_target, target, axis=1)

            # entropy for the uniformniss -- eq.(10)
            ent_v = entropy(cls_order_target, base=num_preds, axis=1)

            # the threshold s_th --- eq.(8)
            skew_neg_th = skew_v.mean() - self.delta

            neg_mask = (skew_v > skew_neg_th).astype(float)
            neg_beta = (1.0 - ent_v * self.ent_neg_w) * neg_mask

            # skew class-balanced effective number --- eq.(7) and weight
            beta = neg_beta
            effect_num = [1.0 - np.power(b, cls) for b, cls in zip(beta,cls_num_list[None,:].repeat(batch_size, 0))]
            per_cls_weights = (1.0 - beta[:,None]) / np.array(effect_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights,1)[:,None] * len(cls_num_list)

            # sample weights
            rel_weight = torch.FloatTensor(per_cls_weights).cuda()

        return rel_weight



if __name__ == "__main__":

    # instances for scr loss function
    scr_inst = SkewBalWeight()
    loss_inst = RelationLossComputation()

    '''
    to show how the scr loss function works,
    we assummed that the predicate biased logits
    are represented by its frequency M^{1/5}.

    M's predicate label list:
    'backgrounds', 'above', 'across', 'against', 'along', 'and',
    'at', 'attached to', 'behind', 'belonging to', 'between',
    'carrying', 'covered in', 'covering', 'eating', 'flying in',
    'for', 'from', 'growing on', 'hanging from', 'has', 'holding',
    'in', 'in front of', 'laying on', 'looking at', 'lying on',
    'made of', 'mounted on', 'near', 'of', 'on', 'on back of',
    'over', 'painted on', 'parked on', 'part of', 'playing',
    'riding', 'says', 'sitting on', 'standing on', 'to', 'under',
    'using', 'walking in', 'walking on', 'watching', 'wearing',
    'wears', 'with'
    '''

    rel_freq = np.array([3043905, 6712, 171, 208, 379, 504, 1829,
                         1413, 10011, 644, 394, 1603, 397, 460,
                         565, 4, 809, 163, 157, 663, 67144,
                         10764, 21748, 3167, 752, 676, 364, 114,
                         234, 15300, 31347, 109355, 333, 793, 151,
                         601, 429, 71, 4260, 44, 5086, 2273,
                         299, 3757, 551, 270, 1225, 352, 47326,
                         4810, 11059])

    freq_bias = np.power(rel_freq[None], 1/5)

    batch_size = 10
    freq_bias = torch.tensor(freq_bias).repeat(batch_size,1).cuda()
    freq_bias = freq_bias + torch.rand_like(freq_bias)

    # skew logits for sample estimates -- eq.(5)
    freq_bias = torch.sigmoid(freq_bias)


    '''
    we also assummed that the predicate logits are also given by its frequency.
    '''

    rel_logit = freq_bias + torch.rand_like(freq_bias)
    rel_label = torch.multinomial(freq_bias[0,:], batch_size).long()

    # compute the sample weights
    rel_weight = scr_inst(freq_bias, rel_label)
    print("rel_weight:{}".format(rel_weight))
    print("=s.t. {}".format(rel_weight.sum(1)))

    # compute the losses
    rel_loss = loss_inst.weighted_loss(rel_logit, rel_label, rel_weight)

    print("rel_loss:{}".format(rel_loss))




