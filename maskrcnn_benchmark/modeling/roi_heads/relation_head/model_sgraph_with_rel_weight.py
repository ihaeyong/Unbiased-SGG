import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils_motifs import to_onehot

from scipy.stats import entropy, skew

class ObjWeight(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, temp=1):
        super(ObjWeight, self).__init__()

        self.temp = temp
        obj_proportion = [1.51000022e+02, 7.11200095e+03, 7.80100115e+03, 2.79550040e+04,
                          1.85960026e+04, 1.01870016e+04, 6.01200089e+03, 1.86200021e+04,
                          1.33890021e+04, 1.70560019e+04, 2.05290026e+04, 2.36060031e+04,
                          9.55500166e+03, 1.25180017e+04, 1.40580018e+04, 8.77800149e+03,
                          8.46100115e+03, 1.20450020e+04, 1.12250017e+04, 1.10760016e+04,
                          3.18050056e+04, 1.28030019e+04, 1.11959015e+05, 2.78410035e+04,
                          7.06400101e+03, 1.04280014e+04, 4.12650061e+04, 1.66920029e+04,
                          3.02280043e+04, 1.29380021e+04, 1.03660017e+04, 1.61690021e+04,
                          1.28790016e+04, 1.38490021e+04, 8.55500137e+03, 6.77600108e+03,
                          9.65100116e+03, 1.74610029e+04, 2.63330039e+04, 4.26000064e+03,
                          3.03930044e+04, 2.34830034e+04, 7.06100111e+03, 1.40520020e+04,
                          2.92530041e+04, 4.62200066e+04, 8.47000128e+03, 5.30800089e+03,
                          1.41710022e+04, 1.57900025e+04, 5.53400098e+03, 5.31400083e+03,
                          2.40130036e+04, 2.78330048e+04, 2.94240042e+04, 1.19410016e+04,
                          7.77100127e+03, 5.34240070e+04, 4.96760072e+04, 1.31340020e+04,
                          2.62800033e+04, 6.89200097e+04, 1.76050023e+04, 1.47900020e+04,
                          2.48630034e+04, 1.39580020e+04, 4.27880055e+04, 2.00700025e+04,
                          7.65700125e+03, 5.24900075e+03, 8.42400137e+03, 5.93600100e+03,
                          1.15990017e+04, 2.24800035e+04, 5.37790078e+04, 9.60800158e+03,
                          1.87770031e+04, 8.01900138e+03, 2.28002041e+05, 1.00240015e+04,
                          2.19920028e+04, 1.47330021e+04, 9.86100138e+03, 1.32520018e+04,
                          1.58450023e+04, 5.47600093e+03, 6.19400102e+03, 4.45780058e+04,
                          8.53500137e+03, 5.63000081e+03, 5.30390083e+04, 1.21247020e+05,
                          8.52600112e+03, 1.13770019e+04, 1.06810017e+04, 2.21680029e+04,
                          1.17900019e+04, 3.06030042e+04, 1.59080025e+04, 4.88220068e+04,
                          1.21800018e+04, 5.21500074e+03, 8.79300103e+03, 7.48300112e+03,
                          1.32110020e+04, 1.41860021e+04, 1.71110023e+04, 5.27600078e+03,
                          1.05970016e+04, 1.11640017e+04, 1.34470017e+04, 1.28930016e+05,
                          2.99080040e+04, 2.48040030e+04, 3.64680048e+04, 4.08530065e+04,
                          5.19400071e+03, 1.49530017e+04, 1.48370020e+04, 1.18100019e+04,
                          5.80900078e+03, 4.36440059e+04, 7.60800101e+03, 7.80200108e+03,
                          5.41930070e+04, 1.42140016e+04, 7.84770096e+04, 1.96950028e+04,
                          8.56000108e+03, 8.15600120e+03, 1.11700018e+04, 5.26400070e+03,
                          4.39800071e+03, 8.59900112e+03, 2.18470029e+04, 3.57830048e+04,
                          1.17560018e+05, 1.80720024e+04, 1.31570019e+04, 2.12340028e+04,
                          7.51500104e+03, 4.20600069e+03, 9.69500141e+03, 1.39160020e+04,
                          1.81910030e+04, 7.02390114e+04, 7.34700122e+03, 1.05250016e+04,
                          8.51100131e+03, 1.06669018e+05, 1.77680027e+04]

        self.obj_prop = np.array(obj_proportion)
        self.obj_idx = self.obj_prop.argsort()[::-1]

    def softmax_with_temp(self,z, T=1):

        z = np.array(z)
        z = z / T
        max_z = np.max(z)
        exp_z = np.exp(z-max_z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        return y

    def forward(self, obj_logits, freq_bias, obj_labels, gamma=0.01):

        batch_freq = freq_bias.sum(0).data.cpu().numpy()
        cls_num_list = batch_freq

        # target [batch_size, batch_size] in {0, 1} and normalize in (0,1)
        target = (obj_labels == torch.transpose(obj_labels[None,:], 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        target_mask = (to_onehot(obj_labels, 151,1) > 0.0).float()
        obj_margin = torch.matmul(target, obj_logits.detach())

        r_type = 'diff'
        if r_type is 'sigmoid':
            rel_margin = 1/torch.sigmoid(rel_margin) * target_mask * gamma
        elif r_type is 'inverse':
            rel_margin = 1/(torch.abs(rel_margin)+1) * target_mask * gamma
        elif r_type is 'diff' :
            obj_mask_logits = obj_logits.detach() * target_mask
            obj_margin = obj_margin * target_mask
            # mean - logits
            obj_diff = obj_margin - obj_mask_logits
            obj_diff_mask = (obj_diff < 0).float()
            obj_margin = obj_margin * obj_diff_mask * gamma

        # Entropy * scale
        cls_order = batch_freq[self.obj_idx]
        ent_v = entropy(cls_order, base=151)

        # skew_v > 0 : more weight in the left tail
        # skew_v < 0 : more weight in the right tail
        skew_v = skew(cls_order)
        if skew_v > 1.0 :
            beta = 1.0 - ent_v * 0.8
        else:
            beta = 0.0

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
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

        batch_freq = freq_bias.sum(0).data.cpu().numpy()
        cls_num_list = batch_freq

        if False:
            rel_margin = (to_onehot(rel_labels, len(self.pred_prop),1) > 0.0).float()
            #batch_label_freq = batch_label.sum(0).data.cpu().numpy()
            #log_batch_label_freq = np.log(1.0 + batch_label_freq)
            #cls_num_list = self.softmax_with_temp(log_batch_label_freq, self.temp)
            fg_idx = np.where(rel_labels.cpu() > 0)[0]
            bg_idx = np.where(rel_labels.cpu() == 0)[0]

            rel_margin[fg_idx, :] = rel_margin[fg_idx, :] * gamma
            rel_margin[bg_idx, :] = rel_margin[bg_idx, :] * gamma

        else:
            # target [batch_size, batch_size] in {0, 1} and normalize in (0,1)
            fg_idx = np.where(rel_labels.cpu() > 0)[0]
            bg_idx = np.where(rel_labels.cpu() == 0)[0]

            target = (rel_labels == torch.transpose(rel_labels[None,:], 0, 1)).float()
            target = target / torch.sum(target, dim=1, keepdim=True).float()

            target_mask = (to_onehot(rel_labels, len(self.pred_prop),1) > 0.0).float()

            l_type = 'none'
            if l_type is 'target_mask' :
                target_mask[bg_idx, :] = len(fg_idx) / (len(fg_idx) + len(bg_idx))
                target_mask[fg_idx, :] = len(bg_idx) / (len(fg_idx) + len(bg_idx))
            elif l_type is 'target':
                target[bg_idx, :] = len(fg_idx) / (len(fg_idx) + len(bg_idx))
                target[fg_idx, :] = len(bg_idx) / (len(fg_idx) + len(bg_idx))
            elif l_type is 'none':
                None

            rel_margin = torch.matmul(target, rel_logits.detach())

            r_type = 'diff'
            if r_type is 'sigmoid':
                rel_margin = 1/torch.sigmoid(rel_margin) * target_mask * gamma
            elif r_type is 'inverse':
                rel_margin = 1/(torch.abs(rel_margin)+1) * target_mask * gamma
            elif r_type is 'diff' :
                rel_mask_logits = rel_logits.detach() * target_mask
                rel_margin = rel_margin * target_mask
                # mean - logits
                rel_diff = rel_margin - rel_mask_logits
                rel_diff_mask = (rel_diff < 0).float()
                rel_margin = rel_margin * rel_diff_mask * gamma


        # Entropy * scale
        cls_order = batch_freq[self.pred_idx]
        ent_v = entropy(cls_order, base=51)

        # skew_v > 0 : more weight in the left tail
        # skew_v < 0 : more weight in the right tail
        skew_v = skew(cls_order)
        if skew_v > 1.0 :
            beta = 1.0 - ent_v * 0.7
        else:
            beta = 0.0

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        rel_weight = torch.FloatTensor(per_cls_weights).cuda()

        return  rel_weight, rel_margin


