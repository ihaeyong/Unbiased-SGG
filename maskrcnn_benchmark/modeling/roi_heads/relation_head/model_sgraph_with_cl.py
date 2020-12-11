import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        #return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))
        return torch.sum(- target * F.log_softmax(logits, -1), -1)

class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.norm = True

    def forward(self, anchor, positive, target):

        if self.norm:
            anchor = F.normalize(anchor, dim=1)
            positive = F.normalize(positive, dim=1)

        batch_size = anchor.size(0)
        label = target

        fg_idx = np.where(target.cpu() != 0)[0]
        bg_idx = np.where(target.cpu() == 0)[0]

        # target [batch_size, 1]
        target = target[:,None]

        # target [batch_size, batch_size] in {0, 1} and normalize in (0,1)
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        c_type = 'full'
        if c_type is 'fc':
            # modified by haeyong.k
            # only foreground contribution to contrastive loss
            # 1.larger variance of background
            # 2.unlabeled labels possible leads to increasing the number of negative samples
            loss_ce = cross_entropy(logit, target, False) * (label > 0).float()
            loss_ce = loss_ce.mean()
        elif c_type is 'wavg':
            loss_ce_fg = cross_entropy(logit, target, False) * (label > 0).float()
            loss_ce_bg = cross_entropy(logit, target, False) * (label == 0).float()

            loss_ce = (1-len(fg_idx)) * loss_ce_fg.mean() + loss_ce_bg.mean() * len(fg_idx)

        elif c_type is 'full':
            loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.get_device()

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0,1)).float().cuda(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # -----------for sg--------------
        anchor_feature = features[:,0,:]
        contrast_feature = features[:,1,:]

        anchor_count = 1
        contrast_count = 1

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.transpose(0,1)),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        #logits_mask = torch.scatter(
        #    torch.ones_like(mask),1,
        #    torch.arange(batch_size * anchor_count).view(-1, 1).cuda(device),
        #    0)
        #mask = mask * logits_mask
        logits_mask = mask

        # ------ sampling --------
        bg_idx = np.where(labels.data ==0 )[0]
        fg_idx = np.where(labels.data !=0 )[0]

        mask_bg = torch.zeros_like(mask)
        mask_bg_fg = torch.zeros_like(mask)
        mask_fg = torch.zeros_like(mask)
        mask_fg_bg = torch.zeros_like(mask)

        mask_bg[bg_idx, :] = mask[bg_idx, :]
        mask_bg_fg[bg_idx, :] = ((mask[bg_idx, :] - 1) < 0).float()

        mask_fg[fg_idx, :] = mask[fg_idx, :]
        mask_fg_bg[fg_idx, :] = ((mask[fg_idx, :] - 1) < 0).float()

        logits_bg_fg = torch.exp(logits) * mask_bg_fg
        logits_fg_bg = torch.exp(logits) * mask_fg_bg

        mask_bg_fg = (logits_bg_fg > 0.9).float()
        mask_fg_bg = mask_bg_fg.transpose(0,1)

        mask +=mask_fg_bg
        logits_mask = mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# usage
# define loss with a temperature `temp`
#criterion = SupConLoss(temperature=10)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
#features = torch.rand(10, 1, 20)
# labels: [bsz]
#labels = torch.randint(0,10,(10,))

# SupContrast
#loss_supcon = criterion(features, labels)
#print(loss_supcon)

# or SimCLR
#loss_sim = criterion(features)
#print(loss_sim)
