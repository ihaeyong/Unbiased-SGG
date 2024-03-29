# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from .model_sgraph_with_rel_weight import RelWeight, ObjWeight, LDAMLoss, NBDTLoss
from .skew_class_weight import SkewBalWeight

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
            mode,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing

        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        self.mode = mode
        self.rel_type = 'sample-weight'
        self.obj_type = 'sample-weight'
        self.gamma = 0.02

        self.weight = 'batchweight'
        #self.weight = 'weight'

        cls_num_list = np.load('./datasets/vg/obj_freq.npy')

        self.obj_weight = ObjWeight(cls_num_list)
        if False:
            self.rel_weight = RelWeight(predicate_proportion, temp=1e0)
        else:
            self.rel_weight = SkewBalWeight()

        self.logSoftmax = nn.LogSoftmax(dim=1)

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:

            if self.obj_type is 'ldam':
                self.criterion_loss = LDAMLoss(max_m=0.001, weight=None, s=1.0)
            else:
                self.criterion_loss = nn.CrossEntropyLoss()

            if self.weight == 'reweight':
                self.criterion_rel_loss = nn.CrossEntropyLoss(self.pred_weight)
            else:
                self.criterion_rel_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits,
                 freq_bias):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if self.weight == 'batchweight':

            if False:
                rel_weight, rel_margin = self.rel_weight(relation_logits,
                                                         freq_bias,
                                                         rel_labels,
                                                         self.gamma)
            else:
                rel_weight = self.rel_weight(freq_bias, rel_labels)
                

            if self.rel_type == 'focal' :
                loss_relation = F.cross_entropy(relation_logits,
                                                rel_labels.long(),
                                                reduction='none',
                                                weight=rel_weight)
                loss_relation = self.focal_loss(loss_relation, self.gamma)

            elif self.rel_type == 'margin':
                loss_relation = F.cross_entropy(relation_logits - rel_margin,
                                                rel_labels.long(),
                                                rel_weight)

            elif self.rel_type == 'var_margin':
                loss_relation = F.cross_entropy((rel_margin - relation_logits)**2,
                                                rel_labels.long(),
                                                rel_weight)

            elif self.rel_type == 'weight':
                loss_relation = F.cross_entropy(relation_logits,
                                                rel_labels.long(),
                                                rel_weight)

            elif self.rel_type == 'sample-weight':

                # label mask
                index = torch.zeros_like(relation_logits, dtype=torch.uint8)
                index.scatter_(1, rel_labels.data.view(-1, 1), 1)
                target_weight = index.type(torch.cuda.FloatTensor) * rel_weight

                output = self.logSoftmax(relation_logits)
                loss = output * target_weight

                # weighted cross entropy
                loss_relation = -loss.sum() / target_weight.sum()

        else:
            loss_relation = self.criterion_rel_loss(relation_logits, rel_labels.long())


        if self.weight == 'batchweight' and self.mode != 'predcls':

            obj_weight, obj_margin = self.obj_weight(refine_obj_logits,
                                                     fg_labels,
                                                     self.gamma)

            if self.obj_type == 'margin':
                loss_refine_obj = F.cross_entropy(refine_obj_logits-obj_margin,
                                                  fg_labels.long(),
                                                  obj_weight)
            elif self.obj_type == 'weight':
                loss_refine_obj = F.cross_entropy(refine_obj_logits,
                                                  fg_labels.long(),
                                                  obj_weight)

            elif self.obj_type == 'ldam':
                loss_refine_obj = self.criterion_loss(refine_obj_logits,
                                                      fg_labels.long())

            elif self.obj_type == 'none':
                loss_refine_obj = F.cross_entropy(refine_obj_logits,
                                                  fg_labels.long())

            elif self.obj_type == 'sample-weight':

                # label mask
                index = torch.zeros_like(refine_obj_logits, dtype=torch.uint8)
                index.scatter_(1, fg_labels.data.view(-1, 1), 1)
                target_weight = index.type(torch.cuda.FloatTensor) * obj_weight

                output = self.logSoftmax(refine_obj_logits)
                loss = output * target_weight

                # weighted cross entropy
                loss_refine_obj = -loss.sum() / target_weight.sum()

        else:
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                             fg_bg_sample=self.attribute_sampling,
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def focal_loss(self, input_values, gamma):

        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):

        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class ReweightLoss(nn.Module):
    def __init__(self, num_class, samples_per_cls):
        super(ReweightLoss, self).__init__()
        self.num_class = num_class
        weights = 1.0 / np.array(samples_per_cls)
        weights = weights / np.sum(weights) * num_class
        self.weights = torch.tensor(weights).float()
    def forward(self, logits, labels):
        weights = self.weights.to(logits.device)
        rw_losses = F.cross_entropy(input=logits, target=labels, weight=weights)
        return rw_losses


class ClassBalancedLoss(nn.Module):
    def __init__(self, num_class, factor, samples_per_cls, loss_type="softmax", beta=0.999, gamma=0):
        super(ClassBalancedLoss, self).__init__()
        self.num_class = num_class
        self.factor = factor
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_samples = samples_per_cls
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_class
        self.weights = torch.tensor(weights).float()

    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                               torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    def CB_losss(self, labels, logits, loss_type, gamma):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        weights_ = self.weights.to(logits.device)

        labels_one_hot = F.one_hot(labels, self.num_class).float()

        weights = weights_.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_class)

        if loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif loss_type == "softmax":
            # pred = logits.softmax(dim = 1)
            # cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
            cb_loss = F.cross_entropy(input = logits, target = labels, weight = weights_)
        return cb_loss

    def forward(self, logits, labels):
        cb_losses = self.CB_losss(labels, logits, self.loss_type, self.gamma)
        return cb_losses * self.factor



def make_roi_relation_loss_evaluator(cfg):

    # mode
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        mode,
    )

    return loss_evaluator
