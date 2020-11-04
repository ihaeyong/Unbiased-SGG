# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from torch.autograd import Variable
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)

        self.matrix = pred_dist

        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs*self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def rels_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        labels = labels.data.cpu().numpy()
        return self.matrix[labels[:, 0],labels[:, 1]]

    def subj_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        labels = labels.data.cpu().numpy()
        return self.matrix[:,labels[:,1],labels[:,2]]

    def obj_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        labels = labels.data.cpu().numpy()
        return self.matrix[labels[:,0],:,labels[:,2]]


    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2] 
        :return: 
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:,:,0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:,:,1].contiguous().view(batch_size, 1, num_obj)

        return joint_prob.view(batch_size, num_obj*num_obj)  @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)
