# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext

from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics

from .model_sgraph import SpectralContext
from .model_sgraph_with_vratt import UnionRegionAttention
from .model_sgraph_with_cl import SupConLoss, NpairLoss

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor

from .model_sgraph_with_geom import Geometric

@registry.ROI_RELATION_PREDICTOR.register("SGraphPredictor")
class SGraphPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(SGraphPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.rel_ctx_layer = config.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)

        obj_classes = statistics['obj_classes']
        rel_classes = statistics['rel_classes']
        att_classes = statistics['att_classes']

        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)

        # init contextual object
        if self.attribute_on:
            self.context_layer = AttributeSpectralContext(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = SpectralContext(config, obj_classes, rel_classes, in_channels)

        # init contextual relation
        if self.rel_ctx_layer > 0:
            self.rel_sg_msg = UnionRegionAttention(obj_dim=256,
                                                   rib_scale=2,
                                                   power=1,
                                                   cfg=config)

        # box feature rois pooling
        in_channels = 256
        pool_all_levels = config.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        self.feature_extractor = make_roi_box_feature_extractor(config,
                                                                in_channels,
                                                                cat_all_levels=pool_all_levels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE

        self.vis_dists = nn.Linear(self.pooling_dim,
                                   self.num_rel_cls, bias=True)

        self.vis_ctx_dists = nn.Linear(self.hidden_dim,
                                       self.num_rel_cls, bias=True)

        self.non_vis_dists = nn.Linear(self.embed_dim * 2,
                                       self.num_rel_cls, bias=True)

        if False:
            self.vis_subj_dists = nn.Linear(self.pooling_dim + 256,
                                            self.num_obj_cls, bias=True)
            self.vis_obj_dists = nn.Linear(self.pooling_dim + 256,
                                           self.num_obj_cls, bias=True)
        else:
            self.vis_att_dists = nn.Linear(self.pooling_dim + 256,
                                           self.num_obj_cls, bias=True)


        # initialize layer parameters
        layer_init(self.vis_dists, xavier=True)
        layer_init(self.vis_ctx_dists, xavier=True)
        layer_init(self.non_vis_dists, xavier=True)
        if False:
            layer_init(self.vis_subj_dists, xavier=True)
            layer_init(self.vis_obj_dists, xavier=True)
        else:
            layer_init(self.vis_att_dists, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.hidden_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        self.geomtric = True
        if self.geomtric:
            geo_dim = 128
            self.geo_embed = Geometric(out_features=geo_dim)
            self.geo_dists = nn.Linear(geo_dim, self.num_rel_cls, bias=True)
            layer_init(self.geo_dists, xavier=True)

        self.post_cat_for_gate = False
        if self.post_cat_for_gate:
            self.post_cat = nn.Linear(256 * 2, self.pooling_dim)
            layer_init(self.post_cat, xavier=True)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features,
                union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # batch info
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(
                roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, obj_ctx_rep, obj_ctx_emb, link_loss = self.context_layer(
                roi_features, proposals, self.freq_bias, rel_pair_idxs, rel_labels, logger)

        obj_reps = obj_ctx_rep.split(num_objs, dim=0)
        obj_embs = obj_ctx_emb.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        prod_embs = []
        pair_preds = []
        for pair_idx, obj_rep, obj_emb, obj_pred in zip(rel_pair_idxs, obj_reps, obj_embs, obj_preds):
            prod_reps.append( torch.cat((obj_rep[pair_idx[:,0]],
                                         obj_rep[pair_idx[:,1]]), dim=-1) )
            prod_embs.append( torch.cat((obj_emb[pair_idx[:,0]],
                                         obj_emb[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]],
                                            obj_pred[pair_idx[:,1]]), dim=1) )

        # prod_rep [:,512], prod_emb [:,400], pair_rep[:,2]
        prod_rep = cat(prod_reps, dim=0)
        prod_emb = cat(prod_embs, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        # geometric embedding
        if self.geomtric:
            geo_embed = self.geo_embed(proposals, rel_pair_idxs)

        # relational message passing
        if self.rel_ctx_layer > 0:
            if self.training :
                rel_labels = torch.cat(rel_labels)
            else:
                rel_labels = None

            union_features = self.rel_sg_msg(
                union_features, prod_rep, prod_emb, geo_embed, rel_labels)

            # information bottlenecks
            iba_loss = self.rel_sg_msg.iba.buffer_capacity.mean() * 1e-2

        # rois pooling
        union_features = self.feature_extractor.forward_without_pool(union_features)

        # ========== update object logit message passing =============
        obj_per_dists = obj_dists.split(num_objs, dim=0)
        with torch.no_grad():
            u_features = union_features.clone().detach()
            u_subj, u_obj = prod_rep.clone().detach().split(256, dim=1)

        if False:
            subj_att_dists = self.vis_subj_dists(torch.cat((u_features, u_subj), dim=-1))
            obj_att_dists = self.vis_obj_dists(torch.cat((u_features, u_obj), dim=-1))
        else:
            subj_att_dists = self.vis_att_dists(torch.cat((u_features, u_subj), dim=-1))
            obj_att_dists = self.vis_att_dists(torch.cat((u_features, u_obj), dim=-1))

        subj_att_dists = subj_att_dists.split(num_rels, dim=0)
        obj_att_dists = obj_att_dists.split(num_rels, dim=0)

        u_type = 'avg_v1'
        u_obj_dists = []
        for logit, subj, obj, pair_idx in zip(obj_per_dists, subj_att_dists, obj_att_dists, rel_pair_idxs):

            M = logit.size(0)
            N = subj.size(0)
            if False:
                subj_mask = torch.zeros(M, N).cuda(logit.get_device())
                obj_mask = torch.zeros_like(subj_mask)
                subj_mask.scatter_(0,pair_idx[:,0][None],1)
                obj_mask.scatter_(0,pair_idx[:,1][None],1)
                subj_mask = subj_mask / subj_mask.sum(1, True)
                obj_mask = obj_mask / obj_mask.sum(1, True)
            else:
                subj_mask = torch.matmul(logit, subj.transpose(0,1))
                obj_mask = torch.matmul(logit, obj.transpose(0,1))

                subj_mask = F.softmax(subj_mask, 1)
                obj_mask = F.softmax(obj_mask, 1)

            mean_subj = torch.matmul(subj_mask, subj)
            mean_obj = torch.matmul(obj_mask, obj)

            if u_type == 'sum':
                logit = logit + mean_subj + mean_obj
            elif u_type == 'avg_v0':
                logit = (logit + mean_subj + mean_obj) / 3
            elif u_type == 'avg_v1':
                alpha = 0.4
                logit = logit + alpha * (mean_subj + mean_obj) / 2

            u_obj_dists.append(logit)

        obj_dists = torch.cat(u_obj_dists, dim=0)

        # use union box and mask convolution
        if self.post_cat_for_gate :
            ctx_gate = self.post_cat(prod_rep)
            union_features = ctx_gate * union_features

        # use frequence bias
        if self.use_bias:
            embed_bias = self.non_vis_dists(prod_emb)
            freq_bias = self.freq_bias.index_with_labels(pair_pred)

        # sum of non-vis/visual dists
        rel_dists, vis_dists, ctx_dists, freq_bias = self.rel_logits(union_features,
                                                                     prod_rep,
                                                                     geo_embed,
                                                                     embed_bias,
                                                                     freq_bias)
        rel_cl_loss = None

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if rel_cl_loss is not None:
            add_losses['rel_cl_loss'] = rel_cl_loss

        if link_loss is not None:
            add_losses['link_loss'] = link_loss

        if iba_loss is not None:
            add_losses['iba_loss'] = iba_loss

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses, freq_bias


    def rel_logits(self, vis_rep, ctx_rep, geo_emb, emb_dists, freq_dists):

        vis_dists = self.vis_dists(vis_rep)
        ctx_dists = self.vis_ctx_dists(ctx_rep)
        geo_dists = self.geo_dists(geo_emb)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + freq_dists + emb_dists + ctx_gate_dists)

        elif self.fusion_type == 'geo':
            #
            freq_bias = torch.sigmoid(freq_dists)
            union_dists = geo_dists

        elif self.fusion_type == 'freq_geo':
            #
            freq_bias = torch.sigmoid(freq_dists)
            union_dists = geo_dists + freq_dists

        elif self.fusion_type == 'sum':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_v1':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_v2':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + torch.sigmoid(emb_dists) + geo_dists

        elif self.fusion_type == 'sum_v3':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + torch.sigmoid(freq_dists) + emb_dists + geo_dists

        elif self.fusion_type == 'sum_v4':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + torch.sigmoid(ctx_dists) + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_v5':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = torch.sigmoid(vis_dists) + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_v6':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_v7':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_v8':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + freq_bias

        elif self.fusion_type == 'sum_v9':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists) + geo_dists

        elif self.fusion_type == 'sum_v10':
            # 18.9, 25.1, 27.7 // ( 2.0 // 3.2) // 51.5, 60.6, 63.2
            freq_bias = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = vis_dists + ctx_dists + torch.sigmoid(freq_dists) + emb_dists + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_softmax':
            # 11.6, 15.5, 17.3 ( 1.8, 2.2) 53.7, 60.5, 62.9
            freq_bias = torch.sigmoid(freq_dists + emb_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_softmax_v1':
            # 9.9, 15.1, 17.2 (7.6, 10.5) 47.3, 58.3, 62.1
            freq_bias = torch.sigmoid(freq_dists + emb_dists)
            union_dists = torch.sigmoid(vis_dists + ctx_dists) + torch.sigmoid(
                freq_dists + emb_dists + geo_dists)

        elif self.fusion_type == 'sum_softmax_v2':
            # 6.0, 8.4, 10.1 (7.0, 10.6) 44.5, 55.2, 59.9
            freq_bias = torch.sigmoid(freq_dists + emb_dists)
            union_dists = torch.sigmoid(vis_dists + ctx_dists) + torch.sigmoid(
                freq_dists) + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_softmax_v3':
            # 7.5, 10. 6, 12.7 ( 8.4, 11.4)  46.7, 56.2, 59.9
            freq_bias = torch.sigmoid(freq_dists + emb_dists)
            union_dists = torch.sigmoid(vis_dists) + torch.sigmoid(ctx_dists) + torch.sigmoid(
                freq_dists) + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_softmax_v4': # vs, sum_softmax
            # 16.3, 20.6, 22.2 (2.7, 4.2) 56.55, 63.6, 65.7
            freq_bias = torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_softmax_v41': # vs, sum_softmax
            # 13.6, 17.7, 19.3 (2.7, 4.2) 56.55, 63.6, 65.7
            freq_bias = torch.sigmoid(freq_dists) + torch.sigmoid(
                emb_dists) + torch.sigmoid(geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_softmax_v42': # vs, sum_softmax
            # 16.3, 20.6, 22.2 (2.7, 4.2) 56.55, 63.6, 65.7
            freq_bias = torch.sigmoid(freq_dists + emb_dists) + torch.sigmoid(geo_dists)
            union_dists = vis_dists + ctx_dists + freq_dists + emb_dists + geo_dists

        elif self.fusion_type == 'sum_softmax_v5': # vs, sum_softmax_v1
            # 
            freq_bias = torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists)
            union_dists = torch.sigmoid(vis_dists + ctx_dists) + torch.sigmoid(
                freq_dists + emb_dists + geo_dists)

        elif self.fusion_type == 'sum_softmax_v6': # vs, sum_softmax_v2
            # 6.9, 10.1, 12.5 (8.9, 12.5) 47.8, 58.4, 62.7
            freq_bias = torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists)
            union_dists = torch.sigmoid(vis_dists + ctx_dists) + torch.sigmoid(
                freq_dists) + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'sum_softmax_v7': # vs, sum_softmax_v3
            # 7.8, 10.6, 12.2  (9.2, 12.2) 49.4, 58.9, 62.6
            freq_bias = torch.sigmoid(freq_dists) + torch.sigmoid(emb_dists)
            union_dists = torch.sigmoid(vis_dists) + torch.sigmoid(ctx_dists) + torch.sigmoid(
                freq_dists) + torch.sigmoid(emb_dists) + torch.sigmoid(geo_dists)

        elif self.fusion_type == 'gate_sum':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            union_dists = alpha * (vis_dists + ctx_dists) + (1.0 - alpha) * (freq_dists + emb_dists)

        elif self.fusion_type == 'gate_geo_sum':
            alpha = torch.sigmoid(freq_dists + emb_dists + geo_dists)
            union_dists = alpha * (vis_dists + ctx_dists) + (1.0 - alpha) * (freq_dists + emb_dists + geo_dists)

        elif self.fusion_type == 'gate_geo_sum_v1':
            alpha = torch.sigmoid(freq_dists)
            union_dists = alpha * (vis_dists + ctx_dists) + (1.0 - alpha) * (freq_dists + emb_dists + geo_dists)

        elif self.fusion_type == 'gate_geo_sum_v2':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            union_dists = alpha * (vis_dists + ctx_dists) + (1.0 - alpha) * (freq_dists + emb_dists + geo_dists)

        elif self.fusion_type == 'gate_geo_sum_v3':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            beta = torch.sigmoid(geo_dists)
            union_dists = (alpha+beta)*(vis_dists+ctx_dists)+(1.0-alpha)*(freq_dists+emb_dists)+(1.0-beta)*geo_dists

        elif self.fusion_type == 'gate_geo_sum_v4':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            beta = torch.sigmoid(geo_dists)
            union_dists = (alpha*beta)*(vis_dists+ctx_dists)+(1.0-alpha)*(freq_dists+emb_dists)+(1.0-beta)*geo_dists

        elif self.fusion_type == 'gate_geo_sum_v5':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            beta = torch.sigmoid(geo_dists)
            union_dists = ctx_dists * (torch.sigmoid(vis_dists) + alpha + beta)

        elif self.fusion_type == 'gate_geo_sum_v6':
            alpha = torch.sigmoid(freq_dists + emb_dists)
            beta = torch.sigmoid(geo_dists)
            union_dists = vis_dists * (torch.sigmoid(ctx_dists) + alpha + beta)

        else:
            print('invalid fusion type')

        return union_dists, vis_dists, ctx_dists, freq_bias


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
