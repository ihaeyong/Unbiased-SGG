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

from .utils_relation import layer_init

from .model_sgraph_with_context import SpectralMessage
from torch.distributions import Normal

from math import pi, sqrt, exp

def gauss(n=7,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


class SpectralContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(SpectralContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.updated_obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.updated_obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        #self.obj_ctx = nn.Linear(self.obj_dim, self.obj_dim)
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # relation context
        self.lin_obj_h = nn.Linear(self.hidden_dim * 8, self.hidden_dim // 2)
        self.out_obj = nn.Linear(self.hidden_dim * 8, len(self.obj_classes))

        # spectral message passing
        self.obj_ctx_layer = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        if self.obj_ctx_layer > 0:
            self.sg_msg = SpectralMessage(config, self.hidden_dim * 8)


        # initialize layers
        #layer_init(self.obj_ctx, xavier=True)
        layer_init(self.lin_obj_h, xavier=True)
        layer_init(self.out_obj, xavier=True)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
                self.obj_ctx_mean = nn.Linear(self.obj_dim + self.embed_dim + 128,
                                              self.obj_dim)

                self.obj_ctx_std = nn.Linear(self.obj_dim + self.embed_dim + 128,
                                             self.obj_dim)

                obj_lamb = [nn.Linear(self.embed_dim + 128,
                                      self.hidden_dim * 8),
                            nn.Sigmoid(),]

                self.obj_lamb = nn.Sequential(*obj_lamb)

                kernel_size = 5
                kernel_1d = torch.FloatTensor(gauss(n=kernel_size, sigma=1))
                self.smooth = nn.Conv1d(1, 1, kernel_size, stride=1)
                self.smooth.weight.data.copy_(kernel_1d)
                self.smooth.weight.requires_grad = False
                self.pad = nn.ReflectionPad1d(int((kernel_size - 1) / 2))

                layer_init(self.obj_ctx_mean)
                layer_init(self.obj_ctx_std)

                self.beta = 10.0 / (self.hidden_dim * 8)
                self.buffer_capacity = None

    def _sample_z(self, mu, log_noise_var):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + Variable(noise_std.data * eps)

    def _calc_capacity(self, mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    def sort_rois(self, proposals):
        c_x = center_x(proposals)

        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_stx(self,
                obj_feats,
                proposals,
                freq_bias,
                rel_pair_idxs,
                obj_labels=None,
                rel_labels=None,
                boxes_per_cls=None,
                ctx_average=False,
                order=False):
        """
        Object context and object classification, modified by haeyong.k
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)

        # Pass object features, sorted by score, into the encoder
        if order:
            encoder_rep = obj_feats[perm].contiguous()
        else:
            encoder_rep = obj_feats

        # untreated decoder input
        batch_size = encoder_rep.shape[0]

        # --- object predictions and spectral message passing ---
        if obj_labels is None:
            obj_dists = self.out_obj(encoder_rep)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels

        num_objs = [len(b) for b in proposals]
        link_loss = None
        for i in range(self.obj_ctx_layer):
            encoder_rep, link_loss = self.sg_msg(num_objs,
                                                 obj_preds,
                                                 encoder_rep,
                                                 freq_bias,
                                                 rel_pair_idxs,
                                                 readout=(i==3),
                                                 rel_labels=rel_labels)

        # --- object predictions and spectral message passing ---
        # Decode in order if True
        if order:
            encoder_inp = encoder_rep[inv_perm]

        # obj. predictions
        obj_dists, obj_preds = self.decoder(
            encoder_rep,
            obj_labels=obj_labels if obj_labels is not None else None,
            boxes_per_cls=boxes_per_cls if boxes_per_cls is not None else None,)

        encoder_rep = self.lin_obj_h(encoder_rep)

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed, link_loss

    def decoder(self, obj_fmap, obj_labels, boxes_per_cls):

        if self.mode == 'predcls' and not self.training:
            obj_dists2 = Variable(to_onehot(obj_labels.data, len(self.obj_classes)))
        else:
            obj_dists2 = self.out_obj(obj_fmap)

        # Do NMS here as a post-processing step
        if self.mode == 'sgdet' and not self.training and boxes_per_cls is not None:
            is_overlap = nms_overlaps(boxes_per_cls).view(
                boxes_for_nms.size(0), boxes_per_cls.size(0),
                boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(out_dists2,1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments[0].new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(
                    out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds = out_commitments

        elif self.mode == 'sgcls':
            obj_preds = (
                obj_labels
                if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1)
        else:
            obj_preds = (
                obj_labels
                if not self.training else obj_dists2[:, 1:].max(1)[1] + 1)

        return obj_dists2, obj_preds

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, freq_bias, rel_pair_idxs, rel_labels=None,
                logger=None, all_average=False, ctx_average=False):

        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if self.effect_analysis :
            x = cat((x, obj_embed, pos_embed), -1)
            lamb = self.obj_lamb(cat((obj_embed, pos_embed), -1))

            # smoothing
            lamb = self.pad(lamb[:,None])
            lamb = self.smooth(lamb).squeeze(1)

            # normal dists
            mean = self.obj_ctx_mean(x)
            std = F.softplus(self.obj_ctx_std(x))

            if False:
                norm = (x - mean) / std
            else:
                obj_prob = Normal(mean, std)
                norm = obj_prob.sample()

            # scaled signal
            scaled_signal = norm * lamb

            # Get sampling parameters
            eps = 1e-8
            noise_var = (1 - lamb + eps)**2
            noise_log_var = torch.log(noise_var)

            # Sample new output values from p(z|r)
            z_norm = self._sample_z(scaled_signal, noise_log_var)

            # Denormalize z to match magnitude of r
            if False:
                obj_pre_rep = torch.clamp(z_norm * std + mean, 0.0)
            else:
                obj_pre_rep = z_norm * std + mean

            self.buffer_capacity = self._calc_capacity(scaled_signal,
                                                       noise_log_var) * self.beta

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists,obj_preds,obj_ctx,perm,inv_perm,ls_transposed,link_loss=self.obj_stx(
            obj_pre_rep, proposals, freq_bias, rel_pair_idxs,
            obj_labels, rel_labels, boxes_per_cls, ctx_average=ctx_average)

        # edge level contextual feature
        if False:
            updated_obj_embed = self.updated_obj_embed(obj_preds.long())
        else:
            updated_obj_embed = F.softmax(obj_dists, dim=1) @ self.updated_obj_embed.weight

        if (all_average or ctx_average) and self.effect_analysis and (not self.training) and False:
            obj_ctx = self.untreated_ctx_feat.view(1, -1).expand(batch_size, -1) * obj_ctx
        else:
            obj_ctx = obj_ctx

        edge_ctx =  obj_ctx
        edge_obj = updated_obj_embed

        # memorize average feature
        if self.training and self.effect_analysis and False:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_ctx_feat = self.moving_average(self.untreated_ctx_feat, obj_ctx)

        return obj_dists, obj_preds, edge_ctx, edge_obj, link_loss

