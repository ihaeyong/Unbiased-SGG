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

from .utils_relation import layer_init, seq_init

from .model_sgraph_with_context import SpectralMessage

from torch.autograd import Variable, grad

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

        self.obj_ctx = nn.Linear(self.obj_dim + self.embed_dim + 128,
                                 self.hidden_dim * 8)

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # relation context
        self.lin_obj_h = nn.Linear(self.hidden_dim * 8, self.hidden_dim // 2)
        self.out_obj = nn.Linear(self.hidden_dim * 8, len(self.obj_classes))

        # spectral message passing
        self.obj_ctx_layer = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        if self.obj_ctx_layer > 0:
            self.sg_msg = SpectralMessage(config, self.hidden_dim * 4)


        # initialize layers
        layer_init(self.obj_ctx, xavier=True)
        layer_init(self.lin_obj_h, xavier=True)
        layer_init(self.out_obj, xavier=True)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_obj_feat",
                                 torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_ctx_feat",
                                 torch.zeros(256))

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
            obj_inp_rep = obj_feats[perm].contiguous()
        else:
            obj_inp_rep = obj_feats

        # === [4096 + 200 + 128, hidden_dim * 2] ===
        encoder_rep = self.obj_ctx(obj_inp_rep)

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
        if (not self.training) and self.effect_analysis and ctx_average and False:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = encoder_rep

        if self.training and self.effect_analysis and False:
            self.untreated_dcd_feat = self.moving_average(
                self.untreated_dcd_feat, decode_inp)

        # Decode in order if True
        if order:
            decoder_inp = decoder_inp[inv_perm]

        # obj. predictions
        obj_dists, obj_preds = self.decoder(
            decoder_inp,
            obj_labels=obj_labels if obj_labels is not None else None,
            boxes_per_cls=boxes_per_cls if boxes_per_cls is not None else None,)

        encoder_rep = self.lin_obj_h(decoder_inp)

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed, decoder_inp

    def decoder(self, obj_fmap, obj_labels, boxes_per_cls):

        if self.mode == 'predcls' and not self.training:
            obj_dists2 = Variable(to_onehot(obj_labels.data, len(self.obj_classes)))
        else:
            obj_dists2 = self.out_obj(obj_fmap)

        # Do NMS here as a post-processing step
        if self.mode == 'sgdet' and not self.training and boxes_per_cls is not None:
            is_overlap = nms_overlaps(boxes_per_cls).view(
                boxes_per_cls.size(0), boxes_per_cls.size(0),
                boxes_per_cls.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(obj_dists2,1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = torch.zeros(obj_dists2.shape[0]).cuda(
                obj_dists2.get_device()).long()

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
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1) * obj_pre_rep
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists,obj_preds,obj_ctx,perm,inv_perm,ls_transposed, obj_reps = self.obj_stx(
            obj_pre_rep, proposals, freq_bias, rel_pair_idxs,
            obj_labels=obj_labels,
            rel_labels=rel_labels,
            boxes_per_cls=boxes_per_cls,
            ctx_average=ctx_average,
            order=False)

        # edge level contextual feature
        if False:
            updated_obj_embed = self.updated_obj_embed(obj_preds.long())
        else:
            updated_obj_embed = F.softmax(obj_dists, dim=1) @ self.updated_obj_embed.weight

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_ctx = self.untreated_ctx_feat.view(1, -1).expand(batch_size, -1) * obj_ctx
        else:
            obj_ctx = obj_ctx

        edge_ctx =  obj_ctx
        edge_obj = updated_obj_embed

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_ctx_feat = self.moving_average(self.untreated_ctx_feat, obj_ctx)

        return obj_dists, obj_preds, edge_ctx, edge_obj, obj_reps




class PostSpectralContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes):
        super(PostSpectralContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # object & relation context
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # relation context
        self.out_obj = nn.Linear(self.hidden_dim * 8, len(self.obj_classes))
        self.obj_ctx = nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8)
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 8)

        # initialize layers
        layer_init(self.out_obj, xavier=True)
        layer_init(self.obj_ctx, xavier=True)

        self.alpha = 0.02

    def decoder(self, obj_fmap, obj_labels, boxes_per_cls):

        if self.mode == 'predcls' and not self.training and False :
            obj_dists2 = Variable(to_onehot(obj_labels.data, len(self.obj_classes)))
        else:
            obj_dists2 = self.out_obj(obj_fmap) * self.alpha

        # Do NMS here as a post-processing step
        if self.mode == 'sgdet' and not self.training and boxes_per_cls is not None:
            is_overlap = nms_overlaps(boxes_per_cls).view(
                boxes_per_cls.size(0), boxes_per_cls.size(0),
                boxes_per_cls.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(obj_dists2,1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = torch.zeros(obj_dists2.shape[0]).cuda(
                obj_dists2.get_device()).long()

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

    def forward(self, x, proposals):

        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        assert proposals[0].mode == 'xyxy'

        batch_size = x.shape[0]
        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object contextual information
        x = self.obj_ctx(x)
        x = self.layer_norm(x)

        # obj. predictions
        obj_dists, obj_preds = self.decoder(
            x,
            obj_labels=obj_labels if obj_labels is not None else None,
            boxes_per_cls=boxes_per_cls if boxes_per_cls is not None else None,)

        return obj_dists




class RelTransform(nn.Module):
    """
    Transform relational features into closes to ones closed to foregrounds
    """

    def __init__(self, cfg):
        super(RelTransform, self).__init__()

        # configure
        self.cfg = cfg

        # predicate inverse proportion
        fg_rel = np.load('./datasets/vg/fg_matrix.npy')
        bg_rel = np.load('./datasets/vg/bg_matrix.npy')
        fg_rel[:,:,0] = bg_rel
        pred_freq = fg_rel.sum(0).sum(0)

        # pred inverse proportion
        pred_inv_prop = 1.0 / np.power(pred_freq, 1/2)
        max_m = 0.03
        self.pred_inv_prop = pred_inv_prop * (max_m / pred_inv_prop.max())

        self.mse_loss = nn.MSELoss() #nn.CrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.max_iter = 1
        self.rand_start = True
        self.rand_end = True
        self.step_size = 0.1

        # transfer
        enc_transf = [
            nn.Linear(4096, 4096 // 2, bias=True),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 2, 4096 // 4, bias=True),
            nn.BatchNorm1d(4096 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 4, 4096 // 8, bias=True),
            nn.BatchNorm1d(4096 // 8),
            nn.ReLU(inplace=True),
        ]

        dec_transf = [
            nn.Linear(4096 // 8, 4096 // 4, bias=True),
            nn.BatchNorm1d(4096 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 4, 4096 // 2, bias=True),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 2, 4096, bias=True),
        ]

        self.enc_transf = nn.Sequential(*enc_transf)
        self.dec_transf = nn.Sequential(*dec_transf)
        self.enc_transf.apply(seq_init)
        self.dec_transf.apply(seq_init)

        self.mean = nn.Linear(4096 // 8, 4096 // 8, bias=True)
        self.std = nn.Linear(4096 // 8, 4096 // 8, bias=True)
        layer_init(self.mean, xavier=True)
        layer_init(self.std, xavier=True)

        #self.rel_logits = nn.Linear(4096, 51, bias=True)
        #layer_init(self.rel_logits, xavier=True)

    def rand_perturb(self, inputs, attack, eps=0.5):

        if attack == 'inf':
            r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
        elif attack == 'randn':
            r_inputs = torch.randn_like(inputs) * inputs.max(1)[0][:,None]
        elif attack == 'l2':
            r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
        elif attack == 'zero':
            r_inputs = torch.zeros_like(inputs)

        return r_inputs

    def make_step(self, grad, attack='l2', step_size=0.1):

        if attack == 'l2':
            grad_norm = torch.norm(grad, dim=1).view(-1, 1)
            scaled_grad = grad / (grad_norm + 1e-10)
            step = step_size * scaled_grad

        elif attack == 'inf':
            step = step_size * torch.sign(grad)

        elif attack == 'none':
            step = step_size * grad

        return step

    def sample_normal(self, mean, logvar):
        # Using torch.normal(means,sds) returns a stochastic tensor which we cannot backpropogate through.
        # Instead we utilize the 'reparameterization trick'.
        # http://stats.stackexchange.com/a/205336
        # http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf
        sd = torch.exp(logvar*0.5)
        e = torch.randn_like(sd) # Sample from standard normal
        z = e.mul(sd).add_(mean)

        return z

    # Loss function
    def criterion(self, x_out, x_in, z_mu, z_logvar):

        mse_loss = self.mse_loss(x_out, x_in)
        kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        loss = (mse_loss + kld_loss) / x_out.size(0) # normalize by batch size

        return loss

    def forward(self, union_features, rel_mean, rel_covar, freq_bias, geo_dists, rel_labels):

        # for inverse frequency
        if True :
            freq_bias = 1 - torch.sigmoid(freq_bias)
        else:
            freq_bias = torch.sigmoid(freq_bias)

        freq_bias = freq_bias * torch.sigmoid(geo_dists)

        freq_bias = F.softmax(freq_bias, 1)
        rel_covar = F.softmax(rel_covar, 1)

        # index of backgrounds / foregrounds
        bg_idx = np.where(rel_labels.cpu() == 0)[0]
        fg_idx = np.where(rel_labels.cpu() > 0)[0]

        # set target relational labels
        if True:
            topk_prob, topk_idx = freq_bias.topk(1, largest=True)
            mask = topk_prob[bg_idx, 0] > 0.5
            rel_labels[bg_idx] = topk_idx[bg_idx,0] * mask.long()
        else:
            topk_idx = torch.multinomial(freq_bias, 1, replacement=True)

            prob = torch.gather(rel_covar[bg_idx,], 1, topk_idx[bg_idx,])

            mask = torch.bernoulli(prob)
            mask_rel_labels = topk_idx[bg_idx] * mask
            rel_labels[bg_idx] = mask_rel_labels[:,0].long()

        # relational dict
        rel_dict = rel_mean[rel_labels[bg_idx]].clone().detach().requires_grad_(True)
        rel_reps = union_features[bg_idx].clone().detach().requires_grad_(True)

        # transformation
        for _ in range(self.max_iter):
            # transformation of relational features
            in_reps = rel_reps
            enc_reps = self.enc_transf(in_reps)

            mean = self.mean(enc_reps)
            logvar = self.std(enc_reps)
            z = self.sample_normal(mean, logvar)

            out_reps = self.dec_transf(z)
            rel_reps = out_reps

            #rel_logits = self.rel_logits(rel_reps)

            # tranformation loss
            #loss = self.ce_loss(rel_logits, mask_rel_labels[:,0].long())
            loss = self.criterion(out_reps, in_reps, mean, logvar)
            grad, = torch.autograd.grad(loss, [z])

            # generate rel_reps
            z = self.sample_normal(mean, logvar)
            z = z - self.make_step(grad, 'l2', self.step_size)

            rel_reps = self.dec_transf(z)

        union_features[bg_idx] = rel_reps.half()

        return union_features, rel_labels
