# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from torch.autograd import Variable

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info

from .utils_relation import layer_init, seq_init

from .model_sgraph_with_context import SpectralMessage

from .model_sgraph_env import VGEnv

from torch.autograd import Variable, grad
import torch.optim as optim

import networkx as nx
import scipy.sparse as sp

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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight).half()

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input.float(), self.weight)
        output = torch.spmm(adj, support.float())
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj




class RLTransform(nn.Module):
    """
    Transform relational features into closes to ones closed to foregrounds
    """

    def __init__(self, cfg):
        super(RLTransform, self).__init__()

        # configure
        self.cfg = cfg

        # initialize agent
        self.agent = ActorCriticNNAgent(VGNet, df=0.1)
        # initialize environment
        self.env = VGEnv(type='train', seed=None)


        rl_fc = [
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ]
        self.rl_fc = nn.Sequential(*rl_fc)
        self.rl_fc.apply(seq_init)

        # feat_dim, h_dim1, h_dim2, dropout rate
        self.link_model = GCNModelVAE(200, 32, 16, 0.0)

    def loss_function(self, preds, labels, mu, logvar, n_nodes, norm,
                      pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(preds,
                                                         labels,
                                                         pos_weight=pos_weight)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        return cost + KLD


    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # return sparse_to_tuple(adj_normalized)
        return self.sparse_mx_to_torch_sparse_tensor(adj_normalized)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def nx_adj(self, pair, num_obj, device):

        # direct graph
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_obj))
        elist = [tuple(e) for e in pair.cpu().numpy().tolist()]
        graph.add_edges_from(elist)
        adj = nx.adjacency_matrix(graph)

        # normalization
        adj_norm = self.preprocess_graph(adj)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        pos_weight = torch.tensor(pos_weight).to(device)
        return adj_norm.to(device), adj_label.to(device), pos_weight, norm

    def graph(self, obj_embed,
              num_objs,
              num_rels,
              freq_bias,
              rel_pair_idxs=None,
              rel_labels=None, device=None):

        rel_mask = torch.zeros_like(rel_labels)
        rel_mask = rel_mask.split(num_rels)
        rel_labels = rel_labels.split(num_rels)

        link_loss = []
        for i in range(len(num_objs)):
            num_obj = num_objs[i]
            #adj = torch.zeros((num_obj, num_obj))
            rel_label = rel_labels[i]
            pair = rel_pair_idxs[i]

            bg_idx = np.where(rel_label.cpu() == 0)[0]
            fg_idx = np.where(rel_label.cpu() > 0)[0]
            #adj[pair[fg_idx,0], pair[fg_idx,1]] = 1

            adj_norm, adj_label, pos_weight, norm = self.nx_adj(
                pair[fg_idx], num_obj, device)

            recovered, mu, logvar = self.link_model(obj_embed[i], adj_norm)
            loss = self.loss_function(preds=recovered, labels=adj_label,
                                      mu=mu, logvar=logvar, n_nodes=num_obj,
                                      norm=norm, pos_weight=pos_weight)

            # find candidate foregrouds
            if True:
                for idx in fg_idx:
                    cand_idx = np.where(
                        (pair.cpu()[bg_idx,0] == pair.cpu()[idx,1]) &
                        (pair.cpu()[bg_idx,1] == pair.cpu()[idx,0]))[0]

                    if recovered[pair[idx,1], pair[idx, 0]] > 0.3 and len(cand_idx) > 0:
                        rel_mask[i][cand_idx] = 1

            else:
                for idx in bg_idx:
                    if recovered[pair[idx,0], pair[idx, 1]] > 0.9 :
                        rel_mask[i][idx] = 1


            link_loss.append(loss)

        link_loss = torch.stack(link_loss).mean() * 0.1
        return cat(rel_mask), link_loss

    def forward(self, obj_embed, union_features, rel_mean, rel_covar,
                freq_bias, geo_dists, rel_labels,num_objs, num_rels,
                rel_pair_idxs):

        # get device
        device = union_features.get_device()

        # sample classes
        # for inverse frequency
        #freq_bias = 1 - torch.sigmoid(freq_bias)
        freq_bias = freq_bias + geo_dists
        #freq_bias = torch.sigmoid(freq_bias + geo_dists)
        freq_bias = F.softmax(freq_bias, 1)

        rel_rt_loss = None
        rel_link_loss = None
        if self.training:

            fg_mask, rel_link_loss =self.graph(obj_embed, num_objs, num_rels, num_rels,
                                           rel_pair_idxs, rel_labels, device)

            # index of backgrounds / foregrounds
            bg_idx = np.where(rel_labels.cpu() == 0)[0]
            fg_idx = np.where(rel_labels.cpu() > 0)[0]

            #topk_idx = torch.multinomial(freq_bias, 1, replacement=True)
            topk_prob, topk_idx = freq_bias.topk(5)
            topk_prob_bg = torch.gather(freq_bias[bg_idx,] * fg_mask[bg_idx][:,None], 1, topk_idx[bg_idx,0][:,None])

            mask_bg = torch.bernoulli(topk_prob_bg)
            #rel_labels_bg = topk_idx[bg_idx,] * mask_bg
            rel_labels_bg = topk_idx[bg_idx,]

            rel_labels[bg_idx] = rel_labels_bg[:,0].long()

            tf_idx = np.where(mask_bg.cpu() > 0)[0]
            tf_idx = bg_idx[tf_idx]

            # RL-trasformation
            rewards = []
            observations = []

            # play out each episode
            X = union_features.clone().detach()
            Y = topk_idx.cpu()
            Y_prob = topk_prob.cpu()

            for ep in tf_idx:
                observation = self.env.reset(X[ep], Y[ep], Y_prob[ep])

                self.agent.new_episode()
                total_reward = 0

                done = False
                while not done:
                    # given env. and observation,agent take a action
                    action, value = self.agent.act(observation, device)
                    observation, label, reward, done, info = self.env.step(action, value, observation)
                    self.agent.store_reward(reward)

                    total_reward += reward

                rewards.append(total_reward)
                #transformation = Variable(torch.tensor(observation)).to(device)

                if label > 0:
                    union_features[ep,] = observation.clone().detach()
                rel_labels[ep] = torch.tensor(label).to(device).long()

            # adjust agent parameters based on played episodes
            if len(tf_idx) > 0:
                rel_rt_loss = self.agent.update(device)

        union_features = self.rl_fc(union_features)

        return union_features, rel_labels, rel_rt_loss, rel_link_loss


class VGNet(nn.Module):
    '''
    A CNN with ReLU activations and a three-headed output, two for the 
    actor and one for the critic
    y1 - action distribution
    y2 - critic's estimate of value

    Input shape:    (batch_size, D_in)
    Output shape:   (batch_size, 40), (batch_size, 1)
    '''

    def __init__(self):

        super(VGNet, self).__init__()

        self.out_dir = nn.Linear(4096, 3)
        self.out_digit = nn.Linear(4096, 51)
        self.out_critic = nn.Linear(4096, 1)

    def forward(self, x):

        pi1 = self.out_digit(x)
        pi1 = F.softmax(pi1, dim=-1)

        pi2 = self.out_dir(x)
        pi2 = F.softmax(pi2, dim=-1)

        # https://discuss.pytorch.org/t/batch-outer-product/4025
        y1 = torch.bmm(pi1.unsqueeze(2), pi2.unsqueeze(1))
        y1 = y1.view(-1, 51 * 3)

        y2 = self.out_critic(x)

        if not self.training:
            y2 = torch.sigmoid(y2)

        return y1, y2

class ActorCriticNNAgent(nn.Module):
    '''
    Neural-net agent that trains using the actor-critic algorithm. The critic
    is a value function that returns expected discounted reward given the
    state as input. We use advantage defined as

        A = r + g * V(s') - V(s)

    Notation:
        A - advantage
        V - value function
        r - current reward
        g - discount factor
        s - current state
        s' - next state
    '''

    def __init__(self, new_network, params=None, lr=1e-4, df=0.1, alpha=0.5):

        super(ActorCriticNNAgent, self).__init__()

        # model and parameters
        if params is not None:
            self.model = new_network(params)
        else:
            self.model = new_network()
        if isinstance(self.model, torch.nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.df = df # discount factor
        self.alpha = alpha # multiply critic updates by this factor

        # initialize replay history
        self.replay = []

        # if trainable is changed to false, the model won't be updated
        self.trainable = True

    def torch_to_numpy(self, tensor):
        return tensor.data.numpy()

    def numpy_to_torch(self, array):
        return torch.tensor(array).float()

    def obs_to_input(self, obs):
        # reshape to (1, 28, 28)
        return obs[np.newaxis, ...]

    def act(self, x, device, env=None, display=False):
        # feed observation as input to net to get distribution as output
        #x = self.obs_to_input(x)
        #x = self.numpy_to_torch(x)
        #eps = self.obs_to_input(eps)
        #eps = self.numpy_to_torch(eps)

        #x = Variable(x).cuda(device).requires_grad_(False)
        #eps = Variable(eps).cuda(device).requires_grad_(False)

        y1, y2 = self.model(x)

        pi = self.torch_to_numpy(y1.cpu()).flatten()
        v  = self.torch_to_numpy(y2.cpu()).squeeze()

        # sample action from distribution
        pi = pi / pi.sum()
        a = np.random.choice(np.arange(3*51), p=pi)

        # update current episode in replay with observation and chosen action
        if self.trainable:
            self.replay[-1]['observations'].append(x)
            #self.replay[-1]['eps'].append(eps)
            self.replay[-1]['actions'].append(a)

        return np.array(a), np.array(v)

    def new_episode(self):
        # start a new episode in replay
        self.replay.append({'observations': [], 'actions': [], 'rewards': []})

    def store_reward(self, r):
        # insert 0s for actions that received no reward; end with reward r
        episode = self.replay[-1]
        T_no_reward = len(episode['actions']) - len(episode['rewards']) - 1
        episode['rewards'] += [0.0] * T_no_reward + [r]

    def _calculate_discounted_rewards(self):
        # calculate and store discounted rewards per episode

        for episode in self.replay:

            R = episode['rewards']
            R_disc = []
            R_sum = 0
            for r in R[::-1]:
                R_sum = r + self.df * R_sum
                R_disc.insert(0, R_sum)

            episode['rewards_disc'] = R_disc

    def update(self, device):

        assert(self.trainable)

        episode_losses = torch.tensor(0.0).to(device)
        N = len(self.replay)
        self._calculate_discounted_rewards()

        for episode in self.replay:

            O = episode['observations']
            #E = episode['eps']
            A = episode['actions']
            R = self.numpy_to_torch(episode['rewards'])
            R_disc = self.numpy_to_torch(episode['rewards_disc'])
            T = len(R_disc)

            # forward pass, Y1 is pi(a | s), Y2 is V(s)
            X = torch.cat([o for o in O])
            #eps = torch.cat([e for e in E])
            Y1, Y2 = self.model(X)
            pi = Y1
            Vs_curr = Y2.view(-1)

            # log probabilities of selected actions
            log_prob = torch.log(pi[np.arange(T), A])

            # advantage of selected actions over expected reward given state
            zero = Variable(torch.tensor([0.])).cuda(device)
            Vs_next = torch.cat((Vs_curr[1:], zero))
            adv = R.to(device) + self.df * Vs_next - Vs_curr

            # ignore gradients so the critic isn't affected by actor loss
            adv = adv.detach()

            # actor loss is -1 * advantage-weighted sum of log likelihood
            # critic loss is the SE between values and discounted rewards
            actor_loss = -torch.dot(log_prob, adv)
            critic_loss = torch.sum((R_disc.to(device) - Vs_curr) ** 2)
            episode_losses += actor_loss + critic_loss * self.alpha

        # backward pass
        #self.optimizer.zero_grad()
        loss = episode_losses / N * 1e-2
        #loss.backward()
        #self.optimizer.step()

        # reset the replay history
        self.replay = []

        return loss

    def copy(self):

        # create a copy of this agent with frozen weights
        agent = ActorCriticNNAgent(VGNet)
        agent.model = copy.deepcopy(self.model)
        agent.trainable = False
        for param in agent.model.parameters():
            param.requires_grad = False

        return agent
