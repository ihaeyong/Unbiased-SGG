import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from maskrcnn_benchmark.data import get_dataset_statistics

from .model_sgraph_with_freq import FrequencyBias
from .utils_sgraph import to_onehot
from .utils_relation import layer_init, seq_init

from collections import Counter
from itertools import combinations, permutations, product


from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head


        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size() # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
          mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        else :
          mask = None
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, rel_input, non_pad_mask=None, slf_attn_mask=None):
        
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, rel_input, rel_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask.float()
            #enc_output = self.pos_ffn(enc_output)
            #enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn




class TransformerEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input_feats, rel_feats, num_objs, num_rels):
        """
        Args:
            input_feats [Tensor] (#total_box, d_model) : bounding box features of a batch
            num_objs [list of int] (bsz, ) : number of bounding box of each image
        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """

        original_input_feats = input_feats
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)


        rel_feats = rel_feats.split(num_rels, dim=0)
        rel_feats = nn.utils.rnn.pad_sequence(rel_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device

        obj_pad_len = max(num_objs)
        rel_pad_len = max(num_rels)

        #num_rels_ = torch.LongTensor(num_rels).to(device).unsqueeze(1).expand(-1, pad_len)

        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, rel_pad_len).unsqueeze(1).repeat(
            1,obj_pad_len, 1)
        num_rels_ = torch.LongTensor(num_rels).to(device).unsqueeze(1).expand(-1, rel_pad_len).unsqueeze(1).repeat(
            1,obj_pad_len, 1)

        #slf_attn_mask = torch.arange(max(num_objs),
        #                             device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(
        #                                 -1, pad_len, -1) # (bsz, pad_len, pad_len)

        obj_mask = torch.arange(obj_pad_len, device=device).unsqueeze(1).expand(-1, rel_pad_len).unsqueeze(0).repeat(
            bsz,1,1).lt(num_objs_)

        rel_mask = torch.arange(rel_pad_len, device=device).unsqueeze(0).expand(obj_pad_len, -1).unsqueeze(0).repeat(
            bsz,1,1).lt(num_rels_)


        mask = obj_mask * rel_mask
        slf_attn_mask = mask != True

        num_objs_non = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, obj_pad_len)

        non_pad_mask = torch.arange(obj_pad_len,
                                    device=device).to(device).view(1, -1).expand(
                                        bsz, -1).lt(num_objs_non).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, rel_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]

        return enc_output

















class SpectralMessage(nn.Module):

    def __init__(self, config, in_channels):
        super(SpectralMessage, self).__init__()

        self.obj_dim = in_channels
        self.adj_sim = False

        self.obj_comp = nn.Sequential(
            nn.Linear(self.obj_dim, self.obj_dim // 4, bias=False),
            nn.ReLU(inplace=True))

        self.obj_decomp = nn.Sequential(
            nn.Linear(self.obj_dim // 4, self.obj_dim, bias=False),
            nn.ReLU(inplace=True))

        self.ou1 = nn.Sequential(
            nn.Linear(self.obj_dim // 4, self.obj_dim // 4, bias=False),
            nn.ReLU(inplace=True))

        if self.adj_sim:
            self.ofc_u = nn.Sequential(
                nn.Linear(self.obj_dim // 2, self.obj_dim // 4, bias=False),
                nn.ReLU(inplace=True))
        else:
            self.K = 1
            self.ofc_u = nn.Sequential(
                nn.Linear(self.obj_dim // 4 * self.K, self.obj_dim // 4, bias=False),
                nn.ReLU(inplace=True))

        # adj. matrix (edges of graph)
        if True:
            self.adj_matrix = nn.Sequential(
                nn.Conv2d(51, 10, 3, stride=1, padding=1, bias=False),
                nn.Conv2d(10,  5, 3, stride=1, padding=1, bias=False),
                nn.Conv2d(5 ,  1, 1, stride=1, bias=False),
                nn.Sigmoid())

        else:
            self.adj_matrix = nn.Sequential(
                nn.Conv2d(51, 10, 1, stride=1, bias=False),
                nn.Conv2d(10,  1, 1, stride=1, bias=False),
                nn.Sigmoid())

        # initialize layers
        self.obj_comp.apply(seq_init)
        self.obj_decomp.apply(seq_init)
        self.ou1.apply(seq_init)
        self.ofc_u.apply(seq_init)
        self.adj_matrix.apply(seq_init)

    def cos_sim(self,x, y):
        '''
        Input: x is a bxNxd matrix
               y is an optional bxMxd matirx
        Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm
        between x[b,i,:] and y[b,j,:]
        i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        x_norm = x/x.norm(dim=1)[:,None]
        y_norm = y/y.norm(dim=1)[:,None]

        dist = torch.matmul(x_norm, y_norm.transpose(0,1))

        return torch.clamp(dist, -1.0, 1.0)

    def laplacian(self, A, N):
        '''
        A : adjacency matrix (edges of a graph)
        N : number of objects
        '''
        # ----- laplacian --------
        # we assume symmetric nomalized Laplacian
        A_hat = A + torch.eye(N).cuda(A.get_device())  # Add self-loops
        D_hat = torch.sum(A_hat, 1)  # Node degrees
        D_hat = (D_hat + 1e-5) ** (-0.5)  # D^-1/2

        # Rescaled normalized graph Laplacian with self-loops
        L_hat = D_hat.view(N, 1) * A_hat * D_hat.view(1, N)

        return L_hat

    def chebyshev(self, A, N, X):
        # K : # Maximum number of hops (filter size)
        D = torch.sum(A, 1)  # Node degrees
        D = (D + 1e-5) ** (-0.5)  # D^-1/2

        # Rescaled normalized graph Laplacian without self-loops
        L_hat = - D.view(N, 1) * A * D.view(1, N)

        # Node features, Average features of 1-hop neighbors
        X_cheb = [X, torch.mm(L_hat, X)]

        # Recursive computation of features projected onto the Chebyshev basis
        if self.K > 2:
            for k in range(2, self.K):
                X_cheb.append(2 * torch.mm(L_hat, X_cheb[k - 1]) - X_cheb[k - 2])

        # Input features in the Chebyshev basis: torch.Size([6, 2, 3])
        X_cheb = torch.stack(X_cheb, 2)

        return X_cheb.view(N, -1)

    def spect_graph(self, num_objs, obj_reps, obj_preds, freq_bias,
                    rel_pair_idxs=None, readout=False, rel_labels=None):

        ofl_l_list = []
        ofl_u_list = []
        adj_link_list = []
        for i in range(len(num_objs)):

            # --------relationship------------------
            device = obj_reps[i].get_device()
            obj_u1 = self.ou1(obj_reps[i])
            num_obj = num_objs[i]

            ### --- adj_{sim} ----------###
            if self.adj_sim:
                adj_sim = self.cos_sim(obj_u1, obj_u1)

                # includes identity nodes
                n_value, n_index = adj_sim.topk(2, dim=1)

                obj_u1 = obj_u1[n_index.data,:]
                obj_u1 = obj_u1.view(num_obj, -1)

            ### --- adj_{rel_dists} --- ###
            obj_l = obj_preds[i]
            obj_idx = [*range(num_obj)]

            prod_idx = torch.LongTensor(
                np.array(list(product(obj_idx, obj_idx)))).cuda(device)

            subj_pl = obj_l[prod_idx[:,0]]
            obj_pl = obj_l[prod_idx[:,1]]

            subj_obj_l = torch.stack((subj_pl, obj_pl), 1)
            rel_u1 = freq_bias.index_with_labels(subj_obj_l)

            adj_fg = rel_u1.view(num_obj, num_obj, -1)[:,:,:]
            adj_fg = self.adj_matrix(adj_fg.permute(2,0,1)[None,:])[0][0]

            # ----- laplacian --------
            if True:
                hat_lap = self.laplacian(adj_fg, num_obj)
                ofl_u = torch.matmul(hat_lap, obj_u1)
            else:
                ofl_u = self.chebyshev(adj_fg, num_obj, obj_u1)

            if readout:
                ofl_u_list.append(ofl_u + obj_l1.mean(0))
            else:
                ofl_u_list.append(ofl_u)

            if self.training:
                adj_gt = torch.zeros_like(adj_fg)
                adj_mask = torch.zeros_like(adj_fg)
                #rel_inv_dists = F.softmax(1./rel_u1, 1)
                rel_inv_dists = F.softmax(rel_u1, 1)
                for j in range(rel_labels[i].size(0)):
                    adj_gt[rel_pair_idxs[i][j,0].data, rel_pair_idxs[i][j,1].data] = 1.0 - rel_inv_dists[j,rel_labels[i][j]]
                    adj_mask[rel_pair_idxs[i][j,0].data, rel_pair_idxs[i][j,1].data] = 1.0

                link_loss = torch.abs(adj_fg-adj_gt) * adj_mask
                adj_link_list.append(link_loss.sum()/ adj_mask.sum())

        refine_u =torch.cat(ofl_u_list, 0)

        if self.training:
            link_loss = torch.stack(adj_link_list).mean()
        else:
            link_loss = None

        return refine_u, link_loss

    def forward(self, num_objs, obj_preds, encoder_reps, freq_bias, rel_pair_idxs=None,
                readout=False, rel_labels=None):
        """
        feature_obj : [feature(num_obj x R^{4096}); embed(R^{200}); position(R^{128})]
        rel_inds :    [rel_inds[:,1], rel_inds[:,2]]
        obj_preds :   [num_obj]
        freq_bias :   [num_obj x 51]
        """
        comp_objs = self.obj_comp(encoder_reps)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_reps = comp_objs.split(num_objs, dim=0)

        ofc_u, link_loss = self.spect_graph(num_objs, obj_reps, obj_preds, freq_bias,
                                            rel_pair_idxs, readout, rel_labels)

        u_m = self.ofc_u(ofc_u)

        decomp_obj = self.obj_decomp(u_m)
        encoder_reps = decomp_obj + encoder_reps

        return encoder_reps, link_loss
