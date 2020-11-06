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
            self.ofc_u = nn.Sequential(
                nn.Linear(self.obj_dim // 4, self.obj_dim // 4, bias=False),
                nn.ReLU(inplace=True))

        # adj. matrix
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        statistics = get_dataset_statistics(config)
        self.freq_bias = FrequencyBias(config, statistics)

        self.adj_matrix = nn.Sequential(
            nn.Conv2d(51, 10, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.Conv2d(10,  5, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.Conv2d(5 ,  1, 1, stride=1, bias=False),
            nn.Tanh())

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

    def spect_graph(self, num_objs, obj_reps, obj_preds,
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
            if False:
                rel_u1 = Variable(torch.FloatTensor(
                    self.freq_bias.rels_with_labels(subj_obj_l))).cuda(device)
            else:
                rel_u1 = self.freq_bias.index_with_labels(subj_obj_l)

            adj_fg = rel_u1.view(num_obj, num_obj, -1)[:,:,:]
            adj_fg = self.adj_matrix(adj_fg.permute(2,0,1)[None,:])[0][0]

            ofl_u = torch.matmul(adj_fg, obj_u1)

            if readout:
                ofl_u_list.append(ofl_u + obj_l1.mean(0))
            else:
                ofl_u_list.append(ofl_u)

            if self.training:
                adj_gt = Variable(
                    torch.zeros(num_obj, num_obj)).cuda(device)
                for j in range(rel_labels[i].size(0)):
                    if rel_labels[i][j].item() > 0:
                        adj_gt[rel_pair_idxs[i][j,0].data, rel_pair_idxs[i][j,1].data] = 1
                    else:
                        adj_gt[rel_pair_idxs[i][j,0].data, rel_pair_idxs[i][j,1].data] = 0.1

                deg = adj_gt.sum(1) + 1.0
                d_hat =  torch.zeros_like(adj_gt)
                n_obj = adj_gt.size(0)
                d_hat[range(n_obj), range(n_obj)] = (deg + 1e-5)**(-0.5)

                lap = torch.zeros_like(adj_gt)
                lap[range(n_obj), range(n_obj)] = 1.0

                # d_hat^T adj_gt
                d_hat_dot_A = torch.matmul(d_hat, adj_gt.transpose(0,1))
                dot_d_hat = torch.matmul(d_hat_dot_A, d_hat.transpose(0,1))
                lap = lap - dot_d_hat

                if False:
                    eig_val,eig_vec = torch.eig(lap,True)
                    img_indices = np.where(eig_val[:,1].cpu() != 0)[0]

                    true_eig_vec = eig_vec
                    if len(img_indices) > 2:
                        for j in range(len(img_indices),2):
                            true_eig_vec[j] = eig_vec[:,j] + eig_vec[:,j+1]
                            true_eig_vec[j+1] = eig_vec[:,j] - eig_vec[:,j+1]

                            eig_val_sorted, indices=torch.sort(eig_val,
                                                               dim=1,
                                                               descending=True)
                    eig_vec_sorted = true_eig_vec.gather(dim=1, index=indices)
                    eig_vec_sorted[-1] = 0

                eig_val, eig_vec = torch.symeig(lap, True, False)
                if num_obj > 2:
                    eig_vec[:,2:] = 0
                lap = eig_vec * lap

                link_loss = torch.abs(adj_fg-lap) * adj_gt
                adj_link_list.append(link_loss.sum() / adj_gt.sum())

        refine_u =torch.cat(ofl_u_list, 0)

        if self.training:
            link_loss = torch.stack(adj_link_list).mean()
        else:
            link_loss = None

        return refine_u, link_loss

    def forward(self, num_objs, obj_preds, encoder_reps, rel_pair_idxs=None,
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

        ofc_u, link_loss = self.spect_graph(num_objs, obj_reps, obj_preds,
                                            rel_pair_idxs, readout, rel_labels)

        u_m = self.ofc_u(ofc_u)

        decomp_obj = self.obj_decomp(u_m)
        encoder_reps = decomp_obj + encoder_reps

        return encoder_reps, link_loss
