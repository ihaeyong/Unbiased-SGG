import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from .model_sgraph import FrequencyBias
#from .utils_sgraph import to_onehot

from maskrcnn_benchmark.data import get_dataset_statistics

from collections import Counter
from itertools import combinations, permutations, product

class ObjectMessage(nn.Module):

    def __init__(self, config, in_channels):
        super(ObjectMessage, self).__init__()

        self.obj_dim = in_channels
        self.adj_sim = True

        self.obj_comp = nn.Sequential(
            nn.Linear(self.obj_dim,
                      self.obj_dim // 4, bias=False),
            nn.ReLU(inplace=True))

        self.obj_decomp = nn.Sequential(
            nn.Linear(self.obj_dim // 4,
                      self.obj_dim, bias=False),
            nn.ReLU(inplace=True))

        self.ou1 = nn.Sequential(
            nn.Linear(self.obj_dim // 4,
                      self.obj_dim // 4, bias=False),
            nn.ReLU(inplace=True))

        if self.adj_sim:
            self.ofc_u = nn.Sequential(
                nn.Linear(self.obj_dim // 2,
                          self.obj_dim // 4, bias=False),
                nn.ReLU(inplace=True))
        else:
            self.ofc_u = nn.Sequential(
                nn.Linear(self.obj_dim // 4,
                          self.obj_dim // 4, bias=False),
                nn.ReLU(inplace=True))

        # adj. matrix
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        statistics = get_dataset_statistics(config)
        self.freq_bias = FrequencyBias(config, statistics)

        self.adj_matrix = nn.Sequential(
            nn.Conv2d(52, 10, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(10,  5, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(5 ,  1, 1, stride=1, bias=False),
            nn.Tanh())

    def rel_index(self, rel_inds):
        n_obj_img = Counter(rel_inds[:,0])
        k_imgs, v_imgs = zip(*n_obj_img.most_common())
        k_imgs = np.array(k_imgs).astype(np.int)

        start = 0
        end = 0
        start_list = []
        end_list = []

        num_imgs = len(k_imgs)
        for k in range(num_imgs):
            end += v_imgs[np.where(k_imgs==k)[0][0]]

            start_list.append(start)
            end_list.append(end)

            # bg rel included
            start += v_imgs[np.where(k_imgs==k)[0][0]]

        return np.stack(start_list,0), np.stack(end_list,0)

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

    def freq_graph(self, obj_feature, rm_obj_dists, num_r_objs, num_objs,
                   rel_inds, rois, readout=False, rel_labels=None):
        """ Create a graph """
        # subjects and objects
        subj = rel_inds[:,1]
        obj = rel_inds[:,2]

        # find index
        obj_s, obj_e = self.rel_index(rois.data)

        # define graph
        device = obj_feature.get_device()

        # object predictions
        obj_preds = rm_obj_dists.max(1)[1]
        obj_dists = Variable(to_onehot(obj_preds.data, 151))

        # m_rel_labels
        if self.training:
            m_rel_labels = Variable(
                torch.zeros(num_objs, num_objs)).cuda(device)
            for j in range(rel_labels.size(0)):
                if rel_labels[j,3].data.cpu().numpy()[0] > 0:
                    m_rel_labels[rel_labels[j,1].data, rel_labels[j,2].data] = 1
                else:
                    m_rel_labels[rel_labels[j,1].data, rel_labels[j,2].data] = 1

        ofl_l_list = []
        ofl_u_list = []
        adj_link_list = []
        for i in range(num_r_objs):
            o_sj = obj_s[i]
            o_ej = obj_e[i]

            # --------relationship------------------
            obj_feats = obj_feature[o_sj:o_ej]
            obj_u1 = self.ou1(obj_feats)

            num_objs = obj_u1.size(0)

            ### --- adj_{sim} ----------###
            if self.adj_sim:
                adj_sim = self.cos_sim(obj_u1, obj_u1)

                # includes identity nodes
                n_value, n_index = adj_sim.topk(2, dim=1)

                obj_u1 = obj_u1[n_index.data,:]
                obj_u1 = obj_u1.view(num_objs, -1)

            ### --- adj_{rel_dists} --- ###
            obj_l = obj_preds[o_sj:o_ej]
            obj_idx = [*range(num_objs)]

            b_obj_dists = obj_dists[o_sj:o_ej, :]
            obj_corr = self.cos_sim(b_obj_dists, b_obj_dists)
            obj_probs = (obj_corr > 0.99).sum(1).float() / len(obj_l)

            prod_idx = torch.LongTensor(
                np.array(list(product(obj_idx, obj_idx)))).cuda(device)

            subj_pl = obj_l[prod_idx[:,0]]
            obj_pl = obj_l[prod_idx[:,1]]

            subj_obj_l = torch.stack((subj_pl, obj_pl), 1)

            rel_u1 = Variable(torch.FloatTensor(
                self.freq_bias.rels_with_labels(subj_obj_l))).cuda(device)

            adj_fg = rel_u1.view(num_objs, num_objs, -1)[:,:,:]
            if False:
                adj_fg = self.adj_matrix(adj_fg.permute(2,0,1)[None,:])[0][0]
            else:
                adj_fg = torch.cat((adj_fg, adj_sim[:,:,None]), 2)
                adj_fg = self.adj_matrix(adj_fg.permute(2,0,1)[None,:])[0][0]

            # off-diag
            #adj_fg = adj_fg / adj_fg.sum(1)[:,None]

            if False:
                alpha_fg = adj_fg / obj_probs[:,None]
                adj_fg = 1/alpha_fg

            ofl_u = torch.matmul(adj_fg, obj_u1)

            if readout:
                ofl_u_list.append(ofl_u + obj_l1.mean(0))
            else:
                ofl_u_list.append(ofl_u)

            if self.training:
                adj_gt = m_rel_labels[o_sj:o_ej, o_sj:o_ej]

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

                eig_v = torch.eig(lap,True)[1]
                eig_v[-1] = 0
                lap = eig_v * lap

                link_loss = torch.abs(adj_fg-lap) * adj_gt
                adj_link_list.append(link_loss.sum() / adj_gt.sum())

        refine_u =torch.cat(ofl_u_list, 0)

        if self.training:
            link_loss = torch.cat(adj_link_list).mean()
        else:
            link_loss = None

        return refine_u, link_loss

    def forward(self, feature_obj, rm_obj_dists, rel_inds, rois, readout, rel_labels=None):
        """
        feature_obj : [feature(num_obj x R^{4096}); embed(R^{200}); position(R^{128})]
        rel_inds :    [rel_inds[:,1], rel_inds[:,2]]
        obj_preds :   [num_obj]
        freq_bias :   [num_obj x 51]
        """

        num_r_objs = rel_inds[:,0].max() + 1
        num_objs = feature_obj.size(0)

        comp_obj = self.obj_comp(feature_obj)

        ofc_u, link_loss = self.freq_graph(comp_obj, rm_obj_dists,
                                           num_r_objs, num_objs,
                                           rel_inds, rois, readout,
                                           rel_labels)

        u_m = self.ofc_u(ofc_u)

        decomp_obj = self.obj_decomp(u_m)
        out_feature = decomp_obj + feature_obj

        return out_feature, link_loss
