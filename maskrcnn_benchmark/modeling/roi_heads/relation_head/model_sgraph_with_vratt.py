import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter

from .model_sgraph_with_iba import PerSampleBottleneck
from .utils_relation import layer_init, seq_init

torch.cuda.manual_seed(2007)

class UnionRegionAttention(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self,
                 obj_dim=256,
                 rib_scale = 1,
                 embedding=False,
                 geometric=False,
                 cfg=None):
        super(UnionRegionAttention, self).__init__()

        self.power = 1
        self.rib_scale = rib_scale
        self.embedding = embedding
        self.geometric = geometric
        self.ch = 1

        self.cfg = cfg

        subjobj_upconv = [
            nn.ConvTranspose2d(obj_dim * 2, 32, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ]

        if self.embedding:
            subjobj_emb_upconv = [
                nn.ConvTranspose2d(200 * 2, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 3, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 3, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            ]

        if self.geometric:
            subjobj_geo_upconv = [
                nn.ConvTranspose2d(128, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 3, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 3, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            ]

        self.subjobj_upconv = nn.Sequential(*subjobj_upconv)

        if self.embedding:
            self.subjobj_emb_upconv = nn.Sequential(*subjobj_emb_upconv)
            self.subjobj_emb_upconv.apply(seq_init)
            self.ch += 1

        if self.geometric:
            self.subjobj_geo_upconv = nn.Sequential(*subjobj_geo_upconv)
            self.subjobj_geo_upconv.apply(seq_init)
            self.ch += 1

        if self.rib_scale == 1:

            subjobj_mask = [
                nn.Conv2d(8*self.ch, 8, 1, stride=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, 1, stride=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                nn.Conv2d(3, 1, 1, stride=1, bias=False),
                nn.Sigmoid(),
            ]

            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
            ]
            union_downconv = [
                nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
            ]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            self.fmap_size = 7
            self.channel = 128
            self.sigma = 1

        if self.rib_scale == 2:
            subjobj_mask = [
                nn.ConvTranspose2d(8*self.ch, 8, 3, stride=2, padding=1, dilation=2,
                                   bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 3, 1, stride=1, dilation=1, bias=False),
                nn.BatchNorm2d(3),
                nn.Conv2d(3, 1, 1, stride=1, dilation=1, bias=False),
                nn.Sigmoid(),
            ]
            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=2,
                                   bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(64),
            ]
            union_downconv = [
                nn.Conv2d(64, 128, 3, stride=2, padding=1, dilation=2, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(256),
            ]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            self.fmap_size = 15
            self.channel = 64
            self.sigma = 2

        if self.rib_scale == 4:

            subjobj_mask = [
                nn.ConvTranspose2d(8 * self.ch, 8, 3, stride=2, padding=1,dilation=1,bias=False),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(3),
                nn.Conv2d(3,1,1,stride=1, bias=False),
                nn.Sigmoid()
            ]

            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,dilation=1,bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32)
            ]

            union_downconv = [
                nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256)
            ]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            self.fmap_size = 25
            self.channel = 32
            self.sigma = 3


        if self.rib_scale == 5:

            subjobj_mask = [
                nn.ConvTranspose2d(8*self.ch,8,3,stride=2,padding=1,dilation=2,bias=False),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8,3,3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(3),
                nn.Conv2d(3,1,1,stride=1, bias=False),
                nn.Sigmoid()
            ]

            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.ConvTranspose2d(256,128,3,stride=2,padding=1,dilation=2,bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128,64,3,padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64,32,3,stride=2,padding=1, bias=False),
                nn.BatchNorm2d(32)
            ]

            union_downconv = [
                nn.Conv2d(32, 64, 3, stride=2, padding=1,dilation=2,bias=False),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256)
            ]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            self.fmap_size = 29
            self.channel = 32
            self.sigma = 4

        self.g_type = 'iba'
        self.r_type = False

        if self.g_type is 'conv':
            g_conv = [nn.Conv2d(self.channel, self.channel, self.fmap_size,
                                stride=1, padding=self.fmap_size // 2),
                      nn.ReLU()]

            self.g_conv = nn.Sequential(*g_conv)
            self.g_conv.apply(seq_init)

        elif self.g_type is 'iba' or self.g_type is 'gcn_iba':
            self.iba = PerSampleBottleneck(sigma=self.sigma,
                                           fmap_size=self.fmap_size,
                                           channel=self.channel,
                                           pred_prop=cfg.MODEL.ROI_RELATION_HEAD.REL_PROP)

        # init weight
        self.subjobj_upconv.apply(seq_init)
        self.subjobj_mask.apply(seq_init)
        self.union_upconv.apply(seq_init)
        self.union_downconv.apply(seq_init)

    def normalized_adj(self, batch, A):

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(3)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(batch, 1, -1, 1) * A * D_hat.view(batch, 1, -1, 1)  # b, 1, N,N

        # Some additional trick I found to be useful
        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

        if self.power > 1:
            A_hat = A_hat**self.power

        return A_hat

    def global_conv(self, x):
        x = self.g_conv(x)
        #x = F.max_pool2d(x, 4)
        return x

    def get_vratt_masks(self):
        return self.vratt_masks

    def get_vratt_fmaps(self):
        return self.vratt_fmaps

    def forward(self, union_fmap, subjobj_fmap,
                subjobj_embed=None,
                geo_embed=None,
                rel_labels=None):
        """
        Input:
        union_fmap: batch x 256 x 7 x 7
        subjobj_fmap: batch x 512
        subjobj_embed : batch x 400
        Output:
        union_fmap: batch x 256 x 7 x 7
        """
        batch = union_fmap.size(0)

        subjobj_upconv = []
        subjobj_rep = self.subjobj_upconv(subjobj_fmap[:,:,None,None])
        subjobj_upconv.append(subjobj_rep)

        if self.embedding:
            subjobj_emb = self.subjobj_emb_upconv(subjobj_embed[:,:,None,None])
            subjobj_upconv.append(subjobj_emb)

        if self.geometric:
            subjobj_geo = self.subjobj_geo_upconv(geo_embed[:,:,None,None])
            subjobj_upconv.append(subjobj_geo)

        # -----union_upconv---------------------
        union_fmap = self.union_upconv(union_fmap)
        residual = union_fmap

        subjobj_upconv = torch.cat(subjobj_upconv, 1)
        mask = self.subjobj_mask(subjobj_upconv)

        # ----graph ----------------------------
        if self.g_type is 'gcn':
            # normalize adjacency matrix
            mask = self.normalized_adj(batch, mask)
            mask = mask.view(batch, 1, -1).expand(-1, self.fmap_size ** 2, -1) # b, N*N, N*N
            union_fmap = union_fmap.view(batch, self.channel, -1).permute(0,2,1) # b, N*N,128
            union_fmap = torch.bmm(mask, union_fmap).permute(0,2,1).view(
                batch, self.channel,
                self.fmap_size, self.fmap_size) # b,128,N,N
        elif self.g_type is 'conv':
            union_fmap = mask * union_fmap
            union_fmap = self.global_conv(union_fmap)

        elif self.g_type is 'skip':
            union_fmap = mask * union_fmap

        elif self.g_type is 'iba':
            union_fmap = self.iba(mask, union_fmap, rel_labels)

        elif self.g_type is 'gcn_iba':
            mask = self.normalized_adj(batch, mask)
            union_fmap = self.iba(mask, union_fmap)

        if self.r_type:
            union_fmap = residual + union_fmap # b,128,N,N


        if not self.training and self.cfg.RIB_FMAP_SAVE:
            self.vratt_masks = mask
            self.vratt_fmaps = union_fmap

        # -----union_downconv------------------
        union_fmap = self.union_downconv(union_fmap.contiguous())

        return union_fmap
