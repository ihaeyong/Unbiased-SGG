import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter

#from lib.iba import PerSampleBottleneck
from .utils_relation import layer_init

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor

class UnionRegionAttention(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, obj_dim=256,union_features=512, cfg=None):
        super(UnionRegionAttention, self).__init__()

        self.rib_scale = 1
        self.rel_iba = False

        self.cfg = cfg.clone()
        in_channels = obj_dim
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)

        subjobj_upconv = [
            nn.ConvTranspose2d(obj_dim * 2, 128, 3, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),]

        subjobj_emb_upconv = [
            nn.ConvTranspose2d(200 * 2,128, 3, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),]

        self.subjobj_upconv = nn.Sequential(*subjobj_upconv)
        self.subjobj_emb_upconv = nn.Sequential(*subjobj_emb_upconv)

        if self.rib_scale == 1:

            subjobj_mask = [
                nn.Conv2d(32*2, 1, 1, stride=1, bias=False),
                nn.Sigmoid(),
            ]

            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),]
            union_downconv = [
                nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            fmap_size = 7
            channel = 256

        if self.rib_scale == 2:
            subjobj_mask = [
                nn.ConvTranspose2d(32*2, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,1,1,stride=1, bias=False),
                nn.Sigmoid(),]
            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),]
            union_downconv = [
                nn.Conv2d(128, 512, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            fmap_size = 13
            channel = 128

        if self.rib_scale == 4:

            subjobj_mask = [
                nn.ConvTranspose2d(32*2, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8,1,1,stride=1, bias=False),
                nn.Sigmoid()]
            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32)]

            union_downconv = [
                nn.Conv2d(32, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 512, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512)]

            self.union_upconv = nn.Sequential(*union_upconv)
            self.union_downconv = nn.Sequential(*union_downconv)

            fmap_size = 25
            channel = 32

        if self.rel_iba:
            self.iba = PerSampleBottleneck(sigma=1.0,
                                           fmap_size=fmap_size,
                                           channel=channel)

        #init
        if False:
            layer_init(self.subjobj_upconv, xavier=True)
            layer_init(self.geo_upconv, xavier=True)
            layer_init(self.union_upconv, xavier=True)
            layer_init(self.union_downconv, xavier=True)


    def normalized_adj(self, batch, A):

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(3)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(batch, 1, -1, 1) * A * D_hat.view(batch, 1, -1, 1)  # b, 1, N,N

        # Some additional trick I found to be useful
        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

        return A_hat

    def forward(self, union_fmap, subjobj_fmap, subjobj_embed):
        """
        union_fmap: 512 x 7 x 7
        sub_rep: subject representation
        obj_rep: obj reprsentation
        sub_cat_emb: embedding of subject catergory
        obj_cat_emb: embedding of object category
        geo_emb: spatial embeddign of object pairs
        """
        batch = union_fmap.size(0)
        subjobj_upconv = self.subjobj_upconv(subjobj_fmap[:,:,None,None])
        subjobj_emb_upconv = self.subjobj_emb_upconv(subjobj_embed[:,:,None,None])

        # -----union_upconv---------------------
        union_fmap = self.union_upconv(union_fmap)
        residual = union_fmap

        subjobj_upconv = torch.cat((subjobj_upconv, subjobj_emb_upconv), 1)
        mask = self.subjobj_mask(subjobj_upconv)

        # normalize adjacency matrix
        mask = self.normalized_adj(batch, mask)

        # ----graph ----------------------------
        mask = mask.view(batch, 1, -1).expand(-1, 7 * 7, -1) # b, N*N, N*N
        union_fmap = union_fmap.view(batch, 128, -1).permute(0,2,1) # b, N*N,128
        union_fmap = torch.bmm(mask, union_fmap).permute(0,2,1).view(batch, 128, 7, 7) # b, 128,N,N
        union_fmap = residual + union_fmap # b,128,N,N

        # -----union_downconv------------------
        union_fmap = self.union_downconv(union_fmap.contiguous())
        union_fmap = self.feature_extractor.forward_without_pool(union_fmap) # (total_num_rel, out_channels)

        return union_fmap
