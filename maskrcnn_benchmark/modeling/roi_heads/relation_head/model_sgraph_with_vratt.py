import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from scipy.linalg import qr

from lib.iba import PerSampleBottleneck
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.sparse_targets import FrequencyBias

class UnionRegionAttention(nn.Module):
    """
    None dim means that not to use that sub module.
    if None of geo, cat and appr was specified, only upconvolution is used.
    """

    def __init__(self, obj_dim=256,union_features=512, conf=None):
        super(UnionRegionAttention, self).__init__()

        self.rib_scale = conf.rib_scale

        subjobj_upconv = [
            nn.ConvTranspose2d(obj_dim * 2, 128, 3, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),]

        geo_upconv = [
            nn.ConvTranspose2d(256, 128, 3, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),]

        self.subjobj_upconv = nn.Sequential(*subjobj_upconv)
        self.geo_upconv = nn.Sequential(*geo_upconv)

        if self.rib_scale == 1:

            subjobj_mask = [
                nn.Conv2d(32*2, 1, 1, stride=1, bias=False),
                nn.Sigmoid()]

            self.subjobj_mask = nn.Sequential(*subjobj_mask)

            union_upconv = [
                nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),]
            union_downconv = [
                nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),]

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

        self.rel_iba = conf.rel_iba

        if self.rel_iba:
            self.iba = PerSampleBottleneck(sigma=1.0,
                                           fmap_size=fmap_size,
                                           channel=channel)

    def forward(self, union_fmap, subj_fmap, obj_fmap, geo_embed):
        """
        union_fmap: 512 x 7 x 7
        sub_rep: subject representation
        obj_rep: obj reprsentation
        sub_cat_emb: embedding of subject catergory
        obj_cat_emb: embedding of object category
        geo_emb: spatial embeddign of object pairs
        """

        subjobj = torch.cat((subj_fmap, obj_fmap), 1)
        subjobj_upconv = self.subjobj_upconv(subjobj[:,:,None,None])

        geo_upconv = self.geo_upconv(geo_embed[:,:,None,None])

        # -----union_upconv---------------------
        union_fmap = self.union_upconv(union_fmap)
        subjobj_upconv = torch.cat((subjobj_upconv, geo_upconv), 1)

        mask = self.subjobj_mask(subjobj_upconv)
        mask_fmap = union_fmap * mask

        if self.rel_iba:
            union_fmap = self.iba(mask, union_fmap)
            if True:
                union_fmap = union_fmap + self.iba(mask, union_fmap)
        else:
            union_fmap = union_fmap + union_fmap * mask

        # -----union_downconv------------------
        union_fmap = self.union_downconv(union_fmap)

        return union_fmap, mask_fmap
