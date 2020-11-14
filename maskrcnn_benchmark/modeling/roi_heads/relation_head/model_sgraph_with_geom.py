import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from scipy.linalg import qr

class Geometric(nn.Module):
    """
    Geometric Embedding
    """

    def __init__(self, out_features=128, bias=False):
        super(Geometric, self).__init__()
        self.geo_features = 9
        self.out_features = out_features
        self.geo_embedding = nn.Sequential(*[
            nn.Linear(self.geo_features,
                      self.out_features,
                      bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        ])

    def forward(self, proposals, rel_pair_idxs):
        subj_boxes = []
        obj_boxes = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            subj_boxes.append(head_proposal.bbox)
            obj_boxes.append(head_proposal.bbox)

        sub_boxes = torch.cat(subj_boxes)
        obj_boxes = torch.cat(obj_boxes)

        sub_widths = sub_boxes[:, 2] - sub_boxes[:, 0] + 1.0
        sub_heights = sub_boxes[:, 3] - sub_boxes[:, 1] + 1.0
        obj_widths = obj_boxes[:, 2] - obj_boxes[:, 0] + 1.0
        obj_heights = obj_boxes[:, 3] - obj_boxes[:, 1] + 1.0

        # angles
        subj_c_x = (sub_boxes[:, 2] + sub_boxes[:, 0]) / 2
        subj_c_y = (sub_boxes[:, 3] + sub_boxes[:, 1]) / 2

        obj_c_x = (obj_boxes[:, 2] + obj_boxes[:, 0]) / 2
        obj_c_y = (obj_boxes[:, 3] + obj_boxes[:, 1]) / 2

        delta_c_x = (subj_c_x - obj_c_x).float()
        delta_c_y = (subj_c_y - obj_c_y).float()
        angles = torch.atan2(delta_c_y, delta_c_x)

        geo = Variable(
            torch.zeros(sub_boxes.size(0), self.geo_features)
        ).cuda(sub_boxes.get_device()).detach()

        geo[:, 0] = (sub_boxes[:, 0] - obj_boxes[:, 0]) / sub_widths
        geo[:, 1] = (sub_boxes[:, 1] - obj_boxes[:, 1]) / sub_heights
        geo[:, 2] = (sub_boxes[:, 2] - obj_boxes[:, 2]) / sub_widths
        geo[:, 3] = (sub_boxes[:, 3] - obj_boxes[:, 3]) / sub_heights

        geo[:, 4] = obj_heights / sub_heights
        geo[:, 5] = obj_widths / sub_widths
        geo[:, 6] = (obj_heights * obj_widths) / (sub_heights * sub_widths)
        geo[:, 7] = (obj_heights + obj_widths) / (sub_heights + sub_widths)
        geo[:, 8] = angles / np.pi

        embedded_geo = self.geo_embedding(geo)

        return embedded_geo




class BoxesOfURA(nn.Module):
    """
    boxes of URA
    """

    def __init__(self, f_size, img_size):
        super(BoxesOfURA, self).__init__()

        self.f_size = f_size
        self.img_size = img_size

    def forward(self, rois, rel_inds):
        subj_boxes = rois[rel_inds[:, 1], :][:, 1:]
        obj_boxes = rois[rel_inds[:, 2], :][:, 1:]

        subj_xmin = subj_boxes[:,0]
        subj_ymin = subj_boxes[:,1]
        subj_xmax = subj_boxes[:,2]
        subj_ymax = subj_boxes[:,3]

        obj_xmin = obj_boxes[:,0]
        obj_ymin = obj_boxes[:,1]
        obj_xmax = obj_boxes[:,2]
        obj_ymax = obj_boxes[:,3]

        xmin = torch.min(subj_xmin, obj_xmin)
        ymin = torch.min(subj_ymin, obj_ymin)

        subj_rois = Variable(
            torch.zeros(rel_inds.size(0), rois.size(1))).cuda(
                rois.get_device()).detach()

        obj_rois = Variable(
            torch.zeros(rel_inds.size(0), rois.size(1))).cuda(
                rois.get_device()).detach()

        subj_rois[:,0] = rois[rel_inds[:,1]][:,0]
        subj_rois[:,1] = (subj_xmin - xmin) / self.img_size * self.f_size
        subj_rois[:,2] = (subj_ymin - ymin) / self.img_size * self.f_size
        subj_rois[:,3] = (subj_xmax - xmin) / self.img_size * self.f_size
        subj_rois[:,4] = (subj_ymax - ymin) / self.img_size * self.f_size

        obj_rois[:,0] = rois[rel_inds[:,2]][:,0]
        obj_rois[:,1] = (obj_xmin - xmin) / self.img_size * self.f_size
        obj_rois[:,2] = (obj_ymin - ymin) / self.img_size * self.f_size
        obj_rois[:,3] = (obj_xmax - xmin) / self.img_size * self.f_size
        obj_rois[:,4] = (obj_ymax - ymin) / self.img_size * self.f_size

        return subj_rois, obj_rois
