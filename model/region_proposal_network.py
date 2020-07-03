import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn
from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 feat_stride=16,
                 proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

    def forward(self, x, img_size, scale=1.):
        n,_,hh,ww=x.shape
        anchor=_enumerate_shifted_anchor(np.array(self.anchor_base),self.feat_stride,hh,ww)

        n_anchor=anchor.shape[0]//(hh*ww)#9个

        h=F.relu(self.conv1(x))#512个3x3卷积(512, H/16,W/16)

        rpn_locs=self.loc(h)#n_anchor（9）*4个1x1卷积，回归坐标偏移量。（9*4，hh,ww）

        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)#转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        rpn_scores = self.score(h)#n_anchor（9）*2个1x1卷积，前景背景。（9*2，hh,ww）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()#转换为（n，hh，ww，9*2）
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)#（n，hh，ww，9，2）
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()#取出可能存在目标的框的分数 （n，hh，ww，9，1）
        rpn_fg_scores = rpn_fg_scores.view(n, -1)#所有的前景概率分数（n，hh*ww*9*1）
        rpn_scores = rpn_scores.view(n, -1, 2)##所有的前景概率分数（n，hh*ww*9，2）

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor