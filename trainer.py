from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from torch import nn
import torch as t
from utils import array_tool as at
from utils.config import opt

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses
    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn# faster-rcnn的对象
        self.rpn_sigma = opt.rpn_sigma# RPN的超参数  3.
        self.roi_sigma = opt.roi_sigma# roi的超参数  1.

        # target creator create gt_bbox gt_label etc as training targets.
        # generate anchor 生成20000多个框  然后 RPN 训练阶段 会 挑 256个 anchor
        # 用于和 bbox 一起计算 loc  label部分会赋值   -1 是忽略  0是负样本 128个  1是正样本 128个
        self.anchor_target_creator = AnchorTargetCreator()
        # 从 训练 20000-->12000-->2000 个ROIS 挑选 128个 正负比例 1:3
        # 测试    12000-->6000>--》300
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper


    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]#bbox的个数
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape# 图像的高宽
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)#features
        # rpn_locs 的维度( hh*ww*9 , 4 ), rpn_scores 维度为( hh*ww*9 , 2 ) ,
        # rois 的维度为( 2000,4 ), roi_indices 用不到, anchor 的维度为
        # ( hh*ww*9 , 4 ), H 和 W 是经过数据预处理后的。计算( H/16 ) x(W/16)x9
        # ( 大概 20000) 个 anchor 属于前景的概率 , 取前 12000 个并经过 NMS 得到 2000 个
        # 近似目标框 G^ 的坐标。 roi 的维度为 (2000,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]#[N,R,4]
        label = labels[0]#[N,R]
        rpn_score = rpn_scores[0]#hh*ww*9
        rpn_loc = rpn_locs[0]#hh*ww*9
        roi = rois#[2000,4]

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input

        # 调用proposal_target_creator函数生成sample roi(128, 4)、
        # gt_roi_loc(128, 4)、 gt_roi_label(128, 1),
        # RoIHead网络利用这
        # sample_roi + featue为输入, 输出是分类(21
        # 类)和回归(进一步微调 bbox)的预测值, 那么分类回归的
        # groudtruth
        # ProposalTargetCreator输出的gt_roi_label/gt_roi_loc

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # 输入 20000 个 anchor 和 bbox ,调用 anchor_target_creator 函数得到
        # 2000 个 anchor 与 bbox 的偏移量与 label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]


        # ------------------ ROI losses (fast rcnn loss) -------------------#
        # roi_cls_loc 为 VGG16RoIHead 的输出( 128,84) , n_sample=128
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        #[128,21,4]
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())


        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        t.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


def _smooth_l1_loss(x, t, in_weight, sigma):
    # https: // www.pianshen.com / article / 4192769463 /
    sigma2 = sigma ** 2 #sigma 的平方
    diff = in_weight * (x - t)# 去 weight==1 的位置计算
    abs_diff = diff.abs()#绝对值
    flag = (abs_diff.data < (1. / sigma2)).float()# 判断值的范围
    #小于 1/simga**2 的 用 0.5*sigma平方* diffs平方
    #大于 1/simga**2     绝对值-0.5//sigma2
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
