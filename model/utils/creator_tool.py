import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


def _unmap(data, cout, index, fill=0):
    """
    :param data:
    :param cout:
    :param index:
    :param fill:
    :return:
    """
    if len(data.shape) == 1:
        ret = np.empty((cout,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((cout,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    """
    删除超出边界的框
    :param anchor:
    :param H:
    :param W:
    :return:
    """
    inside_index = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return inside_index


class AnchorTargetCreator(object):
    """
    这个函数是负责给RPN提供训练的样本的，
    会从20000多个Anchor中抽取 256个Anchor 正负样本的比例为1：1
    正样本的定义：iou>0.7 128个
    负样本的定义：iou<0.3 128个
    这些Anchor用于RPN的  前景背景分类9*2 定位任务 9*4定位回归任务
    """
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size  # 图片的高 宽
        n_anchor = len(anchor)  # Anchor的个数
        inside_index = _get_inside_index(anchor, img_H, img_W)  # 超出边界的框需要删掉，返回的是在图片内部的anchor
        anchor = anchor[inside_index]  # 图像边界内的Anchor
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)  # 根据bbox，给label赋值，返回最大
        loc = bbox2loc(anchor, bbox[argmax_ious])  # 计算出这些anchor和gt 的偏移量
        # 将位于图片内部的框的 label 对应到所有生成的 20000 个框中
        # （label 原本为所有在图片中的框的）
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        # 将anchor和bbox的偏移量，对应到所有生成的 20000 个框中（label 原本为
        # 所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # 定义label的变量，全部填-1
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        # anchor与哪个bbox的iou最大，Iou，bbox与哪一个anchor的iou最大
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious <= self.neg_iou_thresh] = 0  # 设置负样本

        label[gt_argmax_ious] = 1  # 把与每个 bbox 求得 iou 值最大的 anchor 的 label 设为 1

        label[max_ious >= self.pos_iou_thresh] = 1  # 设置正样本

        n_pos = int(self.pos_ratio * self.n_sample)  # 正样本的采样率*总采样个数 = 0.5*256
        pos_index = np.where(label == 1)[0]  # 找到正样本的索引
        if len(pos_index) > n_pos:  # 随机丢弃掉多余的正样本，只采样 128个
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1  # 忽略的置为-1

        n_neg = self.n_sample - np.sum(label == 1)  # 负样本的个数
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)  ##调用bbox_iou函数计算anchor与bbox的IOU， ious：（N,K），N为anchor中第N个，K为bbox中第K个，N大概有15000个
        argmax_ious = ious.argmax(axis=1)  ##1代表行，0代表列
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]  # 求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[N，1]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # 求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,K]

        gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # k个，保证每个目标都有一个

        return argmax_ious, max_ious, gt_argmax_ious


class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms  # 训练取出12000
            n_post_nms = self.n_train_post_nms  # NMS后变成2000
        else:
            n_pre_nms = self.n_test_pre_nms  # 测试加快推理 6000
            n_post_nms = self.n_test_post_nms  # nms 300

        roi = loc2bbox(anchor, loc)  # 将anchor 加上loc 接近GT的形式 映射回原图

        # s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2
        # [ymin,xmin,ymax,xmax]
        # 裁剪将rois的ymin,ymax限定在[0,H]
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        # 裁剪将rois的xmin,xmax限定在[0,W]
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale  #
        hs = roi[:, 2] - roi[:, 0]  # Height
        ws = roi[:, 3] - roi[:, 1]  # Width
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]  # 确保这些anchor都在里面
        roi = roi[keep, :]  # 保留这些roi
        score = score[keep]  # roi对应的分数

        order = score.ravel().argsort()[::-1]  #对score进行降序排
        if n_pre_nms > 0:
            order = order[:n_pre_nms]#保留 train 12000
        roi = roi[order, :]#12000
        score = score[order]#12000

        keep = nms(torch.from_numpy(roi).cuda(),
                   torch.from_numpy(score).cuda(),
                   self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]#2000
        roi = roi[keep.cpu()]
        return roi


class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,# rcnn 的训练过程 采样128个proposals 的框
                 pos_ratio=0.25,# 以1:3的比例 128*0.25=32个正例，96个负例
                 pos_iou_thresh=0.5,#正样本的iou阈值
                 neg_iou_thresh_hi=0.5,#负样本的IOU阈值
                 neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)#cat bbox 保证会有重叠度高的用于计算？

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)#128*0.25=32
        iou = bbox_iou(roi, bbox)#计算roi和bbox的IOU
        gt_assignment = iou.argmax(axis=1)# 用于获取和bbox的iou最大的那几个roi
        max_iou = iou.max(axis=1)#获取这几个roi的iou阈值 这里的几个 等于于本图片内的目标的个数
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1#把GTlabel 变成从1开始的  pytorch 不支持label=0计算

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))#这里pos——index有可能小于 32
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]#这里的采样iou有上界和下界
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image# 负样本采样
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)#只取出正样本和负样本的index
        gt_roi_label = gt_roi_label[keep_index]#取出roi的label
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]#取出roi 正负样本的

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])# 计算bbox和采样roi的偏移量
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label
