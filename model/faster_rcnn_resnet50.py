from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import resnet50
from torchvision.ops import RoIPool
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


# reference:https://github.com/bubbliiiing/faster-rcnn-pytorch
# easy-faster-rcnn.pytorch resnet50
def decom_resnet50():
    resnet_50 = resnet50(pretrained=True)

    # list(resnet50.children()) consists of following modules
    #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
    #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
    #   [5] = Sequential(Bottleneck...),
    #   [6] = Sequential(Bottleneck...),
    #   [7] = Sequential(Bottleneck...),
    #   [8] = AvgPool2d, [9] = Linear
    children = list(resnet_50.children())
    features = children[:-3]
    classifier = children[-3:-1]

    for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        for parameter in parameters:
            parameter.requires_grad = False

    features = nn.Sequential(*features)
    return nn.Sequential(*features), nn.Sequential(*classifier)


class FasterRCNN_ResNet50(FasterRCNN):
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_resnet50()
        # for idx,layer in enumerate(extractor.parameters()):
        #     if idx <len(extractor)-2:
        #         layer.requires_grad = False

        # for param in classifier.parameters():
        #     param.requires_grad = False

        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = ResNet50RoIHead(
            n_class=n_fg_class + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        # 用于初始化 父类
        super(FasterRCNN_ResNet50, self).__init__(
            extractor,
            rpn,
            head,
        )


class ResNet50RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(ResNet50RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        # roi pooling
        pool = self.roi(x, indices_and_rois)
        # flatten
        fc7 = self.classifier(pool)

        fc7 = fc7.view(fc7.size(0), -1)
        # print(fc7.shape)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


if __name__ == '__main__':
    net, cls = decom_resnet50()
    import torch
    print(len(net))
    x = torch.rand(1, 3, 600, 600)
    out = net(x)
    print(cls)
    print(out.shape)
