from __future__ import absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm
from torch.utils import data as data_
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
# from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.faster_rcnn_resnet50 import FasterRCNN_ResNet50
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc
from torch.utils.tensorboard import SummaryWriter
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)  # 解析配置参数
    #
    dataset = Dataset(opt)  # 训练集 voc2007  5011 张
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt, split='val')  # 验证集 2500左右
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNN_ResNet50()  # 生成一个faster-rcnn实例
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()  # tocuda
    if opt.load_path:  # 加载与训练模型
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    best_map = 0
    lr_ = opt.lr
    writer = SummaryWriter('logs', comment='faster-rcnn-vgg16')
    global_step = 0
    for epoch in range(opt.epoch):  # 开始迭代 14轮  0-12  13个epoch

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            loss = trainer.train_step(img, bbox, label, scale)
            rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss = loss
            writer.add_scalar('rpn_loc_loss', rpn_loc_loss.detach().cpu().numpy(), global_step)
            writer.add_scalar('rpn_cls_loss', rpn_cls_loss.detach().cpu().numpy(), global_step)
            writer.add_scalar('roi_loc_loss', roi_loc_loss.detach().cpu().numpy(), global_step)
            writer.add_scalar('roi_cls_loss', roi_cls_loss.detach().cpu().numpy(), global_step)
            writer.add_scalar('total_loss', total_loss.detach().cpu().numpy(), global_step)
            global_step += 1
            if (ii + 1) % opt.plot_every == 0:
                pass
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}'.format(str(lr_), str(eval_result['map']))
        print(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break


if __name__ == '__main__':
    train()
