from __future__ import absolute_import

import matplotlib
from tqdm import tqdm
from torch.utils import data as data_
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
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


def cal_map(**kwargs):
    opt._parse(kwargs)  # 解析配置参数
    #
    testset = TestDataset(opt, split='test')  # 验证集 2500左右
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
    faster_rcnn.eval()
    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    log_info = 'lr:{}, map:{}'.format(str(lr_), str(eval_result['map']))
    print(log_info)

# load pretrained model from ./checkpoints/fasterrcnn_07031445_0.7611494152696302
# 4952it [15:40,  5.64it/s]
# lr:0.001, map:0.5486164353700358

if __name__ == '__main__':
    cal_map()
