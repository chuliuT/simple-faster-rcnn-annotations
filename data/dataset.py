import torch as t
from skimage import transform as sktfs
from torchvision import transforms as tvtfs
from data.voc_dataset import VOCDataset
from data import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    """
    :param img:经过归一化的img（0~1）减去均值除以方差的
    :return:
    """
    img[0, :, :] = (img[0, :, :] * 0.229 + 0.485)#乘以方差+均值
    img[1, :, :] = (img[1, :, :] * 0.224 + 0.224)
    img[2, :, :] = (img[2, :, :] * 0.225 + 0.225)
    return img


def pytorch_normalize(img):
    """
    numpy转tensor 用torchvision包来做变换，然后再变换回numpy
    :param img:
    :return:
    """
    normalize = tvtfs.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    """
    将长边缩放为1000 短边为600
    :param img:
    :param min_size:
    :param max_size:
    :return:
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)#短边的scale
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)#选取小的比例
    img = img / 255.#图像归一化到0~1
    img = sktfs.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)#anti 是否使用高斯滤波
    normalize = pytorch_normalize
    return normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)#对图像缩放处理
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))#对bbox处理
        #图像随机水平翻转
        img, params = util.random_flip(img, x_random=True, return_param=True)
        #对应的bbox
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        #生成类的实例
        self.db = VOCDataset(opt.voc_data_dir)
        # 生成类的实例
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        #调用函数来获取一张图片的标注信息
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        #
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCDataset(opt.voc_data_dir,split=split)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import cv2

    # traindataset = Dataset(opt)
    # train_iter = iter(traindataset)
    # for i in range(3):
    #     img, bbox, label, scale = next(train_iter)
    #     print(img.shape, bbox.shape, label, scale)
    #     img = inverse_normalize(img)
    #     img = img.transpose(1, 2, 0)
    #     drawimg=img.copy()
    #
    #     for idx,bbox_data in enumerate(bbox):
    #         cv2.rectangle(drawimg, (int(bbox_data[1]), int(bbox_data[0])), (int(bbox_data[3]), int(bbox_data[2])), (0, 255, 0),4);
    #
    #     # print(img.shape)
    #     plt.imshow(drawimg)
    #     plt.show()

    testdataset = TestDataset(opt)
    test_iter = iter(testdataset)
    for i in range(3):
        img, ori_img_shape, bbox, label, diff = next(test_iter)
        print(img.shape, bbox.shape, label)
        img = inverse_normalize(img)
        img = img.transpose(1, 2, 0)
        drawimg = img.copy()
        H, W, C = drawimg.shape
        # ori_img_shape
        print(ori_img_shape)
        bbox = util.resize_bbox(bbox, (ori_img_shape[0], ori_img_shape[1]), (H, W))
        for idx, bbox_data in enumerate(bbox):
            cv2.rectangle(drawimg, (int(bbox_data[1]), int(bbox_data[0])), (int(bbox_data[3]), int(bbox_data[2])),
                          (0, 255, 0), 4)
        # print(img.shape)
        plt.imshow(drawimg)
        plt.show()
