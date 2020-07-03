import numpy as np
from PIL import Image
import random


def read_image(path, dtype=np.float32, color=True):
    """
    :param path:图片的路径
    :param dtype:使用浮点32位
    :param color:是否使用RGB
    :return:
    """
    f = Image.open(path)  # PIL的对象 图像的通道为[h,w,c]
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)  # 转成numpy array
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:  # 图像是灰度图的话
        return img[np.newaxis]  # 增加一个维度
    else:
        return img.transpose((2, 0, 1))  # [h,w,c]-->[c,h,w]


def resize_bbox(bbox, in_size, out_size):
    """
    :param bbox:原始的bbox【ymin，xmin，ymax，xmax】
    :param in_size:原始的(h,w)
    :param out_size:变换后的(H,W)
    :return:bbox图像缩放后，bbox需要跟着缩放
    """
    bbox = bbox.copy()#先copy一份
    y_scale = float(out_size[0]) / in_size[0]#对应图像上H的缩放比例
    x_scale = float(out_size[1]) / in_size[1]#对应图像上W的缩放比例
    bbox[:, 0] = y_scale * bbox[:, 0]#缩放H  y部分
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]#缩放W  x部分
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    :param bbox:待翻转的bbox
    :param size:图像宽高
    :param y_flip:y方向翻转标志位
    :param x_flip:x方向翻转标志位
    :return:
    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:#沿着垂直方向翻转
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:#沿着水平方向翻转
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    """
    :param img:输入图像
    :param y_random:
    :param x_random:
    :param return_param:用于判断是否翻转bbox，和沿着什么方向翻转
    :param copy:
    :return:
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:#图像的翻转   y对应于图像Height
        img = img[:, ::-1, :]
    if x_flip:#图像的翻转   对应于图像Width
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img
