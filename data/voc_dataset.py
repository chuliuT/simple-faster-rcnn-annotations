import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image

# VOC是20类加上背景就是21类
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOCDataset:
    #用于解析xml内标记的目标的位置信息[xmin,ymin,xmax,ymax]-->chainer [ymin,xmin,ymax,xmax]方便计算
    def __init__(self, data_dir, split='trainval', use_diffcult=False, return_diffcult=False):
        """
        :param data_dir: 数据集路径
        :param split: 读取的文件名
        :param use_diffcult: 是否为困难样本
        :param return_diffcult: 是否返回困难样本
        """
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))#拼接文件的路径
        self.ids = [id_.strip() for id_ in open(id_list_file)]#使用生成器将图像的名字存于一个list
        self.data_dir = data_dir#VOC2007所在的文件位置
        self.use_diffcult = use_diffcult
        self.return_diffcult = return_diffcult
        self.label_name = VOC_BBOX_LABEL_NAMES#标签的tuple

    def __len__(self):
        return len(self.ids)#返回数据集的大小，也就是图片的总数量

    def get_example(self, i):
        id_ = self.ids[i]#获取一个文件名
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))#xml树解析这个文件名
        bbox = list()#定义list存放bbox信息
        label = list()
        difficult = list()
        for obj in anno.findall('object'):#找到所有的object，遍历
            if not self.use_diffcult and int(obj.find('difficult').text) == 1:#判断是否为困难样本
                continue#跳过困难样本
            difficult.append(int(obj.find('difficult').text))#不是困难样本   diff为0
            bndbox_anno = obj.find('bndbox')#获取bbox的信息
            bbox.append([int(bndbox_anno.find(tag).text) - 1
                         for tag in ('ymin', 'xmin', 'ymax', 'xmax')])#以[ymin,xmin,ymax,xmax]格式存储与list
            name = obj.find('name').text.lower().strip()#获取xml中目标的名字
            label.append(VOC_BBOX_LABEL_NAMES.index(name))#转换为整数（索引）
        bbox = np.stack(bbox).astype(np.float32)#沿着竖直方向堆叠成二为数组
        label = np.stack(label).astype(np.int32)

        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')#拼接图像的路径
        img = read_image(img_path, color=True)#读取图像

        return img, bbox, label, difficult

    __getitem__ = get_example#pytorch数据集需要 复写  __len__ __getitem__两个方法
