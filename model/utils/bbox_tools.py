import numpy as np
import six
from six import __init__


def loc2bbox(src_bbox, loc):
    """
    已知源框和偏移量求目标框
    :param src_bbox:
    :param loc:
    :return:
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # 转化为中心点宽高
    src_height = src_bbox[:, 2] - src_bbox[:, 0]  # ymax-ymin 得到框的高
    src_width = src_bbox[:, 3] - src_bbox[:, 1]  # xmax-xmin  得到框的宽
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height# 计算出中心点的Y值
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width# 计算出中心点的X值

    dy = loc[:, 0::4]#取出中心点的偏移量dy
    dx = loc[:, 1::4]#取出中心点的偏移量dx
    dh = loc[:, 2::4]#
    dw = loc[:, 3::4]
    # src的维度是一维的 需要增加一个维度
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    # 【centery，centerx，H，W】装换成左上角和右下角角标
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    anchor 和 gt 求它们之间的  偏移量
    :param src_bbox:
    :param dst_bbox:
    :return:
    """
    # 左上角和右下角 转换为中心点宽高的形式
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width
    # 左上角和右下角 转换为中心点宽高的形式
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps#最小的正数，保证除法不报错
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    # 求偏移量  anchor相对于bbox（gt）
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    :param bbox_a:
    :param bbox_b:
    :return:
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    #bboxa的shape为[N,1,2] bboxb的shape为[K,2]
    # numpy 广播的性质  输出[N,K]
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    生成左上角的9个框，然后后面会+上一个偏移矩阵，这样避免使用双重循环
    :param base_size:
    :param ratios:
    :param anchor_scales:
    :return:
    """
    # anchor的中心点
    py = base_size / 2.
    px = base_size / 2.
    #anchor的数量 【9】  anchor_base 用于存放这些anchor 【9,4】
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])#乘以basesize  对应原图上的坐标根号
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j# anchor 的所以了从左到右从上到下的顺序
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def __test():
    pass


if __name__ == '__main__':
    __test()
