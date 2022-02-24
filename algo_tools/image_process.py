import math
import random

import cv2
import numpy as np
import requests
from PIL import ImageFont, ImageDraw


def download_img(url):
    """
    根据url下载图片
    :param url: 图片url
    :return: 图片 ndarray 数组
    """
    data = requests.get(url).content
    img = np.asarray(bytearray(data), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def img_crop(img, rect):
    """
    图片裁剪
    :param img:
    :param rect: [x1, x2, y1, y2] -> [水平方向开始坐标, 水平方向结束坐标, ..., ...]
    :return:
    """
    s_h = max(0, rect[1])
    e_h = min(img.shape[0], rect[3])

    s_x = max(0, rect[0])
    e_x = min(img.shape[1], rect[2])

    return img[s_h: e_h, s_x: e_x, ...]


def standard_resize(img, resize_width, resize_height, transform_info=False, pad_value=0, pad_style='center'):
    """
    不改变图片宽高比，对图片进行缩放
    :param img:
    :param resize_height:
    :param resize_width:
    :param transform_info: False 仅返回变换后的图片 True 返回图片缩放率和填充像素， ratio_w, ratio_h, pad_w, pad_h
    :param pad_value 填充像素值
    :param pad_style 填充方式 center, right, left, random_w
    :return:
    """

    def resize(img, flag):
        scale_h = resize_height if flag == 0 else round(resize_width / w * h)
        scale_w = round(resize_height / h * w) if flag == 0 else resize_width
        img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        ratio_h = scale_h / h
        ratio_w = scale_w / w
        return img, ratio_w, ratio_h

    def pad(img):
        pad_img = np.zeros([resize_height, resize_width, 3], dtype=np.uint8) + pad_value
        pad_width = max(resize_width - img.shape[1], 0)
        pad_height = max(resize_height - img.shape[0], 0)
        if pad_style == 'center':
            s_h = math.ceil(pad_height / 2)
            s_w = math.ceil(pad_width / 2)
        elif pad_style == 'right':
            s_h = 0
            s_w = 0
        elif pad_style == 'random_w':
            s_w = random.randint(0, pad_width)
            s_h = math.ceil(pad_height / 2)
        else:
            s_h = pad_height
            s_w = pad_width
        pad_img[s_h:s_h + img.shape[0], s_w:s_w + img.shape[1], :] = img
        return pad_img, s_h, s_w

    h, w = img.shape[:2]
    ratio_w, ratio_h = 1, 1
    # 两个都大, 图片缩放按照差距较大的那个标准，然后对差距较大的做填充
    if h > resize_height and w > resize_width:
        flag = 0 if resize_height / h < resize_width / w else 1
        img, ratio_w, ratio_h = resize(img, flag)

    # 两个都小，图片填充
    elif w <= resize_width and h <= resize_height:
        pass

    # 其中一个大，其中一个小, 对大的那个做缩放，小的做填充
    else:
        flag = 0 if h - resize_height > w - resize_width else 1
        img, ratio_w, ratio_h = resize(img, flag)

    img, pad_h, pad_w = pad(img)

    if transform_info:
        return img, ratio_w, ratio_h, pad_w, pad_h
    return img


def normalize(img, mode='normal'):
    """
    :param img:
    :param mode: none: 对三通道等于0的值不做任何处理，即三通道中等于0的值normalize后依旧为0
    :return:
    """
    mean = np.array([123.675, 116.28, 103.53])[np.newaxis, :]
    std = np.array([58.395, 57.12, 57.375])[np.newaxis, :]

    if mode == 'none':
        condition = (img[:, :, 0] != 0) | (img[:, :, 1] != 0) | (img[:, :, 2] != 0)
        condition = condition[..., np.newaxis]
        img = (img - mean) / std
        img = condition * img
    else:
        img = (img - mean) / std
    return img


def recover_normalize(img, mode='normal'):
    """
    :param img:
    :param mode: none: 对三通道等于0的值不做任何处理，即三通道中等于0的值恢复后依旧为0，需搭配normalize mode=‘none’ 使用
    :return:
    """
    mean = np.array([123.675, 116.28, 103.53])[np.newaxis, :]
    std = np.array([58.395, 57.12, 57.375])[np.newaxis, :]

    if mode == 'none':
        condition = (img[:, :, 0] != 0) | (img[:, :, 1] != 0) | (img[:, :, 2] != 0)
        condition = condition[..., np.newaxis]
        img = img * std + mean
        img = condition * img
    else:
        img = img * std + mean
    return img


def draw_text(img, text, start_point):
    """
    :param img: PIL图片
    :param text:
    :param start_point: (x, y)
    :return:
    """
    ttf = ImageFont.load_default()
    img_draw = ImageDraw.Draw(img)
    img_draw.text(start_point, text, font=ttf, fill=(255, 0, 0))
    return img


def calc_iou(boxes1, boxes2):
    """

    :param boxes1: shape [n, 4] => [[x_start, y_start, x_end, y_end], ...]
    :param boxes2:
    :return:
    """
    backend = np
    if not isinstance(boxes1, np.ndarray):
        import torch
        backend = torch

    lu = backend.maximum(boxes1[..., :2], boxes2[..., :2])
    rd = backend.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算机交集面积
    zero = 0.0 if isinstance(boxes1, np.ndarray) else torch.tensor(0.0)
    intersection = backend.maximum(zero, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # 计算每个box的面积
    square1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    square2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    # 计算并集面积
    esp = 1e-10 if isinstance(boxes1, np.ndarray) else torch.tensor(1e-10)
    union_square = backend.maximum(square1 + square2 - inter_square, esp)

    # numpy 后面两个参数是分别表示最大值与最小值
    return backend.clip(inter_square / union_square, 0.0, 1.0)


def calc_intersection(boxes1, boxes2):
    """
    计算Boxes1 与 boxes2 交集面积与boxes1面积之比
    :param boxes1: shape [n, 4] => [[x_start, y_start, x_end, y_end], ...]
    :param boxes2:
    :return:
    """

    backend = np
    if not isinstance(boxes1, np.ndarray):
        import torch
        backend = torch

    lu = backend.maximum(boxes1[..., :2], boxes2[..., :2])
    rd = backend.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算机交集面积
    zero = 0.0 if isinstance(boxes1, np.ndarray) else torch.tensor(0.0)
    intersection = backend.maximum(zero, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # 计算每个box的面积
    square2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # numpy 后面两个参数是分别表示最大值与最小值
    return backend.clip(inter_square / square2, 0.0, 1.0)


def img_extend_slice(img, slice_w, slice_h, margin_w=0, margin_h=0, need_offset=False):
    """
    :param img: 原图
    :param slice_w: 切片后每张图片的宽度
    :param slice_h: 切片后每张图片的高度
    :param boxes: 目标检测的盒子，如果不为None 则跟随图片一直进行切片分组
    :param allow_size: 所允许的一张切片图最小尺度, 若当前切片宽度或者高度小于allow_size
                        则该部分会被附加到上一张图，不作为一张单独的新图
    :param need_offset 是否需要图片的offset信息
    :param assign_rate box面积保留率，即当前box被裁剪后如果还有assign_rate面积存在，则保留
    :return:
    """

    img_height, img_width = img.shape[:2]
    # assert slice_w < img_width, '切图宽度必须小于原图片宽度'
    # assert slice_h < img_height, '切图高度必须小于原图片高度'

    slice_images = []

    slice_w -= margin_w
    slice_h -= margin_h

    cols = math.ceil((max(img_width - margin_w, 1)) / slice_w)
    rows = math.ceil(max((img_height - margin_h), 1) / slice_h)

    for col in range(cols):
        slice_images.append([])
        w_start = slice_w * col
        # 如果是最后一片且图片不能整除，则通过调整w_start来倒切
        if col == cols - 1 and img_width % slice_w != 0:
            w_start = img_width - slice_w - margin_w
        for row in range(rows):
            # 同上
            h_start = slice_h * row
            if row == rows - 1 and img_height % slice_h != 0:
                h_start = img_height - slice_h - margin_h
            slice_img = img[h_start:h_start + slice_h + margin_h, w_start:w_start + slice_w + margin_w, ...]
            slice_offset = (h_start, w_start)
            if not need_offset:
                slice_images[col].append(slice_img)
            else:
                slice_images[col].append((slice_img, slice_offset))
    return slice_images


if __name__ == '__main__':
    # 测试
    origin_img = np.random.rand(35, 30, 3)
    slice_images = img_extend_slice(origin_img, slice_w=896, slice_h=896, margin_w=200, margin_h=100)