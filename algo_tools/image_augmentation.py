import random

import cv2
import numpy as np


def rotate_argument(img, boxes=None):
    """

    :param img:
    :param boxes: [n, 4, 2] 左上 右上 左下 右下
    :return:
    """
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    random_angle = random.uniform(-20, 20)
    mat = cv2.getRotationMatrix2D((center_x, center_y), random_angle, 1)
    warp_img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
    if boxes is not None:
        boxes = [np.matmul(mat, np.concatenate([box.transpose(), np.ones([1, box.shape[0]])])).transpose() for box in
                 boxes]
        boxes = np.array(boxes)

        min_x = np.min(boxes[:, :, 0], axis=1)
        max_x = np.max(boxes[:, :, 0], axis=1)
        min_y = np.min(boxes[:, :, 1], axis=1)
        max_y = np.max(boxes[:, :, 1], axis=1)

        boxes[:, 0, 0] = min_x
        boxes[:, 0, 1] = min_y
        boxes[:, 1, 0] = max_x
        boxes[:, 1, 1] = min_y
        boxes[:, 2, 0] = min_x
        boxes[:, 2, 1] = max_y
        boxes[:, 3, 0] = max_x
        boxes[:, 3, 1] = max_y

        return warp_img, boxes
    return warp_img


def bright_argument(img, bright_range=(-32, 32), contrast_range=(0.5, 1.5)):
    delta = random.uniform(*bright_range)
    alpha = random.uniform(*contrast_range)
    img = img * alpha + delta
    img = img.astype(np.uint8)
    return img


def exchange_channel(img):
    return img[..., np.random.permutation(3)]


def sp_noise(image, prob=0.2):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    y_sp = np.random.choice(range(image.shape[0]), size=round(prob * image.shape[0]))
    x_sp = np.random.choice(range(image.shape[1]), size=round(prob * image.shape[1]))

    for y in y_sp:
        for x in x_sp:
            val = random.choice([255, 0])
            image[y, x, :] = val

    return image


def hsv_argument(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.astype(np.float32)
    if np.random.randint(2):
        img[..., 1] *= random.uniform(0.5, 1.5)

    if random.randint(0, 1):
        img[..., 0] += random.uniform(-18, 18)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360

    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def scale_transform(img, o2o=True, boxes=None):
    # 图片随机扩张、压缩
    """
    :param img:
    :param boxes: [n, 4, 2] 维度说明 n: box数量 4: 左上角 右上角 左下角 右下角 2: (x, y)
    :param o2o: 是否等比例缩放
    :return:
    """
    height_ratio = np.random.uniform(0.5, 1.5)
    width_ratio = np.random.uniform(0.5, 1.5)
    if o2o:
        width_ratio = height_ratio
    img = cv2.resize(img, (int(width_ratio * img.shape[1]), int(height_ratio * img.shape[0])), interpolation=cv2.INTER_AREA)
    if boxes is not None:
        boxes[:, :, 0] *= width_ratio
        boxes[:, :, 1] *= height_ratio
        return img, boxes
    return img


def patch_image(img, width_range=(0.05, 0.3), height_range=(0.05, 0.3)):
    height, width, channel = img.shape
    # 随机大小
    block_width_ratio = random.uniform(*width_range)
    block_height_ratio = random.uniform(*height_range)
    block_width = round(block_width_ratio * width)
    block_height = round(block_height_ratio * height)
    # 随机黑或白
    block = np.zeros([block_height, block_width, channel], dtype=np.uint8)
    if random.randint(0, 1):
        block += 255
    # 随机位置
    x = random.randint(0, width - block_width - 1)
    y = random.randint(0, height - block_height - 1)
    # 填充
    img[y: y+block_height, x:x+block_width, ...] = block
    return img
