import json
import math
import os
import shutil
from glob import glob

import cv2


def rect2coco(img, img_name, bboxes, labels=None):
    height, width, _ = img.shape
    res = {
        'version': '4.5.7',
        'flags': {},
        'shapes': [
        ],
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width

    }

    for i, box in enumerate(bboxes):
        shape = {
            "label": "0" if labels is None else str(labels[i]),
            "points": [
                [
                    box[0],
                    box[1]
                ],
                [
                    box[2],
                    box[3]
                ]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        res['shapes'].append(shape)

    return res


def coco2yolov4(cocos, img_dir, save_path):
    os.makedirs(save_path, exist_ok=True)
    categories = {}
    for coco in cocos:
        shapes = coco['shapes']
        img_path = coco['imagePath']
        img_width = coco['imageWidth']
        img_height = coco['imageHeight']
        anns = []
        for shape in shapes:
            label = shape['label']
            if categories.get(label) is None:
                categories[label] = len(categories)
            label = categories[label]
            points = shape['points']
            bbox_height = points[1][1] - points[0][1]
            bbox_width = points[1][0] - points[0][0]
            center_x = points[0][0] + bbox_width / 2
            center_y = points[0][1] + bbox_height / 2

            center_x /= img_width
            center_y /= img_height
            bbox_height /= img_height
            bbox_width /= img_width
            if 1 <= center_x or center_x <= 0 or 1 <= center_y or center_y <= 0 or 1 <= bbox_height or bbox_height <= 0 or 1 <= bbox_width or bbox_width <= 0:
                continue
            ann = f'{label} {center_x} {center_y} {bbox_width} {bbox_height}\n'
            anns.append(ann)
        with open(os.path.join(save_path, img_path.lower().replace('jpg', 'txt')), 'w') as f:
            f.writelines(anns)
        shutil.copy2(os.path.join(img_dir, img_path), os.path.join(save_path, img_path))


def aggregate_coco(cocos, one_category=False, category_name=None):
    images = []
    annotations = []
    categories = {'bg': 0} if not one_category else {'bg': 0, category_name: 1}
    for i, coco in enumerate(cocos):
        img = {'file_name': coco['imagePath'], 'id': i, 'height': coco['imageHeight'], 'width': coco['imageWidth']}
        shapes = coco['shapes']
        for shape in shapes:
            # 增加新的类别
            if not one_category and categories.get(shape['label']) is None:
                categories[shape['label']] = len(categories)
            category_id = 1 if one_category else categories[shape['label']]
            area = (shape['points'][1][0] - shape['points'][0][0]) * (shape['points'][1][1] - shape['points'][0][1])
            x1 = shape['points'][0][0]
            y1 = shape['points'][0][1]
            box_width = shape['points'][1][0] - x1
            box_height = shape['points'][1][1] - y1
            annotation = {'id': len(annotations),
                          'image_id': i,
                          'bbox': [x1, y1, box_width, box_height],
                          'category_id': category_id,
                          'area': area,
                          'segmentation': [],
                          'iscrowd': 0}

            annotations.append(annotation)
        images.append(img)
    categories = [{'id': v, 'name': k} for k, v in categories.items()]

    return {'images': images, 'annotations': annotations, 'categories': categories}


def organize(origin_data_path, save_path, save_file_name, delete_data=True, divide=False, divide_rate=0.8,
             one_category=False, category_name=None):
    """
    将标注的数据转换为coco格式
    :param origin_data_path: 转换前数据目录
    :param save_path: 转换后数据目录
    :param save_file_name: annotations文件名
    如果划分验证集则自动添加后缀 f'{save_file_name}-train.json' f'{save_file_name}-val.json'
    :param delete_data: 转换前是否清空save_path中的数据
    :param divide 是否划分验证集
    :param divide_rate 训练集占总数据的百分比
    :param one_category 是否只有一类
    :param category_name 只有一类时的类别名
    :return:
    """

    def save_cocos(ann, file_name):
        ann = aggregate_coco(ann, one_category=one_category, category_name=category_name)
        with open(f'{save_path}/annotations/{file_name}', 'w') as f:
            json.dump(ann, f)

    if delete_data and os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    json_list = glob(f'{origin_data_path}/*.json')
    cocos = []
    for json_file in json_list:
        with open(json_file) as f:
            coco = json.load(f)
        img_name = coco['imagePath']
        img_path = os.path.join(origin_data_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        cv2.imwrite(os.path.join(save_path, img_name), img)
        cocos.append(coco)
    # 判断是否划分验证集
    os.makedirs(f'{save_path}/annotations', exist_ok=True)
    if divide:
        train_cocos = cocos[:math.ceil(divide_rate * len(cocos))]
        val_cocos = cocos[math.ceil(divide_rate * len(cocos)):]
        save_cocos(train_cocos, save_file_name.replace('.json', '-train.json'))
        save_cocos(val_cocos, save_file_name.replace('.json', '-val.json'))
    else:
        save_cocos(cocos, save_file_name)


if __name__ == '__main__':
    coco2yolov4()