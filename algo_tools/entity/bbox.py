from typing import List

import cv2
import numpy as np


class Bbox(object):

    def __init__(self, coord: List[int] = None, box_type=None):
        if coord is None:
            coord = [0 for _ in range(8)]

        self.x1 = coord[0]
        self.y1 = coord[1]
        self.x2 = coord[2]
        self.y2 = coord[3]
        self.x3 = coord[4]
        self.y3 = coord[5]
        self.x4 = coord[6]
        self.y4 = coord[7]

        self.sub_bboxes = []
        self.box_type = box_type

    def append(self, bbox):
        self.sub_bboxes.append(bbox)

    def __getitem__(self, i):
        return self.sub_bboxes[i]

    def coord_rescaling(self, shift_x=0, scale_x=1, shift_y=0, scale_y=1):
        """
        用于对box坐标进行偏移
        @param shift_x x方向上的偏移
        @param scale_x x方向上的尺度
        @param shift_y y方向上的偏移
        @param scale_y y方向上的尺度
        """
        self.x1 = self._coord_rescaling(self.x1, shift_x, scale_x)
        self.x2 = self._coord_rescaling(self.x2, shift_x, scale_x)
        self.x3 = self._coord_rescaling(self.x3, shift_x, scale_x)
        self.x4 = self._coord_rescaling(self.x4, shift_x, scale_x)

        self.y1 = self._coord_rescaling(self.y1, shift_y, scale_y)
        self.y2 = self._coord_rescaling(self.y2, shift_y, scale_y)
        self.y3 = self._coord_rescaling(self.y3, shift_y, scale_y)
        self.y4 = self._coord_rescaling(self.y4, shift_y, scale_y)

        return self

    @staticmethod
    def _coord_rescaling(coord, shift, scale):
        coord = (coord + shift) * scale
        coord = coord if coord > 0 else 0
        return coord

    @property
    def width(self):
        return int(self.x2) - int(self.x1)

    @property
    def height(self):
        return int(self.y3) - int(self.y1)

    @property
    def start_point(self):
        return int(self.x1), int(self.y1)

    @property
    def end_point(self):
        return int(self.x4), int(self.y4)

    @property
    def point(self):
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2), int(self.x3), int(self.y3), int(self.x4),
                int(self.y4)]

    @property
    def coord(self):
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]

    @property
    def area(self):
        return self.width * self.height

    @property
    def accurate_rect(self):
        return [self.x1, self.y1, self.x4, self.y4]

    @property
    def rect(self):
        return [round(self.x1), round(self.y1), round(self.x4), round(self.y4)]

    def __str__(self):
        return f'[{self.x1}, {self.y1}, {self.x4}, {self.y4}] square: {self.area}'

    def inner(self, other, offset=5):
        if other.x1 >= self.x1 - offset and other.x4 <= self.x4 + offset and other.y1 >= self.y1 - offset and other.y4 <= self.y4 + offset:
            return True
        return False

    def draw(self, img, color=(255, 255, 255), color_random=False, thickness=1):
        if color_random:
            color = (np.random.rand(3) * 255)
            color = [int(color) for color in color]
        start_point = (round(self.start_point[0]), round(self.start_point[1]))
        end_point = (round(self.end_point[0]), round(self.end_point[1]))
        cv2.rectangle(img, start_point, end_point, color, thickness)
        return img

    def set_box_type(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            self.box_type = args[0]
        elif kwargs.get('box_type') is not None:
            self.box_type = kwargs['box_type']
        return self

    def from_other_box(self, bbox):
        self.x1, self.y1 = float(bbox.x1), float(bbox.y1)
        self.x2, self.y2 = float(bbox.x2), float(bbox.y2)
        self.x3, self.y3 = float(bbox.x3), float(bbox.y3)
        self.x4, self.y4 = float(bbox.x4), float(bbox.y4)
        return self

    def from_database(self, data, *args, **kwargs):
        self.x1, self.y1 = float(data['startX']), float(data['startY'])
        self.x4, self.y4 = float(data['endX']), float(data['endY'])
        self.x2, self.y2 = self.x4, self.y1
        self.x3, self.y3 = self.x1, self.y4
        self.set_box_type(*args, **kwargs)

        return self

    def from_coco(self, data, *args, **kwargs):
        points = data['points']
        self.x1 = float(points[0][0])
        self.x4 = float(points[1][0])
        self.y1 = float(points[0][1])
        self.y4 = float(points[1][1])
        self.x2, self.y2 = self.x4, self.y1
        self.x3, self.y3 = self.x1, self.y4

        self.set_box_type(*args, **kwargs)

        return self

    def from_relative(self, data, *args, **kwargs):
        self.x1, self.y1 = float(data['left']), float(data['top'])
        self.x4, self.y4 = float(data['width']) + float(data['left']), float(data['height']) + float(data['top'])
        self.x2, self.y2 = self.x4, self.y1
        self.x3, self.y3 = self.x1, self.y4

        self.set_box_type(*args, **kwargs)

        return self

    def from_coord(self, data, *args, **kwargs):
        self.x1, self.y1 = float(data[0]), float(data[1])
        self.x4, self.y4 = float(data[2]), float(data[3])
        self.x2, self.y2 = self.x4, self.y1
        self.x3, self.y3 = self.x1, self.y4

        self.set_box_type(*args, **kwargs)

        return self

    def from_baidu_ocr(self, data, *args, **kwargs):
        text_region = data['text_region']
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = [
            text_region[0][0], text_region[0][1], text_region[1][0], text_region[1][1],
            text_region[3][0], text_region[3][1], text_region[2][0], text_region[2][1]
        ]
        self.set_box_type(*args, **kwargs)
        return self


if __name__ == '__main__':
    test_box = Bbox([1, 2, 3, 4, 5, 6, 7, 8])
