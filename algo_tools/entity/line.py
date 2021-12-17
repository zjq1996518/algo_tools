import operator

import cv2
import numpy as np


def calc_line_gradient(line):
    """
    计算直线的斜率
    """
    x1, y1, x2, y2 = line
    return (y2 - y1) / (x2 - x1 + 1e-18)


def calc_line_equation(line):
    """
    计算直线方程 斜率 截距
    """
    x1, y1, x2, y2 = line
    k = calc_line_gradient(line)
    b = ((y1 - k * x1) + (y2 - k * x2)) / 2
    return k, b


def line_direction(line):
    """
    计算直线方向
    Returns: 0 横向 1 竖向
    """
    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        return 0
    return 1


class Line(object):

    """
    线段存储对象，根据线的方向，保证线的坐标从左到右或是从上到下
    属性：
    x1 第一个点的x坐标
    x2 第二个点的x坐标
    y1 第一个点的y坐标
    y2 第二个点的y坐标
    direction 线的防线， 0 横向， 1 竖向
    slope 斜率
    intercept 截距
    theta 角度
    length 直线长度
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.direction = line_direction([x1, y1, x2, y2])

        # 保证直线坐标一定从上到下 从左到右
        self.adjust_line_coord()

        self.slope, self.intercept = calc_line_equation([x1, y1, x2, y2])
        self.theta = self.calc_theta()
        self.length = self.calc_length()

        # 当前线段可能是由一堆子线合并而成，将这些子线记录下来
        self.sub_lines = []

    def copy(self):
        return Line(self.x1, self.y1, self.x2, self.y2)

    @property
    def start_point(self):
        return self.x1, self.y1

    @property
    def end_point(self):
        return self.x2, self.y2

    @property
    def coord(self):
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]

    @property
    def accurate_coord(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def calc_theta(self):
        theta = np.arctan(self.slope) * (180 / np.pi)
        return theta

    def calc_length(self):
        return float(((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2) ** 0.5)

    def adjust_line_coord(self):
        """
        根据线段方向（横向、竖向）将坐标调整为从左到右 或 从上到下
        :return:
        """
        temp_line = [self.start_point, self.end_point]
        temp_line = sorted(temp_line, key=operator.itemgetter(0)) \
            if self.direction == 0 else sorted(temp_line, key=operator.itemgetter(1))
        self.x1 = temp_line[0][0]
        self.y1 = temp_line[0][1]
        self.x2 = temp_line[1][0]
        self.y2 = temp_line[1][1]

    def line_segment_intersecting(self, other):
        if self.direction == 0:
            if self.x1 < other.x2 and self.x2 > other.x1:
                return True

        if self.direction == 1:
            if self.y1 < other.y2 and self.y2 > other.y1:
                return True

        return False

    def calc_line_gap(self, other):
        if self.direction == 0:
            return self.x1 - other.x2 if self.x1 > other.x1 else other.x1 - self.x2
        return self.y1 - other.y2 if self.y1 > other.y1 else other.y1 - self.y2

    def calc_theta_err(self, other):
        if self.direction == 0:
            theta1 = 180 + self.theta
            theta2 = 180 + other.theta
            return abs(theta1 - theta2)
        else:
            theta1 = self.theta if self.theta >= 0 else 180 - abs(self.theta)
            theta2 = other.theta if other.theta >= 0 else 180 - abs(other.theta)
            return abs(theta1 - theta2)

    def line_combine(self, other):

        self_line = self.copy()
        self_line.sub_lines = self.sub_lines
        self.sub_lines = []

        # 保证不加入和并线
        if len(self_line.sub_lines) == 0:
            self.sub_lines.append(self_line)
        else:
            self.sub_lines += self_line.sub_lines

        if len(other.sub_lines) == 0:
            self.sub_lines.append(other)
        else:
            self.sub_lines += other.sub_lines

        if self.direction == 0:
            combine_coord = [(self.y1, self.x1), (other.y1, other.x1)]
            self.y1, self.x1 = min(combine_coord, key=lambda x: x[1])
            combine_coord = [(self.y2, self.x2), (other.y2, other.x2)]
            self.y2, self.x2 = max(combine_coord, key=lambda x: x[1])

            self.theta = self.calc_theta()
            self.length = self.calc_length()
            self.slope, self.intercept = calc_line_equation([self.x1, self.y1, self.x2, self.y2])
            return

        combine_coord = [(self.y1, self.x1), (other.y1, other.x1)]
        self.y1, self.x1 = min(combine_coord, key=lambda x: x[0])
        combine_coord = [(self.y2, self.x2), (other.y2, other.x2)]
        self.y2, self.x2 = max(combine_coord, key=lambda x: x[0])

        self.theta = self.calc_theta()
        self.length = self.calc_length()
        self.slope, self.intercept = calc_line_equation([self.x1, self.y1, self.x2, self.y2])

    def rotate(self, mat):
        """
        对当前这条直线旋转
        Returns:
        """
        self.x1, self.y1 = np.matmul(mat, np.array([self.x1, self.y1, 1]))
        self.x2, self.y2 = np.matmul(mat, np.array([self.x2, self.y2, 1]))
        self.slope, self.intercept = calc_line_equation([self.x1, self.y1, self.x2, self.y2])
        self.theta = self.calc_theta()
        self.length = self.calc_length()

    def calc_coord_y_by_x(self, x):
        return self.slope * x + self.intercept

    def calc_coord_x_by_y(self, y):
        return (y - self.intercept) / (self.slope + 1e-6)

    def cal_crossover_point(self, other):
        """
        两条线找交点
        :return: x y
        """
        x = (other.intercept - self.intercept) / (self.slope - other.slope)
        y = self.slope * x + self.intercept
        return x, y

    @staticmethod
    def line_extend(line, width, height):
        k, b = calc_line_equation(line)

        x1 = 0
        y1 = k * x1 + b

        x2 = width - 1
        y2 = k * x2 + b

        y3 = 0
        x3 = (y3 - b) / (k + 1e-6)

        y4 = height - 1
        x4 = (y4 - b) / (k + 1e-6)

        points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points = [[int(p) for p in point] for point in points]
        new_line = []
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                new_line.append(x)
                new_line.append(y)

                if len(new_line) == 4:
                    return new_line

    def draw(self, img, color=(255, 255, 255), color_random=False, thickness=1):
        if color_random:
            color = (np.random.rand(3) * 255)
            color = [int(color) for color in color]
        start_point = (round(self.start_point[0]), round(self.start_point[1]))
        end_point = (round(self.end_point[0]), round(self.end_point[1]))
        cv2.line(img, start_point, end_point, color, thickness)
        return img

