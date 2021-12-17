from typing import List

from algo_tools.entity.bbox import Bbox


class TextBbox(Bbox):
    def __init__(self, coord: List[int] = None, box_type=None, text: str = None):
        super().__init__(coord, box_type)

        self.text = text

    def __str__(self):
        return f'[{self.x1}, {self.y1}, {self.x4}, {self.y4}, {self.box_type}, {self.text}]'

    def set_text(self, *args, **kwargs):
        if args is not None and len(args) > 1:
            self.text = args[1]
        elif kwargs.get('text') is not None:
            self.text = kwargs['text']
        return self

    def from_database(self, data, *args, **kwargs):
        super().from_database(data, *args, **kwargs)
        self.set_text(*args, **kwargs)

    def from_coco(self, data, *args, **kwargs):
        super().from_coco(data, *args, **kwargs)
        self.set_text(*args, **kwargs)

    def from_relative(self, data, *args, **kwargs):
        super().from_relative(data, *args, **kwargs)
        self.set_text(*args, **kwargs)
        return self

    def from_coord(self, data, *args, **kwargs):
        super().from_coord(data, *args, **kwargs)
        self.set_text(*args, **kwargs)
        return self

    def from_baidu_ocr(self, data, *args, **kwargs):
        super().from_baidu_ocr(data, *args, **kwargs)
        self.set_text(*args, **kwargs)
        return self


if __name__ == '__main__':
    print(TextBbox().from_coord([1, 2, 3, 4]))