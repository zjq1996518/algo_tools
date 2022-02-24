# algorithm-tools
图像处理常用工具库 把自己在工作中常用的算法、方法封装起来，以便于开发，包括：
+ entity 实体类
    + Bbox Bounding Box
    + Line 线
    + TextBbox 带文本的 Bounding Box
    
ps: line 类应该有点用，bbox根据业务来的，或许不太有用
line 中实现了很多有用的操作，两条直线的合并，
距离计算，线的斜率、角度、正方向等等

+ image_process 图片处理
    + download_img 图片下载
    + img_crop 图片裁剪
    + standard_resize 图片缩放
    + normalize 图片像素标准化
    + recover_normalize 将标准化的图片像素还原
    + draw_text 图片上显示文字，支持中文
    + calc_iou 计算box iou
    + calc_intersection 计算box交集与面积之比
    + img_extend_slice 图片切片，常用于大图片切片为N块小图


+ image_augmentation 图片增强 常用于给模型做训练
    + rotate_argument 旋转增强，支持传入bbox
    + bright_argument 亮度增强
    + exchange_channel 通道随机交换
    + sp_noise 椒盐噪声
    + hsv_argument 对比度 亮度增强
    + scale_transform 图片随机压缩、扩张，支持传入bbox
    + patch_image 在图片中随机替换黑白块
    

+ MultiTaskExecutor 自以为高性能的带进度条的多进程任务处理器<br/>
由于在实际工作中经常涉及到上百万数据的处理，所以设计了这样一个类
## 安装
```angular2html
pip install git+https://github.com/zjq1996518/algo_tools.git
```

## 例子
### 图片处理
```angular2html
import algo_tools

# 图片下载
url = '你的url'
img = algo_tools.image_process.download_img(url)
# 图片裁剪
cropped = algo_tools.image_process.img_crop(img, [0, 0, 50, 50])

```
### 多任务执行器 不带reduce方法
```angular2html
# 多任务处理器，感觉这个比较有用
from algo_tools.multi_task_executor import MultiTaskExecutor


def target(x, y):
    # 虽然这样写会影响进度条显示
    print(x+y)


executor = MultiTaskExecutor(target=target)
tasks = []
for i in range(100):
    # 这里是每个任务传给target的参数
    tasks.append((i, i+1))
executor.execute(tasks)
```
### 多任务执行器 带reduce方法
```angular2html
from algo_tools.multi_task_executor import MultiTaskExecutor


def target(x, y):
    return x+y


def reduce(rsts, u_parm):
    for rst in rsts:
        print(rst+u_parm)


executor = MultiTaskExecutor(target=target, reduce_func=reduce, reduce_param=(1, ))
tasks = []
for i in range(100):
    # 这里是每个任务传给target的参数
    tasks.append((i, i+1))
executor.execute(tasks)
```