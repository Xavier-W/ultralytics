[@TOC]


## 0. python_environment.txt
不要直接安装这个txt!!! 先运行1-7的脚本，显示缺少modules时，在该文件中找到对应的版本，进行pip install XXX==x.x.x

## 1. data_process.py
```python
    data_dir = "/XX/XX/Bijie-landslide-dataset/"    # 给出官方下载的landslide数据集的主路径
    process = data_process(data_dir)    # 新建data_process类

    process.poly2rect()     # 该函数会将Bijie-landslide-dataset/landslide/polygon_coordinate路径下的txt转换为矩形框标注信息，并存放在Bijie-landslide-dataset/landslide/rectangle_coordinate/ 下
    process.data_aug(pos_augloop=7, neg_augloop=4) # 每张图片扩充的数量，正样本扩充数量为pos_augloop，负样本扩充数量为neg_augloop，扩充后的图像数据和标注数据分别放在Bijie-landslide-dataset/augimages/和Bijie-landslide-dataset/augrect_coords/下
    process.dataset2yolo()  # 该函数会将增强扩充后的数据转换为yolo训练格式，并存放在Bijie-landslide-dataset/yolo_dataset/下
    # process.analyse_center(process.aug_img_dir, process.aug_rect_dir)  # 分析增强后的数据的中心点分布和宽高分布
    # process.analyse_boxwh(process.aug_img_dir, process.aug_rect_dir)
```

## 2. train.py
```python
model = YOLO("XX/ultralytics/cfg/models/v8/yolov8-ghost-p6.yaml")  # load a pretrained model (recommended for training)
model.train(data='/XX/ultralytics/cfg/datasets/labdslide.yaml', cfg="/XX/ultralytics/cfg/default.yaml")  # train the model
```
主要修改3个文件，
- ./ultralytics/cfg/models/v8/yolov8-ghost-p6.yaml      模型参数文件，该文件中主要修改nc类别数
- ./ultralytics/cfg/datasets/labdslide.yaml     数据配置文件，可以复制coco128.yml后重命名，修改为下面格式即可
```python
path: Bijie-landslide-dataset/yolo_dataset/ # dataset root dir  # 使用自己的绝对路径
train: images/train # train images (relative to 'path') 128 images      # 不需要改动
val: images/val # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes
names:
  0: labdslide
```
- ./ultralytics/cfg/default.yaml     # 训练参数配置文件，主要修改参数有
```python
model       # 预训练权重
data        # 数据配置文件路径
epochs      # 训练代数
batch       # batchsize
imgsz       # 训练模型的输入尺寸
device      # 使用GPU的编号或者使用CPU
workers     # 加载数据的线程数，一般为8即可，可以根据电脑核数修改
name        # 训练后生成的文件的保存路径
#其他参数可自行百度
```

## 3. predict.py
模型推理文件

只要修改权重路径、图片路径即可，会自动保存在0_outputs文件夹中，也可以自己修改保存路径

## 4. heatmap.py
画热力图

- 修改params中的'weights'权重路径
- 修改最后一行输入中的图片路径

## 5. draw_metrics.py
画柱状图和散点图

- 修改exp_dir_list，该列表为多次实验分别保存的输出路径，所以训练前修改配置文件中的保存路径十分重要！！！路径相同的话新的实验结果就会覆盖旧的结果
- 修改exp_name，这个是最后柱状图显示的样例文字

## 6. visualize_featuremap.py
保存每一层的特征图

- 修改权重路径
- 修改图片路径

它会自动在目录中新建一个文件存储每层的特征图

## 7. add_cloud.py
在图像中增加三个不同程度的云层效果

- 修改图片路径，显示结果窗口需要手动保存

## 8. data_enhance_demo
对一张图片进行四种数据增强方式的demo