import os
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import shutil

class data_process():
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # self.total_image_dir = os.path.join(data_dir, 'images')
        # self.total_label_dir = os.path.join(data_dir, 'labels')
        # self.total_rect_dir = os.path.join(data_dir, 'rect')

        self.pos_dir = os.path.join(self.data_dir, "landslide")
        self.img_dir = os.path.join(self.pos_dir, "image")
        self.poly_dir = os.path.join(self.pos_dir, "polygon_coordinate")
        self.rect_dir = os.path.join(self.pos_dir, "rectangle_coordinate")

        self.neg_dir = os.path.join(self.data_dir, "non-landslide")
        self.neg_img_dir = os.path.join(self.neg_dir, "image")

        self.aug_img_dir = os.path.join(self.data_dir, "augimage")
        self.aug_rect_dir = os.path.join(self.data_dir, "augrect_coords")
        self.yolo_label_dir = os.path.join(self.data_dir, "yolo_label")

    # def integrate_data(self):
    #     self.poly2rect()

    #     img_list = os.listdir(self.img_dir)
    #     print("==>integrating positive data")
    #     for img_name in tqdm(img_list):
    #         img_path = os.path.join(self.img_dir, img_name)
    #         rect_path = os.path.join(self.rect_dir, os.path.splitext(img_name)[0]+'.txt')
    #         dst_img_path = os.path.join(self.total_image_dir, img_name)
    #         dst_rect_path = os.path.join(self.total_rect_dir, os.path.splitext(img_name)[0]+'.txt')
    #         shutil.copy(img_path, dst_img_path)
    #         shutil.copy(rect_path, dst_rect_path)
    #     print("==>integrating negative data")
    #     img_list = os.listdir(self.neg_img_dir)
    #     for img_name in img_list:
    #         img_path = os.path.join(self.img_dir, img_name)
    #         dst_img_path = os.path.join(self.total_image_dir, img_name)
    #         shutil.copy(img_path, dst_img_path)
        
    def limit_bound(self, x, bound):
        return min(max(x, 0), bound)
        
    def poly2rect(self):
        img_list = os.listdir(self.total_image_dir)
        for img_name in tqdm(img_list):
            poly_path = os.path.join(self.poly_dir, os.path.splitext(img_name)[0]+'.txt')
            rect_path = os.path.join(self.rect_dir, os.path.splitext(img_name)[0]+'.txt')

            f_poly = open(poly_path,'r')
            poly_content = f_poly.readlines()
            poly_content = poly_content[2:]
            poly_coords = []
            for line in poly_content:
                line = line.strip().split(' ')
                poly_coords.append((int(line[0]), int(line[1])))
            # 创建多边形
            poly = Polygon(poly_coords)
            # 获取外接矩形
            envelope = poly.envelope
            # 输出外接矩形的坐标
            # print(envelope.bounds)
            x0, y0, x1, y1 = [int(i) for i in envelope.bounds]

            f_rect = open(rect_path, 'w')
            rect_content = str(x0)+' '+str(y0)+' ' +str(x1)+' '+str(y1)+'\n'
            f_rect.write(rect_content)
            f_rect.close()

            # img_path = os.path.join(self.img_dir, img_name)
            # img_data = cv2.imread(img_path)
            # cv2.rectangle(img_data, (int(x0),int(y0)), (int(x1),int(y1)), (0,0,0), 2)
            # cv2.imshow('rect_label', img_data)
            # cv2.waitKey(0)
    
    def analyse_center(self, img_dir, rect_dir):
        center_x = []
        center_y = []
        img_list = os.listdir(img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(img_dir, img_name)
            rect_path = os.path.join(rect_dir, os.path.splitext(img_name)[0]+'.txt')
            f_rect = open(rect_path, 'r')
            rect_content = f_rect.readlines()
            x0, y0, x1, y1 = [int(i) for i in rect_content[0].strip().split(' ')]

            img_data = cv2.imread(img_path)
            h, w, c = img_data.shape

            center_x.append((x0+x1+1)/2/w)
            center_y.append((y0+y1+1)/2/h)
        # 使用scatter()函数绘制散点图
        plt.scatter(center_x, center_y, s=20, cmap='viridis', alpha=0.2)

        # 可选：添加标题和轴标签
        plt.title('Central coordinates')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.gca().set_aspect('equal', adjustable='box')
        # 设置横纵轴的显示区间为0到1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 显示图表
        plt.show()

    def analyse_boxwh(self, img_dir, rect_dir):
        width = []
        height = []
        img_list = os.listdir(img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(img_dir, img_name)
            rect_path = os.path.join(rect_dir, os.path.splitext(img_name)[0]+'.txt')
            f_rect = open(rect_path, 'r')
            rect_content = f_rect.readlines()
            x0, y0, x1, y1 = [int(i) for i in rect_content[0].strip().split(' ')]

            img_data = cv2.imread(img_path)
            h, w, c = img_data.shape

            width.append((x1-x0+1)/w)
            height.append((y1-y0+1)/h)
        # 使用scatter()函数绘制散点图
        plt.scatter(width, height, s=20, cmap='viridis', alpha=0.2)

        # 可选：添加标题和轴标签
        plt.title('width and height distribution of the boxes')
        plt.xlabel('width')
        plt.ylabel('height')
        plt.gca().set_aspect('equal', adjustable='box')
        # 设置横纵轴的显示区间为0到1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 显示图表
        plt.show()

    def data_aug(self, pos_augloop=7, neg_augloop=4): # 每张影像增强的数量
        if os.path.exists(self.aug_img_dir):
            shutil.rmtree(self.aug_img_dir)
            shutil.rmtree(self.aug_rect_dir)
        os.mkdir(self.aug_img_dir)
        os.mkdir(self.aug_rect_dir)
        new_bndbox = []

        # 图片数据增强
        seq = iaa.Sequential([
            # 改变标签文件的数据增强方式,有时需要重新标注    # 注意只要是带有旋转和平移性质的方式都要检查标签是否合适，做人工微调
            # 仿射变换
            # 包含：平移(Translation)、旋转(Rotation)、放缩(zoom)、错切(shear)。
            # 仿设变换通常会产生一些新的像素点,我们需要指定这些新的像素点的生成方法,这种指定通过设置cval和mode两个参数来实现。参数order用来设置插值方法。
            # 只有在mode设置为“constant”时，cval的值才有效。
            iaa.Affine(
                translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},  # 平移
                scale=(0.9, 1.1),  # 图像缩放为90%到110%之间
                # rotate=(-15, 15),  # 旋转±15度之间
                shear=(-10, 10),  # 错切±15度之间
                mode="symmetric"
            ),
            # 选择0到5种方法做变换
            iaa.SomeOf((1, 3),
                    [
                        # 改变标签文件的数据增强方式,有时需要重新标注
                        # 长变成原来的0.5到0.75的随机倍数，宽同理
                        iaa.Resize({"height": (0.8, 1), "width": (0.8, 1)}),

                        # 翻转只使用一个
                        iaa.OneOf([
                            # 镜像翻转
                            iaa.Fliplr(1),  # 对50%的图片进行水平镜像翻转
                            iaa.Flipud(1),  # 对50%的图片进行垂直镜像翻转
                            # 中心对称
                            iaa.Sequential([
                                iaa.Fliplr(1),  # 对50%的图片进行水平镜像翻转
                                iaa.Flipud(1),  # 对50%的图片进行垂直镜像翻转
                            ]),
                        ]),

                    # 不改变标签文件的数据增强方式
                        # 添加噪声只使用一个
                        iaa.OneOf([
                            # 增加高斯噪声
                            iaa.AdditiveGaussianNoise(
                                    loc=0, scale=(0.0, 0.05 * 255)
                                ),
                            # 为每个像素乘以一个值
                            iaa.MultiplyElementwise((0.9, 1.1)),
                            # 为每个像素加上一个值
                            iaa.AddElementwise((-20, 20)),
                            iaa.imgcorruptlike.ShotNoise(severity=1),  # 散粒噪声
                            iaa.imgcorruptlike.ImpulseNoise(severity=1),  # 脉冲噪声
                            iaa.imgcorruptlike.SpeckleNoise(severity=1),  # 斑点噪声
                        ]),

                        # 锐化效果和浮雕效果只使用一个
                        iaa.OneOf([
                            # 图像锐化(image sharpening)
                            # 是补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰，这种滤波方法提高了地物边缘与周围像元之间的反差。
                            # 图像锐化是为了突出图像上地物的边缘、轮廓，或某些线性目标要素的特征。
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),
                            # 浮雕效果，与锐化类似，但是具有压纹效果
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                        ]),

                        # 改变亮度
                        iaa.Multiply((0.8, 1.2)),
                        # 更改对比度
                        iaa.LinearContrast((0.8, 1.2)),

                        # 滤波（模糊）只使用一个
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),  # 高斯滤波
                            iaa.AverageBlur(k=(2, 5)),  # 均值滤波，k指核的大小
                            iaa.MedianBlur(k=(3, 7)),  # 中值滤波
                            iaa.imgcorruptlike.DefocusBlur(severity=1)  # 散焦模糊
                        ]),

                        # 随机丢弃像素
                        iaa.Dropout((0.05, 0.1), per_channel=False),

                        # 颜色变换只使用一个
                        iaa.OneOf([
                            # 色相调整
                            iaa.AddToHue((-10, 0)),
                            # 饱和度调整
                            iaa.AddToSaturation((-10, 10)),
                            # 饱和度以及色相调整
                            iaa.AddToHueAndSaturation((-10, 10), per_channel=True),
                            # 变成灰度图
                            iaa.Grayscale(alpha=(0.4, 0.6), from_colorspace='RGB'),
                            # 改变亮度，并加上颜色滤镜
                            iaa.Multiply((0.8, 1.2), per_channel=0.1)
                        ]),

                        # 自然环境变化只使用一个
                        iaa.OneOf([
                            # 下雨
                            iaa.Sequential([
                                iaa.Rain(drop_size=(0.025, 0.05), speed=(0.25, 0.05)),  # 雨
                                iaa.imgcorruptlike.Spatter(severity=1),  # 溅 123水滴、45泥
                                iaa.MotionBlur(k=3),  # 运动模糊
                            ]),

                            # 下雪
                            iaa.imgcorruptlike.Snow(severity=1),

                            # 起雾
                            iaa.Clouds(),
                        ]),

                        # 限制对比度自适应直方图均衡(CLAHE算法)，本算法与普通的自适应直方图均衡不同地方在于对比度限幅，图像对比度会更自然。
                        iaa.CLAHE(clip_limit=(1, 5)),

                        # 不作任何变换
                        iaa.Noop(),
                    ],
                    random_order=True  # 以随机的方式执行上述扩充
                    )
        ], random_order=True)
        # 得到当前运行的目录和目录当中的文件，其中sub_folders可以为空

        print("==>augmenting positive data")
        img_list = os.listdir(self.img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(self.img_dir, img_name)
            rect_path = os.path.join(self.rect_dir, os.path.splitext(img_name)[0]+'.txt')
            f_rect = open(rect_path, 'r')
            rect_content = f_rect.readlines()
            bndbox = [int(i) for i in rect_content[0].strip().split(' ')]

            img_data = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            for epoch in range(pos_augloop):
                seq_det = seq.to_deterministic()  # 固定变换序列,之后就可以先变换图像然后变换关键点,这样可以保证两次的变换完全相同
                bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[0], y1=bndbox[1], x2=bndbox[2], y2=bndbox[3]),
                    ], shape=img_rgb.shape)

                bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                # boxes_img_aug_list.append(bbs_aug)

                # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                new_bndbox = [int(bbs_aug.bounding_boxes[0].x1),
                                        int(bbs_aug.bounding_boxes[0].y1),
                                        int(bbs_aug.bounding_boxes[0].x2),
                                        int(bbs_aug.bounding_boxes[0].y2)]
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img_rgb])[0]
                aug_name = os.path.splitext(img_name)[0] + "_aug_" + str(epoch)
                save_path = os.path.join(self.aug_img_dir,  aug_name + '.png')
                Image.fromarray(image_aug).save(save_path)

                aug_rect_path = os.path.join(self.aug_rect_dir, aug_name+'.txt')
                f_aug_rect = open(aug_rect_path, 'w')
                new_bndbox = [str(i) for i in new_bndbox]
                aug_rect_content = new_bndbox[0]+' '+new_bndbox[1]+' '+new_bndbox[2]+' '+new_bndbox[3]+'\n'
                f_aug_rect.write(aug_rect_content)
                f_aug_rect.close()

        print("==>augmenting negative data")
        img_list = os.listdir(self.neg_img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(self.neg_img_dir, img_name)
            img_data = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            for epoch in range(neg_augloop):
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img_rgb])[0]
                aug_name = os.path.splitext(img_name)[0] + "_aug_" + str(epoch)
                save_path = os.path.join(self.aug_img_dir,  aug_name + '.png')
                Image.fromarray(image_aug).save(save_path)


    def label2yolo(self):
        print("==>translating rect_txt to yolo_txt")
        if os.path.exists(self.yolo_label_dir):
            shutil.rmtree(self.yolo_label_dir)
        os.mkdir(self.yolo_label_dir)

        rect_list = os.listdir(self.aug_rect_dir)
        for rect_name in tqdm(rect_list):
            rect_path = os.path.join(self.aug_rect_dir, rect_name)
            yolo_label_path = os.path.join(self.yolo_label_dir, rect_name)
            img_path = os.path.join(self.aug_img_dir, os.path.splitext(rect_name)[0]+'.png')
            img_data = cv2.imread(img_path)
            h, w, c = img_data.shape

            f_rect = open(rect_path, 'r')
            rect_content = f_rect.readlines()
            x0, y0, x1, y1 = [int(i) for i in rect_content[0].strip().split(' ')]
            center_x = self.limit_bound((x0+x1)/2/w, 1.0)
            center_y = self.limit_bound((y0+y1)/2/h, 1.0)
            box_w = self.limit_bound((x1-x0+1)/w, 1.0)
            box_h = self.limit_bound((y1-y0+1)/h, 1.0)

            f_yolo_label = open(yolo_label_path, 'w')
            yolo_content = '0'+' '+str(center_x)+' '+str(center_y)+' '+str(box_w)+' '+str(box_h)
            f_yolo_label.write(yolo_content)
            f_yolo_label.close()

    def dataset2yolo(self):
        self.label2yolo()

        print("==>generting yolo datasets")
        yolo_dataset_dir = os.path.join(self.data_dir, 'yolo_dataset')
        if os.path.exists(yolo_dataset_dir):
            shutil.rmtree(yolo_dataset_dir)
        os.mkdir(yolo_dataset_dir)

        images_dir = os.path.join(yolo_dataset_dir, 'images')
        labels_dir = os.path.join(yolo_dataset_dir, 'labels')
        os.mkdir(images_dir)
        os.mkdir(labels_dir)

        images_train_dir = os.path.join(images_dir, 'train')
        images_val_dir = os.path.join(images_dir, 'val')
        labels_train_dir = os.path.join(labels_dir, 'train')
        labels_val_dir = os.path.join(labels_dir, 'val')
        os.mkdir(images_train_dir)
        os.mkdir(images_val_dir)
        os.mkdir(labels_train_dir)
        os.mkdir(labels_val_dir)

        img_list = os.listdir(self.aug_img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(self.aug_img_dir, img_name)
            label_name = os.path.splitext(img_name)[0]+'.txt'
            label_path = os.path.join(self.yolo_label_dir, label_name)
            if not os.path.exists(label_path):
                f = open(label_path,'w')
                f.close()

            if np.random.uniform(0,1)>0.9:
                dst_img_path = os.path.join(images_val_dir, img_name)
                dst_label_path = os.path.join(labels_val_dir, label_name)
                shutil.copy(img_path, dst_img_path)
                shutil.copy(label_path, dst_label_path)
            else:
                dst_img_path = os.path.join(images_train_dir, img_name)
                dst_label_path = os.path.join(labels_train_dir, label_name)
                shutil.copy(img_path, dst_img_path)
                shutil.copy(label_path, dst_label_path)

    def draw_sample(self):
        img_list = os.listdir(self.aug_img_dir)
        for img_name in tqdm(img_list):
            img_path = os.path.join(self.aug_img_dir, img_name)
            rect_name = os.path.splitext(img_name)[0]+'.txt'
            rect_path = os.path.join(self.aug_rect_dir, rect_name)

            img_path = os.path.join(self.aug_img_dir, os.path.splitext(rect_name)[0]+'.png')
            img_data = cv2.imread(img_path)
            h, w, c = img_data.shape

            f_rect = open(rect_path, 'r')
            rect_content = f_rect.readlines()
            x0, y0, x1, y1 = [int(i) for i in rect_content[0].strip().split(' ')]

            cv2.rectangle(img_data, (x0,y0), (x1,y1), ())





if __name__ == "__main__":
    data_dir = "/mnt/hdd0/xnwu/data/Bijie-landslide-dataset/"

    process = data_process(data_dir)

    # process.analyse_center()
    process.data_aug() # 每张图片扩充的数量
    # process.analyse_center(process.aug_img_dir, process.aug_rect_dir)
    # process.analyse_boxwh(process.aug_img_dir, process.aug_rect_dir)
    process.dataset2yolo()


