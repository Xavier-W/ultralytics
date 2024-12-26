# -*- coding:utf-8 -*
import os
import random
import os
import shutil
from tqdm import tqdm

dataset_dir = "./project/dataset"
img_dir = "./project/dataset"
txt_dir = "./project/dataset"
train_ratio = 0.9

train_img_dir = os.path.join(dataset_dir, "train", "images")
train_label_dir = os.path.join(dataset_dir, "train", "labels")
val_img_dir = train_img_dir.replace("train", "val")
val_label_dir = train_label_dir.replace("train", "val")
for data_dir in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

file_names = [os.path.splitext(i)[0] for i in os.listdir(txt_dir) if os.path.splitext(i)[-1] in ['.txt']]
random.shuffle(file_names)
train_num, val_num = 0,0
for file_name in tqdm(file_names):
    txt_path = os.path.join(txt_dir, file_name+'.txt')
    img_path = os.path.join(img_dir, file_name+'.jpg')
    if random.random() < train_ratio:
        shutil.copy(txt_path, os.path.join(train_label_dir, file_name+'.txt'))
        shutil.copy(img_path, os.path.join(train_img_dir, file_name+'.jpg'))
        train_num +=1
    else:
        shutil.copy(txt_path, os.path.join(val_label_dir, file_name+'.txt'))
        shutil.copy(img_path, os.path.join(val_img_dir, file_name+'.jpg'))
        val_num += 1

print("数据集划分完成： 总数量：",len(file_names)," 训练集数量：",train_num," 验证集数量：",val_num)

