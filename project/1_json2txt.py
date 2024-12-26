import json
import os
import glob
import cv2
import shutil
from tqdm import tqdm
import numpy as np
 
def convert_label(json_dir, save_dir, classes):
    json_names = os.listdir(json_dir)
    classes = classes.split(',')
 
    for json_name in tqdm(json_names):
        if os.path.splitext(json_name)[-1] not in ['.json']:
            continue
        path = os.path.join(json_dir, json_name)
        with open(path, 'r') as load_f:
            json_dict = json.load(load_f)
        w, h = json_dict['info']['width'], json_dict['info']['height']
        # save txt path
        txt_path = os.path.join(save_dir, json_name.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')

        new_img_path = os.path.join(save_dir, json_name.replace('.json', '.jpg'))
        shutil.copy(os.path.join(json_dir, json_name.replace('.json', '.jpg')), new_img_path)
 
        for shape_dict in json_dict['objects']:
            label = shape_dict['category']
            label_index = classes.index(label)
            points = shape_dict['segmentation']
            points_nor_list = []
            for point in points:
                points_nor_list.append(point[0] / w)
                points_nor_list.append(point[1] / h)
 
            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)
 
            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)


def check_labels(txt_dir, images_dir):
    txt_names = os.listdir(txt_dir)
    for txt_name in txt_names:
        if os.path.splitext(txt_name)[-1] not in ['.txt']:
            continue
        filename = os.path.splitext(txt_name)[0]

        img_path = os.path.join(images_dir, filename + ".jpg")

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        file_handle = open(os.path.join(txt_dir, txt_name))
        cnt_info = file_handle.readlines()
        new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

        color_map = {"0": (0, 255, 255)}
        for new_info in new_cnt_info:
            print(new_info)
            s = []
            for i in range(1, len(new_info), 2):
                b = [float(tmp) for tmp in new_info[i:i + 2]]
                s.append([int(b[0] * width), int(b[1] * height)])
            cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0]))
        cv2.namedWindow('img2', 0)
        cv2.imshow('img2', img)
        if cv2.waitKey(0) == 27:
            break
 
if __name__ == "__main__":
 
    json_dir = './sam/example/images/'
    save_dir = './ultralytics/project/dataset'
    classes = '__background__,plate,table,cake,knife,fence,person,cup,fork,giraffe'
 
    convert_label(json_dir, save_dir, classes)
    # check_labels(save_dir, save_dir)