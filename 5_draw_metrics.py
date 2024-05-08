import os
import pandas
import csv
import matplotlib.pyplot as plt
import numpy as np
import math


exp_dir_list = [
    "/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp3",
    "/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp4",
    "/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp5"
]
exp_name = ["yolov8n", "yolov8n-ghost", "yolov8n-P6-ghost"]

show_data = []

for exp_dir in exp_dir_list:
    weight_path = os.path.join(exp_dir, 'weights', 'best.pt')
    weight = os.stat(weight_path)
    weight_size = weight.st_size/1024/1024

    csv_path = os.path.join(exp_dir, "results.csv")
    data = pandas.read_csv(csv_path)
    # data1 = csv.reader(csv_path)
    show_data.append([round(float(data[data.columns[i]].tail(1))*100,1) for i in range(4,8)]+[round(weight_size,1)])

# plt.figure(figsize=(12, 9))
# n = 5
# X = np.arange(n) + 1.0
# width = 0.2
# # width:柱的宽度
# exp1 = show_data[0]
# exp2 = show_data[1]
# exp3 = show_data[2]

# plt.bar(X - width, exp1, width=width, facecolor='lightskyblue', label=exp_name[0])

# plt.bar(X, exp2, width=width, facecolor='b', label=exp_name[1])

# plt.bar(X + width, exp3, width=width, facecolor='yellowgreen', label=exp_name[2])

# # x_labels = ["Precision", "Recall", "MAP50"]
# x_labels = ["Precision(%)", "Recall(%)", "MAP50(%)", "MAP5095(%)", "Weights/MB"]
# plt.xticks(X, x_labels)


# # 显示图例
# plt.legend()
# for x, y in zip(X, exp1):
#     plt.text(x - width, y + 0.005, '%.1f' % y, ha='center', va='bottom')

# for x, y in zip(X, exp2):
#     plt.text(x, y + 0.005, '%.1f' % y, ha='center', va='bottom')

# for x, y in zip(X, exp3):
#     plt.text(x + width, y + 0.005, '%.1f' % y, ha='center', va='bottom')

# # plt.show()
# plt.savefig('./0_outputs/comparision_metrics.png')
# print("saved to ./0_outputs/comparision_metrics.png!")




# show_data = np.array(show_data)
# map50_data = show_data[:,2]
# weight_size = show_data[:,-1]
# r_data = np.sqrt(map50_data**2+(6-weight_size)**2)*5
# plt.figure()

# plt.scatter(weight_size[0], map50_data[0], s=r_data[0], cmap='viridis', zorder=1)#, alpha=0.8)
# plt.scatter(weight_size[1], map50_data[1], s=r_data[1], cmap='viridis', zorder=1)#, alpha=0.8)
# plt.scatter(weight_size[2], map50_data[2], s=r_data[2], cmap='viridis', zorder=1)#, alpha=0.8)

# # 可选：添加标题和轴标签
# # plt.title('Central coordinates')
# plt.xlabel('Weight/MB')
# plt.ylabel('MAP50(%)')
# # plt.gca().set_aspect('equal', adjustable='box')
# # 设置横纵轴的显示区间为0到1
# plt.xlim(3, 6)
# plt.ylim(0, 100)
# plt.grid(zorder=0, alpha=0.3)
# k=0
# for i, j in zip(weight_size, map50_data):
#     plt.text(i, j-8, exp_name[k], ha='center', va='bottom')
#     k=k+1

# # 显示图表
# # plt.show()
# plt.savefig('./0_outputs/comparision_weight_map.png')
# print("saved to ./0_outputs/comparision_weight_map.png!")


from thop import profile
from ultralytics import YOLO
import torch
model = YOLO('/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp3/weights/best.pt')  # pretrained YOLOv8n model
input1 = torch.randn(1, 3, 320, 320) 
flops, params = profile(model, inputs=(input1, )) 
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
