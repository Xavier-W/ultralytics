from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("/home/maoyc/wxn/landslide/ultralytics/yolov8n.pt")   # YOLOv8 custom/pretrained model

# im0 = cv2.imread("/home/maoyc/wxn/landslide/ultralytics/20221124160707179_LGWEF6A75MH250240_0_0_0_1__130.jpg")  # path to image file
# h, w = im0.shape[:2]  # image height and width

# # Heatmap Init
# heatmap_obj = heatmap.Heatmap()
# heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
#                      imw=w,
#                      imh=h,
#                      view_img=True,
#                      shape="circle",
#                      classes_names=model.names)

# results = model.track(im0, persist=True)
# im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
# cv2.imwrite("/home/maoyc/wxn/landslide/ultralytics/ultralytics_output.png", im0)

import cv2
import numpy as np
import torch

# 加载YOLOv8模型
# model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)

# 读取图像
image = cv2.imread('/home/maoyc/wxn/landslide/ultralytics/20221124160707179_LGWEF6A75MH250240_0_0_0_1__130.jpg')

# 预测
results = model(image)

# gradcam
normalized_mask = torch.nn.functional.softmax(pr, dim=0).cpu()
plaque_category = 1     ## 修改要查看哪个类别的热力图
plaque_mask = normalized_mask[:,:,:].argmax(axis=0).detach().cpu().numpy()
plaque_mask_float = np.float32(plaque_mask == plaque_category)
print(self.net)
target_layers = [self.net.module.final]      ## 修改要看哪一层的热力图，通过打印出的model结构按照示例格式填入
targets = [SemanticSegmenttationTarget(plaque_category, plaque_mask_float)]

torch.set_grad_enabled(True)

cam = GradCAM(self.net, target_layers)
grayscalse_cam = cam(input_tensor=images, targets=targets)[0,:]
cam_image = show_cam_on_image(origin_image, grayscalse_cam, use_rgb=True)

img = Image.fromarray(cam_image)
img.save("../datasets/heatmap1.png")        ## 修改热力图的保存路径
#####################################

# 获取热力图
heatmap = results.cam()[0]

# 显示热力图
cv2.imshow('Heatmap', heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()
