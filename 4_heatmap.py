import ultralytics
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import torchvision.transforms as transforms
from PIL import Image
import os

plt.rcParams["figure.figsize"] = [3.0, 3.0]

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

img_path = './5.png'
origin_img = cv2.imread(img_path)
# cv2.imshow('origin', img)
img = cv2.resize(origin_img, (320, 320))
cv2.imshow('resize', img)
rgb_img = img.copy()
img = np.float32(img) / 255

model = YOLO('./ultralytics/landslide_exp3/weights/best.pt') 
target_layers =[model.model.model[-3]]
cam = EigenCAM(model, target_layers, task='od')

grayscale_cam = cam(rgb_img)[0, :, :]
heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
# cv2.imshow("heatmap", heatmap)
# cv2.waitKey(0)
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False)
plt.imshow(cam_image)
# plt.show()

results = model.predict(img_path, save=False)
x0,y0,x1,y1 = [int(i) for i in results[0].boxes.xyxy.cpu().numpy()[0].tolist()]
confidence = round(float(results[0].boxes.conf[0]),2)
cam_image = cv2.resize(cam_image, (origin_img.shape[1], origin_img.shape[0]))
cv2.rectangle(cam_image, (x0,y0), (x1,y1), (0,0,255), 2)
cv2.putText(cam_image, str(confidence), (x0,y0-2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv2.imwrite("./0_outputs/heatmap_{}".format(os.path.basename(img_path)), cam_image)