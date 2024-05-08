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
import io

plt.rcParams["figure.figsize"] = [3.0, 3.0]

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

img = cv2.imread('/home/maoyc/wxn/landslide/ultralytics/df007.png')
cv2.imshow('origin', img)
img = cv2.resize(img, (320, 320))
cv2.imshow('resize', img)
rgb_img = img.copy()
img = np.float32(img) / 255

model = YOLO('/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp3/weights/best.pt') 
target_layers =[model.model.model[-1].cv3[-1]]
cam = EigenCAM(model, target_layers, task='od')

grayscale_cam = cam(rgb_img)[0, :, :]
heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
cv2.imshow("heatmap", heatmap)
cv2.waitKey(0)
# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False)
# plt.imshow(cam_image)
# plt.show()