import sys
sys.path.append("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics")
from ultralytics import YOLO
 
# Load a model
# model = YOLO("yolov8n-obb.pt")  # load an official model
model = YOLO("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/yolov8n-obb.pt")  # load a custom model
 
# Predict with the model
results = model("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/1.jpeg", save=True)  # predict on an image