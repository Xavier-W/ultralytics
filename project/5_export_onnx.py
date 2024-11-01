import sys
sys.path.append("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics")

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/yolov8n-obb.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'

