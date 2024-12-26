'''
实例分割训练
'''
import sys
sys.path.append("/ultralytics")
from ultralytics import YOLO

#train
model = YOLO('./ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml').load('yolov8n-seg.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='./ultralytics/ultralytics/cfg/datasets/seg20241226.yaml', epochs=150, imgsz=160,batch=1, workers=0)

