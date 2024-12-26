import sys
sys.path.append("/ultralytics")
from ultralytics import YOLO

# Load a model
model = YOLO("./ultralytics/runs/segment/train8/weights/best.pt")  # load a custom model

# Predict with the model
results = model("./ultralytics/project/dataset/000000000113.jpg", save=True)  # predict on an image