from ultralytics import YOLO
 
# Load a model
# model = YOLO("yolov8n-obb.pt")  # load an official model
model = YOLO("./best.pt")  # load a custom model
 
# Predict with the model
results = model("./demo.jpg", save=True)  # predict on an image