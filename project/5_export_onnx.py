from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolov8n.onnx")

# Run inference
result = onnx_model.predict(source='<your img path>', save=True, imgsz=480)
print(result)
# 其中imgsz表示图片尺寸
