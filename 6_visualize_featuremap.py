from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
model = YOLO("/home/maoyc/wxn/landslide/ultralytics/ultralytics/landslide_exp3/weights/best.pt")  # 模型文件路径

results = model("/home/maoyc/wxn/landslide/ultralytics/df007.png", visualize=True)  # 要预测图片路径和使用可视化
