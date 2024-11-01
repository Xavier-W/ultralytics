import sys
sys.path.append("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics")

from ultralytics import YOLO
import cv2
import numpy as np
# Load the exported ONNX model
onnx_model = YOLO("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/yolov8n-obb.onnx")

# Run inference
result = onnx_model.predict(source='/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/8d680db1ea0d468cbc8beda94014f804.png', save=True, imgsz=1024)

polypoints = result[0].obb.xyxyxyxy
for points in polypoints:
    # 将 float32 的点转换为整数，并构造成正确的形状
    draw_points = np.array([[int(point[0]), int(point[1])] for point in points], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(result[0].orig_img, [draw_points], True, (0, 255, 0), 2)
cv2.imshow("res", result[0].orig_img)
cv2.waitKey(0)
print(result)
