from ultralytics import YOLO
import os
import cv2

# Load a model
model = YOLO('./ultralytics/landslide_exp3/weights/best.pt')  # pretrained YOLOv8n model
# model = YOLO('./yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model("./ddfgf10183_aug_2.png")  # return a list of Results objects

# results = model.predict("./20221124160707179_LGWEF6A75MH250240_0_0_0_1__130.jpg", save=True)
image_path = "./5.png"
results = model.predict(image_path, save=False)

# for i in 
x0,y0,x1,y1 = [int(i) for i in results[0].boxes.xyxy.cpu().numpy().tolist()[0]]
confidence = round(float(results[0].boxes.conf),2)
img_data = cv2.imread(image_path)
cv2.rectangle(img_data, (x0,y0), (x1,y1), (0,0,255), 2)
cv2.putText(img_data, str(confidence), (x0,y0-2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv2.imwrite("./0_outputs/result_"+os.path.basename(image_path), img_data)
print()
# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk