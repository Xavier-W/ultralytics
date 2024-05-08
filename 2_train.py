from ultralytics import YOLO

# Load a model
# model = YOLO("/home/maoyc/wxn/landslide/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
model = YOLO("/home/maoyc/wxn/landslide/ultralytics/ultralytics/cfg/models/v8/yolov8-ghost-p6.yaml")  # load a pretrained model (recommended for training)

# Use the model
model.train(data='/home/maoyc/wxn/landslide/ultralytics/ultralytics/cfg/datasets/labdslide.yaml', cfg="/home/maoyc/wxn/landslide/ultralytics/ultralytics/cfg/default.yaml")  # train the model