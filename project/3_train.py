from ultralytics import YOLO
 
model_yaml_path = "./ultralytics/cfg/models/v8/yolov8-obb.yaml"
#数据集配置文件
data_yaml_path = './ultralytics/cfg/datasets/coco128.yaml'
#预训练模型
pre_model_name = 'yolov8s-obb.pt'
 
def main():
    model = YOLO(model_yaml_path).load(pre_model_name)  # build from YAML and transfer weights
 
    model.train(data=data_yaml_path,
                epochs=500,
                imgsz=640,
                batch=6,
                workers=5,
                name="train_obb/exp")
 
if __name__ == '__main__':
    main()
 
# yolo obb train data=data/hat.yaml model=yolov8s-obb.pt epochs=200 imgsz=640 device=0