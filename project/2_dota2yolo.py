import sys
sys.path.append("/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/")
from ultralytics.data.converter import convert_dota_to_yolo_obb


class_mapping = {
    "person": 0,
    "ship": 1,
    "dog": 2,
}
convert_dota_to_yolo_obb('/mnt/hdd0/xnwu/code/wxn/heatmap/ultralytics/project/datasets', class_mapping)