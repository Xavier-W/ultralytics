import numpy as np

# 参数初始化
u, v = 1080, 0  # 目标像素坐标
D_h = 18  # 水平距离（单位：米）
H_c = 3  # 相机离地高度（单位：米）
theta = np.radians(5)  # 相机俯仰角（单位：弧度）
fx, fy = 1747.2, 1736.2  # 焦距
cx, cy = 1073.7, 1921.4  # 主点坐标

# 内参矩阵
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

# # 从像素坐标到归一化相机坐标
# x_c = (u - cx) / fx
# y_c = (v - cy) / fy

# 计算归一化坐标
pixel_coords = np.array([u, v, 1])
normalized_coords = np.linalg.inv(K).dot(pixel_coords)
x_c, y_c, _ = np.linalg.inv(K).dot(pixel_coords)

# 考虑俯仰角的旋转矩阵
R = np.array([[1, 0, 0],
              [0, np.cos(theta), -np.sin(theta)],
              [0, np.sin(theta), np.cos(theta)]], dtype=np.float32)

# 相机坐标
camera_coords = np.array([x_c, y_c, 1], dtype=np.float32)
world_coords = R @ camera_coords

# 使用水平距离计算世界坐标
x_w = world_coords[0] * D_h
y_w = world_coords[1] * D_h
z_w = world_coords[2] * D_h

# 计算目标离地高度
H_t = H_c - y_w

print(f"目标的离地高度为: {H_t:.2f} 米")
