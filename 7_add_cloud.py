import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# 读取图像
img_path = r"/mnt/hdd0/xnwu/code/wxn/landslide/ultralytics_landslide/df007.png"
img = cv2.imread(img_path)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 创建数据增强器
aug1 = iaa.CloudLayer(intensity_mean=240, intensity_freq_exponent=-2.5, intensity_coarse_scale=8, alpha_min=0.2, alpha_multiplier=0.4,
                      alpha_size_px_max=200, alpha_freq_exponent=-3.0, sparsity=1.2, density_multiplier=1.0, seed=None, name=None,
                      random_state="deprecated", deterministic="deprecated")
aug2 = iaa.CloudLayer(intensity_mean=240, intensity_freq_exponent=-2.5, intensity_coarse_scale=8, alpha_min=0.4, alpha_multiplier=0.4,
                      alpha_size_px_max=200, alpha_freq_exponent=-3.0, sparsity=1.2, density_multiplier=1.0, seed=None, name=None,
                      random_state="deprecated", deterministic="deprecated")
aug3 = iaa.CloudLayer(
                    intensity_mean=240, 
                    intensity_freq_exponent=-2.5, intensity_coarse_scale=4, alpha_min=0.6, alpha_multiplier=0.4,
                      alpha_size_px_max=200, alpha_freq_exponent=-3.0, sparsity=1.2, density_multiplier=1.0, 
                      seed=None, name=None,
                      random_state="deprecated", deterministic="deprecated"
                      )

# 对图像进行数据增强
Augmented_image1 = aug1(image=image)
Augmented_image2 = aug2(image=image)
Augmented_image3 = aug3(image=image)

# 展示原始图像和数据增强后的图像
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0][0].imshow(image)
axes[0][0].set_title("Original Image")
axes[0][1].imshow(Augmented_image1)
axes[0][1].set_title("Augmented Image1")
axes[1][0].imshow(Augmented_image2)
axes[1][0].set_title("Augmented Image2")
axes[1][1].imshow(Augmented_image3)
axes[1][1].set_title("Augmented Image3")
plt.show()


# import cv2
# import numpy as np

# img = cv2.imread("./df007.png")
# mask = cv2.imread("./mask.png")
# mask = cv2.resize(mask, (img.shape[1], img.shape[0]))


# rows = len(img)  # 图像像素行数
# cols = len(img[0])  # 图像像素列数
# M = np.float32([[1, 0, 60],  # 横坐标向右移动50像素
#                 [0, 1, 0]])  # 纵坐标向下移动100像素
# mask = cv2.warpAffine(mask, M, (cols, rows))
# # cv2.imshow("img", img)  # 显示原图
# cv2.imshow("dst", mask)  # 显示仿射变换效果



# show = img+mask*0.4
# show[show>255] = 255
# show[show<0] = 0
# show = show.astype(np.uint8)
# cv2.imshow('show',show)
# cv2.waitKey(0)