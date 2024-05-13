import cv2
import numpy as np

# 读取图像
img = cv2.imread('./df007.png')

# 调整亮度的alpha系数
alpha = 2.0  # 举例，增加亮度
# 调整亮度
img_brighter = cv2.convertScaleAbs(img, alpha=alpha)
# 显示原始图像和调整后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Brighter Image', img_brighter)


# 将BGR图像转换为HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 分离HSV通道
h, s, v = cv2.split(hsv_image)
# 调整饱和度
# 饱和度因子factor的范围通常是0到2
# 0表示完全去饱和（灰度图像），1表示保持不变，大于1表示增加饱和度
factor = 1.2
s = s * factor
s = ((np.max(s)-np.min(s))/(s-np.min(s)+0.000001)*255).astype(np.uint8)
# 合并HSV通道
hsv_image = cv2.merge([h, s, v])
# 将HSV图像转换回BGR
adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
# 显示调整后的图像
cv2.imshow('Adjusted Saturation', adjusted_image)


def add_gaussian_noise(image, mean=0, stddev=25):
    """
    添加高斯噪声到图像中
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, stddev, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy
# 添加高斯噪声
gaussian_noisy_image = add_gaussian_noise(img)
# 显示添加噪声后的图像
cv2.imshow('Gaussian Noisy Image', gaussian_noisy_image.astype(np.uint8))



def random_pan(image, max_pan=50):
    """
    对图像进行随机平移
    """
    # 获取图像的尺寸
    height, width = image.shape[:2]
    # 生成随机平移量
    x_pan = np.random.randint(-max_pan, max_pan)
    y_pan = np.random.randint(-max_pan, max_pan)
    # 定义平移矩阵
    M = np.float32([[1, 0, x_pan], [0, 1, y_pan]])
    # 进行仿射变换
    pan_image = cv2.warpAffine(image, M, (width, height))
    return pan_image
# 添加随机平移
panned_image = random_pan(img)
# 显示平移后的图像
cv2.imshow('Panned Image', panned_image)


cv2.waitKey(0)
# cv2.destroyAllWindows()