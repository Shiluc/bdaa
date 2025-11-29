import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# --- 配置 ---
IMAGE_PATH = 'test.JPG' # 请替换成你自己的图片路径
N_COLORS = 5 # 你想把图片简化成几种颜色？

# --- 1. 读取和预处理图像 ---
# OpenCV 读取的是 BGR，matplotlib 需要 RGB，需转换
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"错误：找不到图片 {IMAGE_PATH}，请确保文件存在。")
    exit()
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 显示原图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

# 将图像数据重塑为像素列表。
# 原图形状是 (高度, 宽度, 3)，我们需要变成 (像素总数, 3)
# 这样每一行代表一个像素点的 RGB 值，作为 K-Means 的输入数据点。
pixels = image_rgb.reshape((-1, 3))

print(f"正在分析图像中的 {pixels.shape[0]} 个像素点...")
print(f"正在聚类出主要的 {N_COLORS} 种颜色，请稍候...")

# --- 2. 核心：K-Means 聚类 ---
# 这是一个无监督学习过程，机器自己寻找颜色的聚集中心
kmeans = KMeans(n_clusters=N_COLORS, random_state=42, n_init='auto')
kmeans.fit(pixels)

# 获取聚类中心（也就是找到的那 K 个最具代表性的颜色）
colors = kmeans.cluster_centers_
# 获取每个像素被分配到了哪个中心
labels = kmeans.labels_

print("主要颜色 (RGB) 如下:\n", colors.astype(int))

# --- 3. 重建艺术图像 ---
# 用每个像素所属的聚类中心的颜色，替换掉原来的颜色
new_pixels = colors[labels]
# 将形状还原回图像的原始尺寸 (高度, 宽度, 3)
new_image = new_pixels.reshape(image_rgb.shape).astype(np.uint8)

# 显示处理后的图像
plt.subplot(1, 2, 2)
plt.title(f"Artistic Style ({N_COLORS} colors)")
plt.imshow(new_image)
plt.axis('off')

plt.tight_layout()
print("处理完成！弹窗显示对比图。")
plt.show()