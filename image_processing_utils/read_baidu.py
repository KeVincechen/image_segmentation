import cv2
import numpy as np
from PIL import Image as PILImage
import os
from tqdm import tqdm

img_dir = 'H:/baidu\lab_train/lab_train/'
dst_dir = 'E:\workspace\sandiao\data\Multi-class/train_data/baidu\mask_train/'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
img_list = os.listdir(img_dir)

cmap = []
cmap.extend([0, 0, 0])
cmap.extend([224, 64, 64])  # 红色
cmap.extend([96, 96, 96])  # 灰色
cmap.extend([96, 160, 32])  # 浅绿色
cmap.extend([32, 64, 0])  # 深绿色
cmap.extend([192, 192, 64])  # 橙黄色
cmap.extend([128, 164, 164])  # 灰绿色
cmap.extend([32, 224, 224])  # 浅蓝色
cmap.extend([255, 255, 32])  # 金黄色

for file in tqdm(img_list):
    img_path = os.path.join(img_dir, file)
    dst_path = os.path.join(dst_dir, file)
    mask = PILImage.open(img_path)
    mask = np.array(mask) + 10
    mask[mask<10] = 0
    mask[mask==10] = 1
    mask[mask==11] = 3
    mask[mask==12] = 4
    mask[mask==13] = 7
    mask[mask==14] = 2
    mask[mask==15] = 5
    mask[mask>15] = 0
    mask = PILImage.fromarray(mask)
    mask.putpalette(cmap)
    mask.save(dst_path)
