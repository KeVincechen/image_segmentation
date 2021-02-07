import os
import sys
from os import listdir
from torch.utils.data import Dataset
from osgeo import gdal
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_suffix='.tif', mask_suffix='.tif'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.list_file_names = list(map(lambda file: file.split('.')[0], listdir(imgs_dir)))

    def __len__(self):
        return len(self.list_file_names)

    def __getitem__(self, i):
        file_name = self.list_file_names[i]
        img_file_path = os.path.join(self.imgs_dir, file_name + self.img_suffix)
        mask_file_path = os.path.join(self.masks_dir, file_name + self.mask_suffix)
        img = self.read_image(img_file_path)
        mask = self.read_image(mask_file_path)
        return self.transform(img, mask)

    def transform(self, img: np.ndarray, mask: np.ndarray):
        pass

    def read_image(self, image_path):
        dataset = gdal.Open(image_path)
        if not dataset:
            sys.exit(f'{image_path} 图片读取错误！')
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        img_array = dataset.ReadAsArray(0, 0, width, height)
        return img_array
