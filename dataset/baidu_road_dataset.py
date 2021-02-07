from .base_dataset import BaseDataset
import numpy as np
import torch


class BaiduRoadDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir, img_suffix='.jpg', mask_suffix='.png'):
        super(BaiduRoadDataset, self).__init__(imgs_dir, masks_dir, img_suffix, mask_suffix)

    def transform(self, img: np.ndarray, mask: np.ndarray):
        img = torch.FloatTensor(img) / 255
        mask = torch.FloatTensor(mask).unsqueeze(0) / 255
        return img, mask
