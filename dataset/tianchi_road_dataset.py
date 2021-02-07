from .base_dataset import BaseDataset
import numpy as np
import torch


class TianchiRoadDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir, img_suffix='.tif', mask_suffix='.png'):
        super(TianchiRoadDataset, self).__init__(imgs_dir, masks_dir, img_suffix, mask_suffix)

    def transform(self, img: np.ndarray, mask: np.ndarray):
        img = img[:3, :, :]
        img = torch.FloatTensor(img) / 255
        mask = torch.FloatTensor(mask) / 255
        return img, mask
