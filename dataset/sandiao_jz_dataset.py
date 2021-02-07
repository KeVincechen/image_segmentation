from .base_dataset import BaseDataset
import numpy as np
import torch


class SandiaoJzDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir):
        super(SandiaoJzDataset, self).__init__(imgs_dir, masks_dir)

    def transform(self, img: np.ndarray, mask: np.ndarray):
        if len(mask.shape) == 3:
            mask = mask[0, :, :]
        img = torch.FloatTensor(img) / 255
        mask = torch.FloatTensor(mask).unsqueeze(0) / 255
        return img, mask
