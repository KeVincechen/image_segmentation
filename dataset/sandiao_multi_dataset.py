from dataset import BaseDataset
import numpy as np
import torch
from torch.utils.data import DataLoader


class SandiaoMultiDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir):
        super(SandiaoMultiDataset, self).__init__(imgs_dir, masks_dir)

    def transform(self, img: np.ndarray, mask: np.ndarray):
        if len(mask.shape) == 3:
            mask = mask[0, :, :]
        img = torch.FloatTensor(img) / 255
        mask = torch.LongTensor(mask) // 31
        return img, mask


if __name__ == '__main__':
    dataset = SandiaoMultiDataset(
        r'F:\company_project\cangqiongshuma\sandiao\data\Multi-class\train_data\images',
        r'F:\company_project\cangqiongshuma\sandiao\data\Multi-class\train_data\labels'
    )
    data_loader = DataLoader(dataset)
    print(iter(data_loader).__next__())
