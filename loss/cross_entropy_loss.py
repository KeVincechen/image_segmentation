import os
import cv2
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class MyCrossEntropyLoss:
    def __call__(self, logits, y_true, weight=None):
        '''
        根据模型最后输出的通道数，自动选择相应的交叉熵损失函数。
        :param num_output_channels: 输出图的通道数
        :return:
        '''
        out_channels = logits.shape[1]
        if out_channels == 1:  # 输出为1通道，是二分类任务
            loss = F.binary_cross_entropy_with_logits(logits, y_true, weight)
        else:  # 输出为多通道，多分类任务。
            loss = F.cross_entropy(logits, y_true.long(), weight)
        return loss

    @classmethod
    def get_weights_balanced(cls, mask_dir):
        """
        通过统计所有图片中各个类别有效面积的比例，再自动设置权重，使得样本近似均衡
        :param mask_dir:
        :return:
        """
        masks_merge = []
        mask_list = os.listdir(mask_dir)
        for mask_file in tqdm(mask_list):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) // 31
            masks_merge.append(mask)
        masks_merge = np.array(masks_merge)
        classes, counts = np.unique(masks_merge, return_counts=True)
        weights = counts.max() / counts
        return weights


if __name__ == '__main__':
    myloss = MyCrossEntropyLoss()
    weights = myloss.get_weights_balanced(
        'E:\workspace\sandiao\data\Multi-class/train_data\eight-class/train_data\labels')
    print(weights)
