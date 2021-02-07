from functools import reduce
import torch


def acc(logits, y_true):
    '''
    计算平均精度
    :param logits: 模型最后的输出图（不是预测图）
    :param y_true: 标签图
    :param num_output_channels: 最后输出图的通道数
    :return: 精度
    '''
    y_pred = get_ypred(logits)
    acc = y_pred.eq(y_true).sum().item() / (reduce(lambda x, y: x * y, y_true.shape))
    return acc


def b_iou(logits, y_true):
    '''
    计算二分类模型的iou得分，
    :return:
    '''
    y_pred = get_ypred(logits)
    intersection = torch.sum(y_pred * y_true)  # 交集
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection  # 并集
    iou = intersection / union
    return iou.item()


def get_ypred(logits):
    '''
    将模型输出转换成预测图
    :param logits: 模型输出
    :return:
    '''
    out_channels = logits.shape[1]
    if out_channels == 1:  # 二分类
        y_pred = torch.sigmoid(logits)
        y_pred = torch.where(y_pred > 0.5, 1, 0)
    else:  # 多分类
        y_pred = logits.argmax(dim=1)

    return y_pred
