import torch
import torch.nn.functional as F


class DiceBceLoss:
    def __init__(self, batch=True):
        super(DiceBceLoss, self).__init__()
        self.batch = batch
        self.bce_loss = F.binary_cross_entropy_with_logits

    def soft_dice_coeff(self, model_output, y_true):
        y_pred = torch.sigmoid(model_output)
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, model_output, y_true):
        loss = 1 - self.soft_dice_coeff(model_output, y_true)
        return loss

    def __call__(self, model_output, y_true, weight):
        a = self.bce_loss(model_output, y_true)
        b = self.soft_dice_loss(model_output, y_true)
        return a + b
