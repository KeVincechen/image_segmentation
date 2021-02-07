"""
@Time: 2021/2/4 23:04 
@Author: xiashuobad
@File: base_module.py
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　━　　　 ┃
      ┃　┳┛　  ┗┳　  ┃
      ┃　　　　　　   ┃
      ┃　　　┻　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   你的孤独
        ┃　　　┃   虽败犹荣
        ┃　　　┗━━━━━━━━━┓
        ┃　　　　　　　    ┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛
"""
import logging
from pytorch_lightning import LightningModule
from torch import Tensor
import torch
from loss import *
from metrics import *
from torch.optim import *


class BaseModule(LightningModule):
    def __init__(self, loss_func='MyCrossEntropyLoss', loss_weights=None, metrics='acc', optim='Adam', lr=1e-4):
        super(BaseModule, self).__init__()
        self.lr = lr
        self.optim = optim
        self.loss_func = eval(f'{loss_func}()')
        self.loss_weights = torch.FloatTensor(loss_weights).cuda() if loss_weights else None
        self.metrics = metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y, weight=self.loss_weights)
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        precision = eval(f'{self.metrics}')(logits, y)
        self.log(self.metrics, precision, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = self.loss_func(logits, y, weight=self.loss_weights)
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        val_precision = eval(f'{self.metrics}')(logits, y)
        self.log(f'val_{self.metrics}', val_precision, prog_bar=True, on_epoch=True)
        return {'val_loss': val_loss, f'val_{self.metrics}': val_precision}

    def validation_epoch_end(self, outputs):
        outputs = outputs[-1]
        res = {k: v.item() if isinstance(v, Tensor) else v for k, v in outputs.items()}
        logging.info(f'epoch: {self.current_epoch} {res}')

    def on_train_epoch_end(self, outputs):
        outputs = outputs[0][-1][0]
        res = {}
        for k, v in outputs.items():
            if k.endswith('epoch'):
                res[k] = v.item() if isinstance(v, Tensor) else v
        logging.info(f'epoch: {self.current_epoch}, {res}')

    def configure_optimizers(self):
        optimizer = eval(f'{self.optim}')(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    pass
