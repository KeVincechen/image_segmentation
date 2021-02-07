import logging

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from dataset import *
from loss import *
from torch.utils.data import DataLoader, random_split
import os
import argparse, yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Train:
    def __init__(self):
        self.log = self.get_logger()

    def get_logger(self):
        """
        创建日志对象
        :return:
        """
        base_dir = os.path.dirname(__file__)  # 当前文件所在目录
        log_dir = os.path.join(base_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = logging.getLogger()
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        logger.addHandler(fh)
        logger.addHandler(ch)
        format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        fh.setFormatter(format)
        logger.setLevel(logging.DEBUG)
        return logger

    def fit(self, imgs_dir, masks_dir, network, dataset, in_channels, num_classes, loss_weights=None, batch_size=16,
            learning_rate=1e-4, checkpoint_path=None, split_val_rate=0.1, num_workers=4, optim='Adam',
            backbone_name=None, save_top_k=10, save_mode='max', model_saved_filename='model-{epoch:04d}-{val_acc:.4f}',
            backbone_pretrained=False, gpus=1, accumulate_grad_batches=8, loss_func='MyCrossEntropyLoss',
            metrics='acc'):
        self.dataset = eval(f'{dataset}(r"{imgs_dir}", r"{masks_dir}")')
        n_val_data = int(len(self.dataset) * split_val_rate)
        n_train_data = len(self.dataset) - n_val_data
        self.log.info(f'训练集数量：{n_train_data}，验证集数量：{n_val_data}')
        train_data, val_data = random_split(self.dataset, [n_train_data, n_val_data])
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size * 2, num_workers=num_workers, drop_last=True,
                                pin_memory=True)
        checkpoint_callback = ModelCheckpoint(
            monitor=f'val_{metrics}',
            filename=model_saved_filename,
            save_top_k=save_top_k,
            mode=save_mode
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = eval(f'{network}.load_from_checkpoint("{checkpoint_path}")')
            model.lr = learning_rate
            model.loss_func = eval(f'{loss_func}()')
            model.metrics = metrics
            model.loss_weights = torch.FloatTensor(loss_weights).cuda() if loss_weights else None
            model.in_channels = in_channels
            model.num_classes = num_classes
            model.backbone_pretrained = backbone_pretrained
            model.optim = optim
            self.log.info(f'加载预训练模型：{checkpoint_path}')
        else:
            model_params = {
                'in_channels': in_channels,
                'num_classes': num_classes,
                'backbone_name': backbone_name,
                'backbone_pretrained': backbone_pretrained,
                'loss_func': loss_func,
                'loss_weights': loss_weights,
                'metrics': metrics,
                'optim': optim,
                'lr': learning_rate
            }
            if not backbone_name:  # 模型中没有用到resnet等预训练模型结构，如unet
                del model_params['backbone_name']
                del model_params['backbone_pretrained']
            model = eval(f'{network}(**{model_params})')
        self.log.info(f'训练参数:\n'
                      f'imgs_dir: {imgs_dir}\n'
                      f'masks_dir: {masks_dir}\n'
                      f'network: {network}\n'
                      f'backbone:{backbone_name}\n' 
                      f'dataset: {dataset}\n'
                      f'loss_weights: {loss_weights}\n'
                      f'in_channels: {in_channels}\n'
                      f'num_classes: {num_classes}\n'
                      f'batch_size: {batch_size}\n'
                      f'learning_rate: {learning_rate}\n'
                      f'checkpoint_path: {checkpoint_path}\n'
                      f'split_val_rate: {split_val_rate}\n'
                      f'num_workers: {num_workers}\n'
                      f'save_top_k: {save_top_k}\n'
                      f'save_mode: {save_mode}\n'
                      f'model_saved_filename: {model_saved_filename}\n'
                      f'backbone_pretrained: {backbone_pretrained}\n'
                      f'gpus: {gpus}\n'
                      f'accumulate_grad_batches: {accumulate_grad_batches}\n'
                      f'optim: {optim}\n'
                      f'loss_function: {loss_func}\n'
                      f'metrics: {metrics}\n')

        trainer = Trainer(gpus=gpus, precision=16, accumulate_grad_batches=accumulate_grad_batches,
                          callbacks=[checkpoint_callback])

        trainer.fit(model, train_loader, val_loader)

    @staticmethod
    def parse_args(yaml_file=''):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-y',
            '--yaml',
            help='yaml文件路径',
            default=yaml_file,
            type=str
        )
        args = parser.parse_args()
        yaml_file = args.yaml
        with open(yaml_file, encoding='utf-8') as f:
            params = yaml.full_load(f.read())
        return params


if __name__ == '__main__':
    trainer = Train()
    params = trainer.parse_args(r'yamls\tianchi-road-unet.yaml')
    trainer.fit(**params)
