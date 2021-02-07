import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

model_dict = {
    'resnet18': (models.resnet50, 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': (models.resnet34, 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'resnet50': (models.resnet50, 'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'resnet101': (models.resnet101, 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'resnet152': (models.resnet152, 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'resnext50_32x4d': (models.resnext50_32x4d, 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
    'resnext101_32x8d': (models.resnext101_32x8d, 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'wide_resnet50_2': (models.wide_resnet50_2, 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'),
    'wide_resnet101_2': (models.wide_resnet101_2, 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'),
}


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(
            x)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(
            out))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x)))  # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(
            out)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(
            out))  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(
            x)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(
            out)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out


class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, back_bone_name='resnet50', pretrained=False):
        super(ResNet_Bottleneck_OS16, self).__init__()
        model, model_url = model_dict.get(back_bone_name)
        if pretrained:
            logging.info(f'加载预训练backbone模型:{back_bone_name}')
        resnet = model(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        self.layer5 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.backbone(x)  # (shape: (batch_size, 4*256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4)  # (shape: (batch_size, 4*512, h/16, w/16))

        return output


class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2
            print("pretrained resnet, 18")
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
            print("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x)  # (shape: (batch_size, 256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4)  # (shape: (batch_size, 512, h/16, w/16))

        return output


class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, in_channels=3, back_bone_name='resnet34', pretrained=False):
        super(ResNet_BasicBlock_OS8, self).__init__()
        model, model_url = model_dict.get(back_bone_name)
        if pretrained:
            logging.info(f'加载预训练backbone模型:{back_bone_name}')
        resnet = model(pretrained=pretrained)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[1:-4])
        if back_bone_name == 'resnet18':
            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
        elif back_bone_name == 'resnet34':
            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1,
                                 dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1,
                                 dilation=4)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        x = self.conv1(x)
        c3 = self.resnet(x)  # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)

        output = self.layer4(c3)  # (shape: (batch_size, 256, h/8, w/8))
        output = self.layer5(output)  # (shape: (batch_size, 512, h/8, w/8))

        return output
