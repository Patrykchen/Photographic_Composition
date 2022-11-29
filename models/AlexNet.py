"""
Created on Tue April 26 2022
@author: Wang Zhicheng
"""
import torch
from torch import nn
import models.vgg_ as models

from functools import partial
from typing import Any, Optional

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BackboneBase_ALEX(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str):
        super().__init__()
        features = list(backbone.features.children())
        if name == 'alexnet':
            self.body = nn.Sequential(*features[:13])  # 16x down-sample
        self.avgpool = backbone.avgpool
        self.num_channels = num_channels
        classifier = list(backbone.classifier.children())
        self.classifier = nn.Sequential(*classifier[:6],
                                        nn.Linear(4096, self.num_channels))
        self.num_channels = num_channels

    def forward(self,tensor_list):
        xs = self.body(tensor_list)
        xs = self.avgpool(xs)
        xs = torch.flatten(xs, 1)
        xs = self.classifier(xs)
        return xs

class Backbone_AlexNet(BackboneBase_ALEX):
    def __init__(self, name: str, num: int):
         #backbone = torch.hub.load('/pre_checkpoints/alexnet-owt-7be5be79.pth',
         #                          'alexnet', pretrained=True,source='local')
         backbone = AlexNet()
         backbone.load_state_dict(torch.load('pre_checkpoints/alexnet-owt-7be5be79.pth'))
         num_channels = num
         # super().__init__相当于进行了父类的初始化，得到了对应的参数
         super().__init__(backbone, num_channels, name)


def build_backbone(args,num):
    backbone = Backbone_AlexNet(args.backbone, num)
    return backbone