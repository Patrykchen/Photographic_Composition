"""
Created on Tue April 26 2022
@author: Wang Zhicheng
"""

import torch
from torch import nn
import models.resnet_ as models

from functools import partial
from typing import Any, Optional

class BackboneBase_RES(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str):
        super().__init__()
        features = list(backbone.children())
        if name == 'resnet':
            self.body = nn.Sequential(*features[:9])  # 16x down-sample
        self.num_channels = num_channels
        self.classifier = nn.Sequential(nn.Linear(2048, self.num_channels))

    def forward(self,tensor_list):
        xs = self.body(tensor_list)
        xs = torch.flatten(xs, 1)
        xs = self.classifier(xs)
        return xs

class Backbone_ResNet(BackboneBase_RES):
    def __init__(self, name: str, num: int):
         #backbone = torch.hub.load('/pre_checkpoints/alexnet-owt-7be5be79.pth',
         #                          'alexnet', pretrained=True,source='local')
         backbone = models.resnet101('pre_checkpoints/resnet101-cd907fc2.pth')
         num_channels = num
         # super().__init__相当于进行了父类的初始化，得到了对应的参数
         super().__init__(backbone, num_channels, name)


def build_backbone(args,num):
    backbone = Backbone_ResNet(args.backbone, num)
    return backbone