"""
Created on Sat. June 7 2022
@author: Wang Zhicheng
"""

import torch
from torch import nn
import torch.nn.functional as F
import models.vgg_ as models
import numpy as np


class saliency_model(nn.Module):
    def __init__(self,backbone:nn.Module):
        super().__init__()
        features = list(backbone.features.children())
        self.backbone = nn.Sequential(*features[:24])
        #self.backbone.requires_grad = False
        self.body2 = nn.Sequential(*features[24:32],
                                   nn.MaxPool2d(stride=1,kernel_size=5,padding=2,ceil_mode=True))
        self.body3 = nn.Sequential(*features[34:43])

        self.encoder = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Conv2d(1280,64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64,1,kernel_size=(1,1)),
                                     nn.UpsamplingBilinear2d(scale_factor=8))
        prior_size = (int(224 / 8), int(224 / 8))
        self.prior = nn.Parameter(torch.ones((1, 1, prior_size[0], prior_size[1]), requires_grad=True))

    def forward(self,tensor_list):
        xs = tensor_list
        xs1 = self.backbone(xs)
        xs2 = self.body2(xs1)
        xs3 = self.body3(xs2)
        xs_c = torch.cat((xs1,xs2,xs3),dim=1)
        output = self.encoder(xs_c)
        return output

def build_saliency():
    backbone_vgg = models.vgg16_bn(pretrained=True)
    backbone_saliency = saliency_model(backbone_vgg)
    checkpoint = torch.load('pre_checkpoints/saliency.pth', map_location='cpu')
    backbone_saliency.load_state_dict(checkpoint['model'])
    return backbone_saliency