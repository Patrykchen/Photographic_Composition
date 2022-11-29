"""
Created on Tue April 19 2022
@author: Wang Zhicheng
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.backbone import build_backbone
import models.AlexNet as AlexNet
import models.ResNet as ResNet
import models.DenseNet as DenseNet
class PCPNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(1000, self.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def feature_get(self,x):
        x1 = self.backbone.body1[:3](x)
        aaa = self.backbone.body1[:3]
        x2 = self.backbone.body2(x1)
        x3 = self.backbone.body3(x2)
        x4 = self.backbone.body4(x3)
        return x1,x2,x3,x4

class SetCriterion_PR(nn.Module):
    def __init__(self,pos_weight=1,reduction='mean',epsilon=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self,logits,target):
        #logits = F.sigmoid(logits,dim=1)
        loss = - self.pos_weight * target * torch.log(logits + self.epsilon) - \
               (1 - target) * torch.log(1 - logits + self.epsilon)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def build(args, training):
    num = 9
    if args.backbone == 'vgg16_bn':
        backbone = build_backbone(args,num=9)
        model = PCPNet(backbone)
        criterion = SetCriterion_PR()
        return model, criterion
    if args.backbone == 'alexnet':
        backbone = AlexNet.build_backbone(args,num=9)
        model = PCPNet(backbone)
        criterion = SetCriterion_PR()
        return model, criterion
    if args.backbone == 'resnet':
        backbone = ResNet.build_backbone(args,num=9)
        model = PCPNet(backbone)
        criterion = SetCriterion_PR()
        return model, criterion
    if args.backbone == 'densenet':
        backbone = DenseNet.build_backbone(args,num=9)
        model = PCPNet(backbone)
        criterion = SetCriterion_PR()
        return model, criterion
