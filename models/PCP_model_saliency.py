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

class PCP_model(nn.Module):
    def __init__(self,backbone_vgg:nn.Module,backbone_saliency:nn.Module):
        super().__init__()
        # 显著性图获取
        saliency_backbone = list(backbone_saliency.backbone.children())
        saliency_body2 = list(backbone_saliency.body2.children())
        saliency_body3 = list(backbone_saliency.body3.children())
        saliency_encoder = list(backbone_saliency.encoder.children())
        self.body1 = nn.Sequential(*saliency_backbone)
        self.body2 = nn.Sequential(*saliency_body2)
        self.body3 = nn.Sequential(*saliency_body3)
        self.encoder = nn.Sequential(*saliency_encoder)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # 分类获取
        classifier = list(backbone_vgg.classifier.children())
        self.classifier = nn.Sequential(*classifier[:6],
                                        nn.Linear(4096, 9),
                                        nn.Softmax())

    def forward(self,tensor_list):
        #xs = tensor_list.clone()
        #temp = self.backbone_saliency(xs)
        #temp = 0
        # # x1 = self.backbone(xs)
        # # x2 = self.backbone_body2(x1)
        # # x3 = self.backbone_body3(x2)
        # # xs_c = torch.cat((x1, x2, x3), dim=1)
        # #output = self.backbone_encoder_saliency(xs_c)
        #output =(output - torch.min(output))/(torch.max(output) - torch.min(output))
        # #a = np.array(a)
        # # 显著性图获取
        # xs = xs  # 这边改成两倍的话，效果非常糟糕
        xs = tensor_list.clone()
        # xs1 = tensor_list.clone()
        # xs1 = self.body1(xs1)
        # xs2 = self.body2(xs1)
        # xs3 = self.body3(xs2)
        # xs_c = torch.cat((xs1, xs2, xs3), dim=1)
        # output = self.encoder(xs_c)
        # output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        # xs = xs*(1+0.1*output)
        xs = self.body1(xs)
        xs = self.body2(xs)
        xs = self.body3(xs)
        xs = self.avgpool(xs)
        xs = torch.flatten(xs, 1)
        output1 = self.classifier(xs)
        return output1

class SetCriterion_PR(nn.Module):
    def __init__(self,pos_weight=1,reduction='mean',epsilon=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.epsilon = epsilon


    def forward(self,logits,target,model):
        #logits = F.sigmoid(logits,dim=1)
        loss = - self.pos_weight * target * torch.log(logits + self.epsilon) - \
               (1 - target) * torch.log(1 - logits + self.epsilon)
        norm = 0
        for name, parameters in model.state_dict().items():
            if 'weight' in name:
                norm += torch.sum(torch.pow(parameters,2))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        loss = loss + norm*1e-5
        return loss

def build_(args):
    backbone_vgg = models.vgg16_bn(pretrained=True)
    backbone_saliency = saliency_model(backbone_vgg)
    checkpoint = torch.load(args.backbone_saliency_path, map_location='cpu')
    backbone_saliency.load_state_dict(checkpoint['model'])
    model = PCP_model(backbone_vgg,backbone_saliency)
    criterion = SetCriterion_PR()
    return model, criterion