import torch
from torch import nn
import models.vgg_ as models
class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:44])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.avgpool = backbone.avgpool
        self.num_channels = num_channels
        classifier = list(backbone.classifier.children())
        self.classifier = nn.Sequential(*classifier[:6],
                                        nn.Linear(4096, self.num_channels))
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self,tensor_list):
        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
            xs = self.avgpool(xs)
            xs = torch.flatten(xs, 1)
            xs = self.classifier(xs)
            return xs
        else:
            xs = self.body(tensor_list)
            xs = self.avgpool(xs)
            xs = torch.flatten(xs, 1)
            xs = self.classifier(xs)
            return xs

class Backbone_VGG(BackboneBase_VGG):
    def __init__(self, name: str, return_interm_layers: bool, num: int):
         if name == 'vgg16_bn':
             backbone = models.vgg16_bn(pretrained=True)
         elif name == 'vgg16':
             backbone = models.vgg16(pretrained=True)
         num_channels = num
         # super().__init__相当于进行了父类的初始化，得到了对应的参数
         super().__init__(backbone, num_channels, name, return_interm_layers)

def build_backbone(args,num):
    backbone = Backbone_VGG(args.backbone, True, num)
    return backbone

if __name__ == "__main__":
    backbone = Backbone_VGG('vgg16_bn', True)