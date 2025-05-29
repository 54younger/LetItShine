import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights, regnet_y_16gf, RegNet_Y_16GF_Weights

from .resnet_simclr import ResNetSimCLR


def load_resnet(channel, c_out=1, pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512*4, c_out)
    return model

def load_regnet(channel, c_out=1, pretrained=True):
    # weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
    # model = regnet_y_32gf(weights=weights)
    weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
    model = regnet_y_16gf(weights=weights)
    model.stem[0] = nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
    # model.fc = nn.Linear(3712, c_out)
    model.fc = nn.Linear(3024, c_out)
    return model
    

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

class SinglemodalNet(nn.Module):
    def __init__(self, model_name, channel, pretrained=True, checkpoint=None):
        super().__init__()
        self.simclr = checkpoint is not None
        if 'resnet' in model_name:
            if checkpoint is not None:
                print('Using contrastive learning pretrained checkpoint!')
                self.model = ResNetSimCLR(model_name, channel)
                self.model.load_state_dict(checkpoint['state_dict'])
#                freeze_parameters(self.model)
                dim_mlp = self.model.backbone.fc[0].in_features
                self.model.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 1))
            else:
                self.model = load_resnet(channel, 1, pretrained)
        elif 'regnet' in model_name:
            self.model = load_regnet(channel, 1, pretrained)

    def forward(self, x):
        x = self.model(x)
        return x


class MultimodalNet(nn.Module):
    # fusion_mode could be E:early, L:late, and I:intermediate
    def __init__(self, model_name, channel, fusion_mode='E', pretrained=True, checkpoint_BF=None, checkpoint_FL=None, checkpoint_E=None):
        super().__init__()
        self.fusion_mode = fusion_mode

        c1, c2 = 3, channel-3 # for RGB and RGBa

        if fusion_mode == 'E':
            if 'resnet' in model_name:
                if checkpoint_E is not None:
                    print('E:Using contrastive learning pretrained checkpoint!')
                    model = ResNetSimCLR(model_name, channel)
                    model.load_state_dict(checkpoint_E['state_dict'])
                    self.model = model.backbone
 #                   freeze_parameters(model)
                    self.model.fc = nn.Linear(512*4, 1)
                else:    
                    self.model = load_resnet(channel, 1, pretrained)
            elif 'regnet' in model_name:
                self.model = load_regnet(c1+c2, 1, pretrained)
        elif fusion_mode == 'L':
            if 'resnet' in model_name:
                out_dim = 2048 if model_name == 'resnet50' else 512
                if checkpoint_BF is not None and checkpoint_BF.get('state_dict', None) is not None:
                    print('BF:Using contrastive learning pretrained checkpoint!')
                    model1 = ResNetSimCLR(model_name, c1)
                    model1.load_state_dict(checkpoint_BF['state_dict'])
                    model1 = model1.backbone
                    #freeze_parameters(model1)
                else:
                    #checkpoint = {}
                    #for k, v in checkpoint_BF.items():
                    #    checkpoint[k[6:]] = v
                    model1 = load_resnet(c1, 1, pretrained)
                    #model1.load_state_dict(checkpoint)

                if checkpoint_FL is not None and checkpoint_FL.get('state_dict', None) is not None:
                    print('FL:Using contrastive learning pretrained checkpoint!')
                    model2 = ResNetSimCLR(model_name, c2)
                    model2.load_state_dict(checkpoint_FL['state_dict'])
                    model2 = model2.backbone
                    #freeze_parameters(model2)
                else:    
                    #checkpoint = {}
                    #for k, v in checkpoint_FL.items():
                    #    checkpoint[k[6:]] = v
                    model2 = load_resnet(c2, 1, pretrained)
                    #model2.load_state_dict(checkpoint)

                self.model1 = nn.Sequential(
                    model1.conv1, model1.bn1, model1.relu, model1.maxpool,
                    model1.layer1, model1.layer2, model1.layer3, model1.layer4)
                self.model2 = nn.Sequential(
                    model2.conv1, model2.bn1, model2.relu, model2.maxpool,
                    model2.layer1, model2.layer2, model2.layer3, model2.layer4)

            elif model_name == 'regnet':
                out_dim = 3024 # 3712, 3024
                model1 = load_regnet(c1, 1, pretrained)
                model2 = load_regnet(c2, 1, pretrained)

                self.model1 = nn.Sequential(
                    model1.stem, model1.trunk_output)
                self.model2 = nn.Sequential(
                    model2.stem, model2.trunk_output)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                              nn.Linear(out_dim*2, 256),
                              nn.ReLU(), 
                              nn.Dropout(0.2),
                              nn.Linear(256, 1))


    def forward(self, x):
        x1, x2 = x
        if self.fusion_mode == 'E':
            x = torch.cat([x1, x2], dim=1)
            return self.model(x)

        if self.fusion_mode == 'L':
            x1 = self.model1(x1)
            x2 = self.model2(x2)
            x1 = self.avgpool(x1).flatten(start_dim=1)
            x2 = self.avgpool(x2).flatten(start_dim=1)
            x_fusion = torch.cat([x1, x2], dim=1)
            return self.fc(x_fusion)
 


