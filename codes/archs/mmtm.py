import os
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from .resnet_simclr import ResNetSimCLR


sigmoid = nn.Sigmoid()


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class ResBlock(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        if c1 == c2:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv2d(c1, c2, 1, stride, 0)

        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.BatchNorm2d(c2)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block(x) + self.downsample(x)
        return self.relu(x)
        
# channel attention, also called SEBlock
class MMTM(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        c_reduce = c//4
        self.se = nn.Sequential(
            nn.Linear(c*2, c_reduce), nn.ReLU(),
            nn.Linear(c_reduce, c*2)
        )
        
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1_mean = x1.mean(dim=[-2, -1])
        x2_mean = x2.mean(dim=[-2, -1])
        signal = self.se(torch.cat([x1_mean, x2_mean], dim=-1))
        signal = signal.unsqueeze(-1).unsqueeze(-1)
        s1, s2 = torch.chunk(signal, 2, dim=1)

        return x1 * sigmoid(s1), x2 * sigmoid(s2)


class MMTNet(nn.Module):
    def __init__(self, pretrained=True, checkpoint_BF=None, checkpoint_FL=None):
        super().__init__()
        self.name = 'mmtm'
        self.dropout = nn.Dropout(0.3)
        c_BF, c_FL = 3, 4 # for RGB and RGBa

        if checkpoint_BF is not None:
            print('BF: Using contrastive learning pretrained checkpoint!')
            model_BF = ResNetSimCLR('resnet50', c_BF)
            model_BF.load_state_dict(checkpoint_BF['state_dict'])
            self.model_BF = model_BF.backbone
        else:
            self.model_BF = torchvision.models.resnet50(pretrained=pretrained)

        if checkpoint_FL is not None:
            print('FL: Using contrastive learning pretrained checkpoint!')
            model_FL = ResNetSimCLR('resnet50', c_FL)
            model_FL.load_state_dict(checkpoint_FL['state_dict'])
            self.model_FL = model_FL.backbone
#            freeze_parameters(self.model_FL)    
        else:
            self.model_FL = torchvision.models.resnet50(pretrained=pretrained)
            self.model_FL.conv1 = nn.Conv2d(c_FL, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # define models for BF and FL
        # self.conv1_BF = nn.Conv2d(c_BF, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_BF = self.model_BF.conv1
        self.bn1_BF = self.model_BF.bn1
        self.relu_BF = self.model_BF.relu
        self.maxpool_BF = self.model_BF.maxpool
        self.layer1_BF = self.model_BF.layer1
        self.layer2_BF = self.model_BF.layer2
        self.layer3_BF = self.model_BF.layer3
        self.layer4_BF = self.model_BF.layer4
        self.avgpool_BF = self.model_BF.avgpool

        # self.conv1_FL = nn.Conv2d(c_FL, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_FL = self.model_FL.conv1
        self.bn1_FL = self.model_FL.bn1
        self.relu_FL = self.model_FL.relu
        self.maxpool_FL = self.model_FL.maxpool
        self.layer1_FL = self.model_FL.layer1
        self.layer2_FL = self.model_FL.layer2
        self.layer3_FL = self.model_FL.layer3
        self.layer4_FL = self.model_FL.layer4
        self.avgpool_FL = self.model_FL.avgpool
        
        self.mmtm1 = MMTM(c=256)
        self.mmtm2 = MMTM(c=512)
        self.mmtm3 = MMTM(c=1024)
        self.mmtm4 = MMTM(c=2048)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        
        self.fc_fusion = nn.Linear(128, 1)


    def forward(self, x):
        x_BF, x_FL = x

        x_BF = self.conv1_BF(x_BF)
        x_BF = self.bn1_BF(x_BF)
        x_BF = self.relu_BF(x_BF)
        x_BF = self.maxpool_BF(x_BF)

        x_FL = self.conv1_FL(x_FL)
        x_FL = self.bn1_FL(x_FL)
        x_FL = self.relu_FL(x_FL)
        x_FL = self.maxpool_FL(x_FL)

        # ==========
        x_BF = self.layer1_BF(x_BF)
        x_FL = self.layer1_FL(x_FL)
        x_BF, x_FL = self.mmtm1(x_BF, x_FL)

        # ==========
        x_BF = self.layer2_BF(x_BF)
        x_FL = self.layer2_FL(x_FL)
        x_BF, x_FL = self.mmtm2(x_BF, x_FL)

        # ==========
        x_BF = self.layer3_BF(x_BF)
        x_FL = self.layer3_FL(x_FL)
        x_BF, x_FL = self.mmtm3(x_BF, x_FL)

        # ==========
        x_BF = self.layer4_BF(x_BF)
        x_FL = self.layer4_FL(x_FL)
        x_BF, x_FL = self.mmtm4(x_BF, x_FL)

        # ==========
        x_BF = self.avgpool_BF(x_BF)
        x_BF = x_BF.view(x_BF.size(0), -1)
        x_FL = self.avgpool_FL(x_FL)
        x_FL = x_FL.view(x_FL.size(0), -1)
        x_fusion = x_BF + x_FL

        x_fusion = self.fusion_mlp(x_fusion)
        x_fusion = self.dropout(x_fusion)
        logit_fusion = self.fc_fusion(x_fusion)

        return logit_fusion


