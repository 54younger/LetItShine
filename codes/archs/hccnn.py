import os
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from .resnet_simclr import ResNetSimCLR

sigmoid = nn.Sigmoid()

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

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = None 
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

class FusionBlock(nn.Module):
    def __init__(self, c=64, first_stage=False):
        super().__init__()
        c_in = c * 2 if first_stage else c * 3
        self.cr1 = nn.Sequential(nn.Conv2d(c_in, c, 3, 1, 1), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(c, c//4, 3, 1, 1), nn.ReLU(), nn.Conv2d(c//4, c, 3, 1, 1))

    def forward(self, x1, x2, x3=None):
        x_con = torch.cat([x1, x2], dim=1)
        if x3 is not None:
            x_con = torch.cat([x_con, x3], dim=1)
            
        x1_cr = self.cr1(x_con)
        x2_cr = self.cr2(x1_cr) + x1_cr
        return x2_cr

class MsABlock(nn.Module):
    def __init__(self, c=64, cp=64, stride=1):
        super().__init__()
        self.cr1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, cp))

        self.resblock = Bottleneck(c, cp, stride=stride)
            
    def forward(self, x):
        x_cr1 = self.cr1(x).unsqueeze(-1).unsqueeze(-1)
        x_cr2 = self.resblock(x)
        x_fusion = x_cr1 + x_cr2
        x_att = torch.sigmoid(x_fusion)
        return x_att * x


class HyperConnection(nn.Module):
    def __init__(self, c=64, cp=64, stride=1, first_stage=False):
        super().__init__()
        self.fusion = FusionBlock(c, first_stage=first_stage)
        self.resblocks = nn.Sequential(ResBlock(c, c//4), ResBlock(c//4, cp, stride=stride))
        self.msa = MsABlock(cp, cp)

    def forward(self, x1, x2, x3=None):
        x = self.fusion(x1, x2, x3)
        x = self.resblocks(x)
        x = self.msa(x)
        return x


class HcCNN(nn.Module):
    def __init__(self, channel=7, pretrained=True, checkpoint_BF=None, checkpoint_FL=None):
        super().__init__()
        self.name = 'hccnn'
        self.dropout = nn.Dropout(0.3)
        self.dropout2d = nn.Dropout2d(0.5)
        c_BF, c_FL = 3, channel-3 # for RGB and RGBa
        
        if checkpoint_BF is not None:
            print("BF:Using contrastive learning...")
            model_BF = ResNetSimCLR('resnet50', c_BF)
            model_BF.load_state_dict(checkpoint_BF['state_dict'])
            self.model_BF = model_BF.backbone
#            freeze_parameters(self.model_BF)
        else:
            self.model_BF = torchvision.models.resnet50(pretrained=True)
        
        if checkpoint_FL is not None:
            print("FL:Using contrastive learning...")
            model_FL = ResNetSimCLR('resnet50', c_FL)
            model_FL.load_state_dict(checkpoint_FL['state_dict'])
            self.model_FL = model_FL.backbone
#            freeze_parameters(self.model_FL)
        else:
            self.model_FL = torchvision.models.resnet50(pretrained=True)

        # define models for BF and FL
        self.conv1_BF = nn.Conv2d(c_BF, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_BF = self.model_BF.bn1
        self.relu_BF = self.model_BF.relu
        self.maxpool_BF = self.model_BF.maxpool
        self.layer1_BF = self.model_BF.layer1
        self.layer2_BF = self.model_BF.layer2
        self.layer3_BF = self.model_BF.layer3
        self.layer4_BF = self.model_BF.layer4
        self.avgpool_BF = self.model_BF.avgpool

        self.conv1_FL = nn.Conv2d(c_FL, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_FL = self.model_FL.bn1
        self.relu_FL = self.model_FL.relu
        self.maxpool_FL = self.model_FL.maxpool
        self.layer1_FL = self.model_FL.layer1
        self.layer2_FL = self.model_FL.layer2
        self.layer3_FL = self.model_FL.layer3
        self.layer4_FL = self.model_FL.layer4
        self.avgpool_FL = self.model_FL.avgpool

        
        self.hcm1 = HyperConnection(c=64, cp=256, stride=1, first_stage=True)
        self.hcm2 = HyperConnection(c=256, cp=512, stride=2)
        self.hcm3 = HyperConnection(c=512, cp=1024, stride=2)
        self.hcm4 = HyperConnection(c=1024, cp=2048, stride=2)
        self.hcm_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            Swish_Module())
            
        self.BF_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            Swish_Module())
        self.FL_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            Swish_Module())

        self.fc_BF = nn.Linear(128, 1)
        self.fc_FL = nn.Linear(128, 1)
        self.fc_fusion = nn.Linear(128, 1)
        self.fc_concat = nn.Linear(128*3, 1)


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
        x_fusion = self.hcm1(x_BF, x_FL, None)
        x_BF = self.layer1_BF(x_BF)
        x_FL = self.layer1_FL(x_FL)

        # ==========
        x_fusion = self.hcm2(x_BF, x_FL, x_fusion)
        x_BF = self.layer2_BF(x_BF)
        x_FL = self.layer2_FL(x_FL)

        # ==========
        x_fusion = self.hcm3(x_BF, x_FL, x_fusion)
        x_BF = self.layer3_BF(x_BF)
        x_FL = self.layer3_FL(x_FL)

        # ==========
        x_fusion = self.hcm4(x_BF, x_FL, x_fusion)
        x_BF = self.layer4_BF(x_BF)
        x_FL = self.layer4_FL(x_FL)

        # ==========
        x_BF = self.avgpool_BF(x_BF)
        x_BF = x_BF.view(x_BF.size(0), -1)
        x_FL = self.avgpool_FL(x_FL)
        x_FL = x_FL.view(x_FL.size(0), -1)
        x_fusion = self.hcm_pool(x_fusion)
        x_fusion = x_fusion.view(x_fusion.size(0), -1)

        x_BF = self.BF_mlp(x_BF)
        x_BF = self.dropout(x_BF)
        logit_BF = self.fc_BF(x_BF)

        x_FL = self.FL_mlp(x_FL)
        x_FL = self.dropout(x_FL)
        logit_FL = self.fc_FL(x_FL)
          
        x_fusion = self.fusion_mlp(x_fusion)
        x_fusion = self.dropout(x_fusion)
        logit_fusion = self.fc_fusion(x_fusion)

        x_concat = torch.cat([x_BF, x_FL, x_fusion], dim=1)
        x_concat = self.dropout(x_concat)
        logit_final = self.fc_concat(x_concat)

        return logit_final

