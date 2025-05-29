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
        

class AFBlock(nn.Module):
    def __init__(self, c=64, cp=64, stride=1, first_stage=False):
        super().__init__()
        self.first_stage = first_stage
        self.cr1 = nn.Sequential(
            nn.Conv2d(c, c//2, 1, 1, 0), nn.ReLU(),
            nn.Conv2d(c//2, c//4, 1, 1, 0), nn.ReLU(),
            nn.Conv2d(c//4, 1, 1, 1, 0)
        )
        self.cr2 = nn.Sequential(
            nn.Conv2d(c, c//2, 1, 1, 0), nn.ReLU(),
            nn.Conv2d(c//2, c//4, 1, 1, 0), nn.ReLU(),
            nn.Conv2d(c//4, 1, 1, 1, 0)
        )
       
        if not first_stage:
            self.resblocks = nn.Sequential(
                ResBlock(cp, c, stride=stride), ResBlock(c, c)
            )
            self.cr3 = nn.Sequential(
                nn.Conv2d(c, c//2, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(c//2, c//4, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(c//4, 1, 1, 1, 0)
            )
            self.cr4 = nn.Sequential(
                nn.Conv2d(c, c//2, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(c//2, c//4, 1, 1, 0), nn.ReLU(),
                nn.Conv2d(c//4, 1, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
            
    def forward(self, x1, x2, x_fusion=False):
        x1_cr = self.cr1(x1)
        x2_cr = self.cr2(x2)
        x12_cr = torch.cat([x1_cr, x2_cr], dim=1)
        x12_cr = self.softmax(x12_cr)
        x1_cr, x2_cr = torch.chunk(x12_cr, 2, dim=1)

        x3 = x1 * x1_cr + x2 * x2_cr
        if self.first_stage:
            return x3

        x3_cr = self.cr3(x3)
        x_fusion = self.resblocks(x_fusion)
        x4_cr = self.cr4(x_fusion)
        x34_cr = torch.cat([x3_cr, x4_cr], dim=1) 
        x34_cr = self.softmax(x34_cr)
        x3_cr, x4_cr = torch.chunk(x34_cr, 2, dim=1)
        out = x3 * x3_cr + x_fusion * x4_cr
        return out


class CABlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        c_reduce = c//8
        self.to_qk1 = nn.Conv2d(c, c_reduce*2, 1, 1, 0)
        self.to_qk2 = nn.Conv2d(c, c_reduce*2, 1, 1, 0)
        self.to_v1 = nn.Conv2d(c, c, 1, 1, 0)
        self.to_v2 = nn.Conv2d(c, c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        q1, k1 = torch.chunk(self.to_qk1(x1), 2, dim=1)
        q2, k2 = torch.chunk(self.to_qk2(x2), 2, dim=1)

        q1 = q1.view(b, -1, h*w).permute(0, 2, 1) # b, n, c
        q2 = q2.view(b, -1, h*w).permute(0, 2, 1) # b, n, c
        k1 = k1.view(b, -1, h*w) # b, c, n
        k2 = k2.view(b, -1, h*w) # b, c, n

        v1 = self.to_v1(x1).view(b, -1, h*w) # b, c, n
        v2 = self.to_v2(x2).view(b, -1, h*w) # b, c, n

        attn1 = self.softmax(torch.bmm(q1, k1)) # b, n, n
        attn2 = self.softmax(torch.bmm(q2, k2)) # b, n, n
        
        x1 = torch.bmm(v1, attn2).view(b, c, h, w) * self.gamma + x1
        x2 = torch.bmm(v2, attn1).view(b, c, h, w) * self.beta + x2
        return x1, x2


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


class CAFNet(nn.Module):
    def __init__(self, channel=7, pretrained=True, checkpoint_BF=None, checkpoint_FL=None):
        super().__init__()
        self.name = 'cafnet'
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

        
        self.cab1 = CABlock(c=64)
        self.fus1 = AFBlock(c=64, cp=64, stride=1, first_stage=True)
        self.cab2 = CABlock(c=256)
        self.fus2 = AFBlock(c=256, cp=64, stride=1, first_stage=True)
        self.cab3 = CABlock(c=512)
        self.fus3 = AFBlock(c=512, cp=256, stride=2)
        self.cab4 = CABlock(c=1024)
        self.fus4 = AFBlock(c=1024, cp=512, stride=2)
        self.avgpool_fusion = nn.AdaptiveAvgPool2d((1, 1))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 128),
            Swish_Module(),
        )
            
        self.BF_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            Swish_Module(),
        )
        self.FL_mlp = nn.Sequential(
            nn.Linear(2048, 128),
            Swish_Module(),
        )
        

        self.fc_BF = nn.Linear(128, 1)
        self.fc_FL = nn.Linear(128, 1)
        self.fc_fusion = nn.Linear(128, 1)
        self.fc_concat = nn.Linear(128*2, 1)


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
        #m_BF, m_FL = self.cab1(x_BF, x_FL)
        #x_fusion = self.fus1(m_BF, m_FL)
        x_BF = self.layer1_BF(x_BF)
        x_FL = self.layer1_FL(x_FL)

        # ==========
        m_BF, m_FL = self.cab2(x_BF, x_FL)
#        m_BF, m_FL = x_BF, x_FL
        x_fusion = self.fus2(m_BF, m_FL)
        x_BF = self.layer2_BF(m_BF)
        x_FL = self.layer2_FL(m_FL)

        # ==========
        m_BF, m_FL = self.cab3(x_BF, x_FL)
 #       m_BF, m_FL = x_BF, x_FL
        x_fusion = self.fus3(m_BF, m_FL, x_fusion)
        x_BF = self.layer3_BF(m_BF)
        x_FL = self.layer3_FL(m_FL)

        # ==========
        m_BF, m_FL = self.cab4(x_BF, x_FL)
  #      m_BF, m_FL = x_BF, x_FL
        x_fusion = self.fus4(m_BF, m_FL, x_fusion)
        x_BF = self.layer4_BF(m_BF)
        x_FL = self.layer4_FL(m_FL)

        # ==========
        x_BF = self.avgpool_BF(x_BF)
        x_BF = x_BF.view(x_BF.size(0), -1)
        x_FL = self.avgpool_FL(x_FL)
        x_FL = x_FL.view(x_FL.size(0), -1)
        x_fusion = self.avgpool_fusion(x_fusion)
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
        #x_concat = torch.cat([x_BF, x_FL], dim=1)
        x_concat = self.dropout(x_concat)
        logit_final = self.fc_concat(x_concat)

        return logit_final
        # return logit_BF, logit_FL, logit_fusion, logit_final


