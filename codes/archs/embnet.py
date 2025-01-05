import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .attention import SpatialTransformer
from .resnet_simclr import ResNetSimCLR


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


class EfficientCrossAttention(nn.Module):
    def __init__(self, reduction=8, in_channels=64, context_dim=None):
        super().__init__()
        reduce_dim = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, reduce_dim, 1, 1, 0, bias=False)
        self.att = SpatialTransformer(in_channels=reduce_dim, n_heads=reduce_dim//32, context_dim=context_dim)
        self.conv2 = nn.Conv2d(reduce_dim, in_channels, 1, 1, 0, bias=False)

    def forward(self, x_in, context=None):
        x = self.conv1(x_in)
        x = self.att(x, context)
        x = self.conv2(x)
        return x + x_in


class EmbNet(nn.Module):
    def __init__(self, channel=7, pretrained=True, checkpoint_BF=None, checkpoint_FL=None, context_dim=128):
        super().__init__()
        self.name = 'embnet'
        c_BF, c_FL = 3, channel-3 # for RGB and RGBa

        if checkpoint_BF is not None:
            print("BF:Using CL...")
            model_BF = ResNetSimCLR('resnet50', c_BF)
            model_BF.load_state_dict(checkpoint_BF['state_dict'])
            self.model_BF = model_BF.backbone
            # freeze_parameters(self.model_BF)
        else:
            self.model_BF = torchvision.models.resnet50(pretrained=True)
            self.model_BF.fc = nn.Linear(512*4, context_dim)

        if checkpoint_FL is not None:
            print("FL:Using CL...")
            model_FL = ResNetSimCLR('resnet50', c_FL)
            model_FL.load_state_dict(checkpoint_FL['state_dict'])
            self.model_FL = model_FL.backbone
#            freeze_parameters(self.model_FL)
        else:
            self.model_FL = torchvision.models.resnet50(pretrained=True)
            self.model_FL.conv1 = nn.Conv2d(c_FL, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model_FL.fc = nn.Linear(512*4, context_dim)


        self.conv_in = nn.Sequential(
            nn.Conv2d(c_BF, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.model_BF.bn1, self.model_BF.relu, self.model_BF.maxpool
        )

        self.layer1 = self.model_BF.layer1
        self.layer2 = self.model_BF.layer2
        self.layer3 = self.model_BF.layer3
        self.layer4 = self.model_BF.layer4
        self.avgpool = self.model_BF.avgpool

        # self.att1 = SpatialTransformer(in_channels=64, n_heads=2, context_dim=context_dim)
        # self.att2 = SpatialTransformer(in_channels=256, n_heads=8, context_dim=context_dim)
        # self.att3 = SpatialTransformer(in_channels=512, n_heads=16, context_dim=context_dim)
        # self.att4 = SpatialTransformer(in_channels=1024, n_heads=32, context_dim=context_dim)
        self.att1 = EfficientCrossAttention(reduction=2, in_channels=64, context_dim=context_dim)
        self.att2 = EfficientCrossAttention(reduction=4, in_channels=256, context_dim=context_dim)
        self.att3 = EfficientCrossAttention(reduction=8, in_channels=512, context_dim=context_dim)
        self.att4 = EfficientCrossAttention(reduction=8, in_channels=1024, context_dim=context_dim)

        self.BF_mlp = nn.Linear(2048, context_dim)
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(), nn.Linear(context_dim*2, 1)
        )

    def forward(self, x):
        x_BF, x_FL = x

        embedding_FL = self.model_FL(x_FL).unsqueeze(1)

        x_BF = self.conv_in(x_BF)

        x_BF = self.att1(x_BF, embedding_FL)
        x_BF = self.layer1(x_BF)

        x_BF = self.att2(x_BF, embedding_FL)
        x_BF = self.layer2(x_BF)

        x_BF = self.att3(x_BF, embedding_FL)
        x_BF = self.layer3(x_BF)

        x_BF = self.att4(x_BF, embedding_FL)
        x_BF = self.layer4(x_BF)

        embedding_BF = self.avgpool(x_BF).flatten(start_dim=1)
        embedding_BF = self.BF_mlp(embedding_BF)

        logit = self.fusion_mlp(torch.cat([embedding_BF, embedding_FL.squeeze(1)], dim=1))

        return logit




