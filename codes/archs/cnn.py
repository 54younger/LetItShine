import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    # 128 - 64 - 32 - 16 - 8
    def __init__(self, name, channel=3, dim=64, depth=[3, 4, 6, 3], num_classes=1):
        super().__init__()
        self.name = 'cnn'
        self.conv_in = nn.Sequential(
            nn.Conv2d(channel, dim, 5, 2, 2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
            )

        self.conv_main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, 3, 1, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.Conv2d(dim*2, dim*4, 3, 1, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim*4, dim*4, 3, 1, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim*8, 3, 1, 1),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(dim*8, dim*8, 3, 1, 1),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            nn.Conv2d(dim*8, dim*16, 3, 1, 1),
            nn.BatchNorm2d(dim*16),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim*16, num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_main(x)

        out = self.fc(self.flatten(x))

        return out
        # return torch.sigmoid(out)




