# models/yolov8_custom.py
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True):
        super(C2f, self).__init__()
        self.blocks = nn.Sequential(*[
            Conv(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1)
            for i in range(num_blocks)
        ])
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            return x + self.blocks(x)
        else:
            return self.blocks(x)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, 1, 1, 0)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, kernel_size // 2)
        self.conv3 = Conv(out_channels, out_channels, 1, 1, 0)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.maxpool(x))
        x = self.conv3(self.maxpool(x))
        return x

class YOLOv8Custom(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Custom, self).__init__()
        self.backbone = nn.Sequential(
            Conv(3, 64, 3, 2, 1), # P1/2
            Conv(64, 128, 3, 2, 1), # P2/4
            C2f(128, 128, 3),
            Conv(128, 256, 3, 2, 1), # P3/8
            C2f(256, 256, 6),
            Conv(256, 512, 3, 2, 1), # P4/16
            C2f(512, 512, 6),
            Conv(512, 1024, 3, 2, 1), # P5/32
            C2f(1024, 1024, 3),
            SPPF(1024, 1024, 5)
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(1536, 512, 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(768, 256, 3),
            Conv(256, 256, 3, 2, 1),
            C2f(768, 512, 3),
            Conv(512, 512, 3, 2, 1),
            C2f(1536, 1024, 3),
            nn.Conv2d(1024, num_classes, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
