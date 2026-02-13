import torch
import torch.nn as nn
from torchvision import models

from src.models.unet import DoubleConv

resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
encoder = torch.nn.Sequential(*list(resnet.children())[:-2])

class UNetTL(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.encoder = encoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv1 = DoubleConv(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = DoubleConv(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = DoubleConv(64, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv4 = DoubleConv(32, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        x = self.outc(x)
        return x
