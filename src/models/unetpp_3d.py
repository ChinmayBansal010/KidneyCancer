import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPP3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_ch=32):
        super().__init__()

        self.pool = nn.MaxPool3d(2)

        self.conv00 = ConvBlock(in_channels, base_ch)
        self.conv10 = ConvBlock(base_ch, base_ch * 2)
        self.conv20 = ConvBlock(base_ch * 2, base_ch * 4)
        self.conv30 = ConvBlock(base_ch * 4, base_ch * 8)

        self.conv01 = ConvBlock(base_ch + base_ch * 2, base_ch)
        self.conv11 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2)
        self.conv21 = ConvBlock(base_ch * 4 + base_ch * 8, base_ch * 4)

        self.conv02 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch)
        self.conv12 = ConvBlock(base_ch * 4 + base_ch * 4, base_ch * 2)

        self.conv03 = ConvBlock(base_ch * 3 + base_ch * 2, base_ch)

        self.final = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))

        x01 = self.conv01(torch.cat([x00, F.interpolate(x10, scale_factor=2)], 1))
        x11 = self.conv11(torch.cat([x10, F.interpolate(x20, scale_factor=2)], 1))
        x21 = self.conv21(torch.cat([x20, F.interpolate(x30, scale_factor=2)], 1))

        x02 = self.conv02(torch.cat([x00, x01, F.interpolate(x11, scale_factor=2)], 1))
        x12 = self.conv12(torch.cat([x10, x11, F.interpolate(x21, scale_factor=2)], 1))

        x03 = self.conv03(torch.cat([x00, x01, x02, F.interpolate(x12, scale_factor=2)], 1))

        return self.final(x03)