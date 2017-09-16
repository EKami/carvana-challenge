import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class UNetOriginal(nn.Module):
    def __init__(self, in_shape):
        super(UNetOriginal, self).__init__()
        channels, height, width = in_shape

        self.down1 = nn.Sequential(
            ConvRelu2d(channels, 64, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(64, 64, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.down2 = nn.Sequential(
            ConvRelu2d(64, 128, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(128, 128, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.down3 = nn.Sequential(
            ConvRelu2d(128, 256, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(256, 256, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.maxPool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.down4 = nn.Sequential(
            ConvRelu2d(256, 512, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(512, 512, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.maxPool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.center = nn.Sequential(
            ConvRelu2d(512, 1024, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(1024, 1024, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.upSample1 = nn.Upsample(size=(1024, 1024), scale_factor=(2, 2), mode="bilinear")

        self.up1 = nn.Sequential(
            ConvRelu2d(1024, 512, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(512, 512, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.upSample2 = nn.Upsample(size=(512, 512), scale_factor=(2, 2), mode="bilinear")

        self.up2 = nn.Sequential(
            ConvRelu2d(512, 256, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(256, 256, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.upSample3 = nn.Upsample(size=(256, 256), scale_factor=(2, 2), mode="bilinear")

        self.up3 = nn.Sequential(
            ConvRelu2d(256, 128, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(128, 128, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.upSample4 = nn.Upsample(size=(128, 128), scale_factor=(2, 2), mode="bilinear")

        self.up4 = nn.Sequential(
            ConvRelu2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(64, 64, kernel_size=(3, 3), stride=1, padding=0)
        )

        # 1x1 convolution at the last layer
        self.output_seg_map = nn.Conv2d(64, 2, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x = self.maxPool1(self.down1(x))  # Calls the forward() method of each layer
        x = self.maxPool2(self.down2(x))
        x = self.maxPool3(self.down3(x))
        x = self.maxPool4(self.down4(x))

        x = self.center(x)

        x = self.up1(self.upSample1(x))
        x = self.up2(self.upSample2(x))
        x = self.up3(self.upSample3(x))
        x = self.up4(self.upSample4(x))

        out = self.output_seg_map(x)
        return out
