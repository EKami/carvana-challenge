import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu2d, self).__init__()
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

        self.down1 = ConvBnRelu2d(channels, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.down2 = ConvBnRelu2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.down3 = ConvBnRelu2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)

        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.down4 = ConvBnRelu2d(64, 128, kernel_size=(3, 3), stride=2, padding=0)
        self.down5 = ConvBnRelu2d(128, 128, kernel_size=(3, 3), stride=2, padding=0)
        self.down6 = ConvBnRelu2d(128, 128, kernel_size=(3, 3), stride=2, padding=0)

        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.down7 = ConvBnRelu2d(128, 256, kernel_size=(3, 3), stride=2, padding=0)
        self.down8 = ConvBnRelu2d(256, 256, kernel_size=(3, 3), stride=2, padding=0)
        self.down9 = ConvBnRelu2d(256, 256, kernel_size=(3, 3), stride=2, padding=0)

        self.maxPool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.down10 = ConvBnRelu2d(256, 512, kernel_size=(3, 3), stride=2, padding=0)
        self.down11 = ConvBnRelu2d(512, 512, kernel_size=(3, 3), stride=2, padding=0)
        self.down12 = ConvBnRelu2d(512, 512, kernel_size=(3, 3), stride=2, padding=0)

        self.maxPool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center1 = ConvBnRelu2d(512, 1024, kernel_size=(3, 3), stride=2, padding=0)
        self.center2 = ConvBnRelu2d(1024, 1024, kernel_size=(3, 3), stride=2, padding=0)

        self.upSample1 = nn.Upsample(size=(1024, 1024), scale_factor=(2, 2))  # TODO Bilinear?

        self.up1 = ConvBnRelu2d(1024, 512, kernel_size=(3, 3), stride=2, padding=0)
        self.up2 = ConvBnRelu2d(512, 512, kernel_size=(3, 3), stride=2, padding=0)
        self.up3 = ConvBnRelu2d(512, 512, kernel_size=(3, 3), stride=2, padding=0)

        self.upSample2 = nn.Upsample(size=(512, 512), scale_factor=(2, 2))

        self.up4 = ConvBnRelu2d(512, 256, kernel_size=(3, 3), stride=2, padding=0)
        self.up5 = ConvBnRelu2d(256, 256, kernel_size=(3, 3), stride=2, padding=0)
        self.up6 = ConvBnRelu2d(256, 256, kernel_size=(3, 3), stride=2, padding=0)

        self.upSample3 = nn.Upsample(size=(256, 256), scale_factor=(2, 2))

        self.up7 = ConvBnRelu2d(256, 128, kernel_size=(3, 3), stride=2, padding=0)
        self.up8 = ConvBnRelu2d(128, 128, kernel_size=(3, 3), stride=2, padding=0)
        self.up9 = ConvBnRelu2d(128, 128, kernel_size=(3, 3), stride=2, padding=0)

        self.upSample4 = nn.Upsample(size=(128, 128), scale_factor=(2, 2))

        self.up10 = ConvBnRelu2d(128, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.up11 = ConvBnRelu2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)
        self.up12 = ConvBnRelu2d(64, 64, kernel_size=(3, 3), stride=2, padding=0)

        self.output_seg_map = nn.Conv2d(64, 2, kernel_size=(3, 3), padding=0, stride=2)

    def forward(self, x):
        x = self.down1.forward(x)
        x = self.down2.forward(x)
        x = self.down3.forward(x)
        d = 0
