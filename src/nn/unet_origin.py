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


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvRelu2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.convr2 = ConvRelu2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.down_blueprint = None

    def get_down_blueprint(self):
        """
            A method which returns the Tensor after the Conv/Relu operations but
            before the maxpooling
        Returns:
            nn.Variable: The blueprint Tensor
        """
        return self.down_blueprint

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        self.down_blueprint = x
        x = self.maxPool(x)
        return x


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(StackDecoder, self).__init__()

        self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2, 2), mode="bilinear")
        self.convr1 = ConvRelu2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        # Crop + concat step between these 2
        self.convr2 = ConvRelu2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        x = self.convr1(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr2(x)
        return x


class UNetOriginal(nn.Module):
    def __init__(self, in_shape):
        super(UNetOriginal, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(channels, 64)
        self.down2 = StackEncoder(64, 128)
        self.down3 = StackEncoder(128, 256)
        self.down4 = StackEncoder(256, 512)

        self.center = nn.Sequential(
            ConvRelu2d(512, 1024, kernel_size=(3, 3), stride=1, padding=0),
            ConvRelu2d(1024, 1024, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, upsample_size=(56, 56))
        self.up2 = StackDecoder(in_channels=512, out_channels=256, upsample_size=(104, 104))
        self.up3 = StackDecoder(in_channels=256, out_channels=128, upsample_size=(200, 200))
        self.up4 = StackDecoder(in_channels=128, out_channels=64, upsample_size=(392, 392))

        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x = self.down1(x)  # Calls the forward() method of each layer
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.center(x)

        x = self.up1(x, self.down4.get_down_blueprint())
        x = self.up2(x, self.down3.get_down_blueprint())
        x = self.up3(x, self.down2.get_down_blueprint())
        x = self.up4(x, self.down1.get_down_blueprint())

        out = self.output_seg_map(x)
        out = torch.squeeze(out, dim=1)
        return out
