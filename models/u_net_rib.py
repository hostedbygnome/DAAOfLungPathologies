import torch as t
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.d_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
        # x = self.d_conv(inputs)
        # return x


# class _EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=False):
#         super(_EncoderBlock, self).__init__()
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         ]
#         if dropout:
#             layers.append(nn.Dropout())
#         # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encode = nn.Sequential(*layers)
#
#     def forward(self, inputs):
#         x = self.encode(inputs)
#         p = self.pool(x)
#         return x, p
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        self.d_conv = _ConvBlock(in_channels, out_channels)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.d_conv(inputs)
        p = self.pool(x)
        return x, p

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.d_conv = _ConvBlock(in_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = t.cat([x, skip], 1)
        x = self.d_conv(x)
        return x


class UNetRib(nn.Module):
    def __init__(self):
        super(UNetRib, self).__init__()
        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _ConvBlock(512, 1024)
        self.dec4 = _DecoderBlock(1024, 512)
        self.dec3 = _DecoderBlock(512, 256)
        self.dec2 = _DecoderBlock(256, 128)
        self.dec1 = _DecoderBlock(128, 64)
        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        # init_weights(self)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        center = self.center(p4)

        d4 = self.dec4(center, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = t.sigmoid(self.out(d1))
        return out
