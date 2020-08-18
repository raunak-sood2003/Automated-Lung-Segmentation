import torch
import torch.nn as nn
from collections import OrderedDict

class UGenerator(nn.Module):

    def __init__(self, in_c, out_c):
        super(UGenerator, self).__init__()
        self.max_pool2D = nn.MaxPool2d(2)
        self.down_conv1 = self.double_conv(in_c, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        self.down_conv5 = self.double_conv(512, 1024)

        self.up_conv1 = self.up_conv(1024, 512)
        self.down_convDecoder1 = self.double_conv(1024, 512)

        self.up_conv2 = self.up_conv(512, 256)
        self.down_convDecoder2 = self.double_conv(512, 256)

        self.up_conv3 = self.up_conv(256, 128)
        self.down_convDecoder3 = self.double_conv(256, 128)

        self.up_conv4 = self.up_conv(128, 64)
        self.down_convDecoder4 = self.double_conv(128, 64)

        self.out = nn.Conv2d(64, out_c, kernel_size = 1)

    def up_conv(self, in_c, out_c, kernel_size = 2, stride = 2):
        up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True))
        return up_conv
    def double_conv(self, in_c, out_c, kernel_size = 3, padding = 1):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True))
        return conv

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x) # Ouputs passed to decoder
        p1 = self.max_pool2D(x1)
        x2 = self.down_conv2(p1) # Ouputs passed to decoder
        p2 = self.max_pool2D(x2)
        x3 = self.down_conv3(p2) # Ouputs passed to decoder
        p3 = self.max_pool2D(x3)
        x4 = self.down_conv4(p3) # Ouputs passed to decoder
        p4 = self.max_pool2D(x4)
        x5 = self.down_conv5(p4)

        # Decoder
        p6 = self.up_conv1(x5) # 48
        x6 = torch.cat([p6, x4], dim = 1)
        x6 = self.down_convDecoder1(x6) # 44

        p7 = self.up_conv2(x6) # 88
        x7 = torch.cat([p7, x3], dim = 1)
        x7 = self.down_convDecoder2(x7)

        p8 = self.up_conv3(x7)
        x8 = torch.cat([p8, x2], dim = 1)
        x8 = self.down_convDecoder3(x8)

        p9 = self.up_conv4(x8)
        x9 = torch.cat([p9, x1], dim = 1)
        x9 = self.down_convDecoder4(x9)

        output = self.out(x9)
        output = torch.sigmoid(output)

        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels, kernel_size=(4, 4), padding=1, conv_stride=2):
        super(Discriminator, self).__init__()

        self.leaky_alpha = 0.2

        # INPUT (1, 512, 512)

        self.convblock1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 2, kernel_size, conv_stride, padding)),
            ('activation1', nn.LeakyReLU(self.leaky_alpha))
        ])) # (2, 256, 256)

        self.convblock2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(2, 4, kernel_size, conv_stride, padding)),
            ('activation2', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm2', nn.BatchNorm2d(4))
        ])) # (4, 128, 128)

        self.convblock3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(4, 8, kernel_size, conv_stride, padding)),
            ('activation3', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm3', nn.BatchNorm2d(8))
        ])) # (8, 64, 64)

        self.convblock4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(8, 16, kernel_size, conv_stride, padding)),
            ('activation4', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm4', nn.BatchNorm2d(16))
        ])) # (16, 32, 32)

        self.convblock5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(16, 32, kernel_size, conv_stride, padding)),
            ('activation5', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm5', nn.BatchNorm2d(32))
        ])) # (32, 16, 16)

        self.convblock6 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(32, 64, kernel_size, conv_stride, padding)),
            ('activation6', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm6', nn.BatchNorm2d(64))
        ])) # (64, 8, 8)

        self.convblock7 = nn.Sequential(OrderedDict([
            ('conv7', nn.Conv2d(64, 128, kernel_size, conv_stride, padding)),
            ('activation7', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm7', nn.BatchNorm2d(128))
        ])) # (128, 4, 4)

        self.convblock8 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(128, 256, kernel_size, conv_stride, padding)),
            ('activation8', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm8', nn.BatchNorm2d(256))
        ])) # (256, 2, 2)

        self.convblock9 = nn.Sequential(OrderedDict([
            ('conv9', nn.Conv2d(256, 1, kernel_size, conv_stride, padding)),
            ('activation9', nn.Sigmoid()),

        ])) # (1, 1, 1)

    def forward(self, X):
        X = self.convblock1(X)
        X = self.convblock2(X)
        X = self.convblock3(X)
        X = self.convblock4(X)
        X = self.convblock5(X)
        X = self.convblock6(X)
        X = self.convblock7(X)
        X = self.convblock8(X)
        X = self.convblock9(X)

        return X

if __name__ == '__main__':
    net_G = UGenerator(1, 1)
    net_D = Discriminator(1)

    rand = torch.rand(1, 1, 512, 512)
    gen_out = net_G(rand)
    dis_out = net_D(gen_out)
    print(gen_out.size())
    print(dis_out.size())