import torch
import torch.nn as nn

class UNET(nn.Module):

    def __init__(self, in_c, out_c):
        super(UNET, self).__init__()
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


if __name__ == '__main__':

    input_tensor = torch.rand(1, 1, 512, 512)
    unet = UNET(1, 1)
    print(unet(input_tensor).size())
