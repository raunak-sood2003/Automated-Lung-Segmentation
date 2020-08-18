import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from math import *

class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.max_pool2D = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_conv1 = self.double_conv(1, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        self.down_conv5 = self.double_conv(512, 1024)

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.down_convDecoder1 = self.double_conv(1024, 512)

        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.down_convDecoder2 = self.double_conv(512, 256)

        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.down_convDecoder3 = self.double_conv(256, 128)

        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.down_convDecoder4 = self.double_conv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size = 1)


    def double_conv(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True))
        return conv

    def crop_tensor(self, tensor, target_tensor):
        # tensor is larger than target_tensor
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta_2 = delta // 2

        if delta % 2 == 0:
            return tensor[:, :, delta_2:tensor_size-delta_2, delta_2:tensor_size-delta_2]
        else:
            return tensor[:, :, delta_2:tensor_size - delta_2-1, delta_2:tensor_size - delta_2-1]




    def forward(self, x_in):
        # Encoder
        x1 = self.down_conv1(x_in) # Ouputs passed to decoder
        x2 = self.max_pool2D(x1)
        x3 = self.down_conv2(x2) # Ouputs passed to decoder
        x4 = self.max_pool2D(x3)
        x5 = self.down_conv3(x4) # Ouputs passed to decoder
        x6 = self.max_pool2D(x5)
        x7 = self.down_conv4(x6) # Ouputs passed to decoder
        x8 = self.max_pool2D(x7)
        x9 = self.down_conv5(x8)

        # Decoder
        x = self.up_conv1(x9) # 48
        x_cropped = self.crop_tensor(x7, x)
        x = self.down_convDecoder1(torch.cat([x, x_cropped], 1)) # 44

        x = self.up_conv2(x) # 88
        x_cropped = self.crop_tensor(x5, x)
        x = self.down_convDecoder2(torch.cat([x, x_cropped], 1))

        x = self.up_conv3(x)
        x_cropped = self.crop_tensor(x3, x)
        x = self.down_convDecoder3(torch.cat([x, x_cropped], 1))

        x = self.up_conv4(x)
        x_cropped = self.crop_tensor(x1, x)
        x = self.down_convDecoder4(torch.cat([x, x_cropped], 1))

        x = self.out(x)
        x = torch.sigmoid(x)

        return x


if __name__ == '__main__':

    input_tensor = torch.rand(16, 1, 512, 512)
    unet = UNET()
    print(unet(input_tensor).size())






