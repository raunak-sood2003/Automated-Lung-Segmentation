import torch
import torch.nn as nn
import torch.nn.functional as F
from clsfr_data_loading import lungCT_clsfr_DATA
import numpy as np

class lungPartitionClsfr(nn.Module):
    def __init__(self, in_c, out_c):
        super(lungPartitionClsfr, self).__init__()
        self.max_pool2d = nn.MaxPool2d(2)
        self.down_conv1 = self.double_conv(in_c, 64)
        self.down_conv2 = self.double_conv(64, 64)
        self.down_conv3 = self.double_conv(64, 128)
        self.down_conv4 = self.double_conv(128, 128)
        self.linear1 = nn.Linear(28*28*128, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, out_c)
        
    def double_conv(self, in_c, out_c, kernel_size = 3, padding = 0):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True))
        return conv
    
    def forward(self, x0):
        x1 = self.down_conv1(x0)
        p1 = self.max_pool2d(x1)
        x2 = self.down_conv2(p1)
        p2 = self.max_pool2d(x2)
        x3 = self.down_conv3(p2)
        p3 = self.max_pool2d(x3)
        x4 = self.down_conv4(p3)
        p4 = self.max_pool2d(x4)
        p4 = p4.view(-1, 28*28*128)
        x5 = self.linear1(p4)
        x5 = F.relu(x5)
        x6 = self.linear2(x5)
        x6 = F.relu(x6)
        x7 = self.linear3(x6)
        x7 = torch.softmax(x7, 1)
        
        return x7

if __name__ == '__main__':
    ex_tensor = torch.Tensor(1, 1, 512, 512)
    model = lungPartitionClsfr(1, 3)
    out_tensor = model(ex_tensor)
    print("Shape:", out_tensor.size())
    print(out_tensor)