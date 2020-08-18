import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import OrderedDict


class Generator(nn.Module):
    def __init__(self, in_channels, kernel_size=(5, 5), pool_size=(2, 2), padding = 2, conv_stride = 1,
                 pool_stride=2):
        super(Generator, self).__init__()

        self.dropout_proba = 1

        #INPUT: (1, 512, 512)

        self.downblock1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 4, kernel_size, conv_stride, padding)),
            ('activation1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm1', nn.BatchNorm2d(4)),
            ('dropout1', nn.Dropout(self.dropout_proba))
        ]))  # (4, 256, 256)

        self.downblock2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(4, 16, kernel_size, conv_stride, padding)),
            ('activation2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm2', nn.BatchNorm2d(16)),
            ('dropout2', nn.Dropout(self.dropout_proba))
        ])) # (16, 128, 128)

        self.downblock3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(16, 32, kernel_size, conv_stride, padding)),
            ('activation3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm3', nn.BatchNorm2d(32)),
            ('dropout3', nn.Dropout(self.dropout_proba))
        ])) # (32, 64, 64)

        self.downblock4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(32, 64, kernel_size, conv_stride, padding)),
            ('activation4', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm4', nn.BatchNorm2d(64)),
            ('dropout4', nn.Dropout(self.dropout_proba))
        ])) # (64, 32, 32)

        self.downblock5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(64, 128, kernel_size, conv_stride, padding)),
            ('activation5', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm5', nn.BatchNorm2d(128)),
            ('dropout5', nn.Dropout(self.dropout_proba))
        ])) # (128, 16, 16)

        self.downblock6 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(128, 256, kernel_size, conv_stride, padding)),
            ('activation6', nn.ReLU()),
            ('maxpool6', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm6', nn.BatchNorm2d(256)),
            ('dropout6', nn.Dropout(self.dropout_proba))
        ])) # (256, 8, 8)

        self.downblock7 = nn.Sequential(OrderedDict([
            ('conv7', nn.Conv2d(256, 512, kernel_size, conv_stride, padding)),
            ('activation7', nn.ReLU()),
            ('maxpool7', nn.MaxPool2d(pool_size, pool_stride)),
            ('batchnorm7', nn.BatchNorm2d(512)),
            ('dropout7', nn.Dropout(self.dropout_proba))
        ])) # (512, 4, 4)

        self.upblock1 = nn.Sequential(OrderedDict([
            ('up_conv1', nn.ConvTranspose2d(512, 256, kernel_size, conv_stride, padding)),
            ('up_activation1', nn.ReLU()),
            ('upsample1', nn.Upsample((8, 8))),
            ('up_batchnorm1', nn.BatchNorm2d(256)),
            ('up_dropout1', nn.Dropout(self.dropout_proba))
        ])) # (256, 8, 8)

        self.upblock2 = nn.Sequential(OrderedDict([
            ('up_conv2', nn.ConvTranspose2d(256, 128, kernel_size, conv_stride, padding)),
            ('up_activation2', nn.ReLU()),
            ('upsample2', nn.Upsample((16, 16))),
            ('up_batchnorm2', nn.BatchNorm2d(128)),
            ('up_dropout2', nn.Dropout(self.dropout_proba))
        ])) # (128, 16, 16)

        self.upblock3 = nn.Sequential(OrderedDict([
            ('up_conv3', nn.ConvTranspose2d(128, 64, kernel_size, conv_stride, padding)),
            ('up_activation3', nn.ReLU()),
            ('upsample3', nn.Upsample((32, 32))),
            ('up_batchnorm3', nn.BatchNorm2d(64)),
            ('up_dropout3', nn.Dropout(self.dropout_proba))
        ])) # (64, 32, 32)

        self.upblock4 = nn.Sequential(OrderedDict([
            ('up_conv4', nn.ConvTranspose2d(64, 32, kernel_size, conv_stride, padding)),
            ('up_activation4', nn.ReLU()),
            ('upsample4', nn.Upsample((64, 64))),
            ('up_batchnorm4', nn.BatchNorm2d(32)),
            ('up_dropout4', nn.Dropout(self.dropout_proba))
        ])) # (32, 64, 64)

        self.upblock5 = nn.Sequential(OrderedDict([
            ('up_conv5', nn.ConvTranspose2d(32, 16, kernel_size, conv_stride, padding)),
            ('up_activation5', nn.ReLU()),
            ('upsample5', nn.Upsample((128, 128))),
            ('up_batchnorm5', nn.BatchNorm2d(16)),
            ('up_dropout5', nn.Dropout(self.dropout_proba))
        ])) # (16, 128, 128)

        self.upblock6 = nn.Sequential(OrderedDict([
            ('up_conv6', nn.ConvTranspose2d(16, 4, kernel_size, conv_stride, padding)),
            ('up_activation6', nn.ReLU()),
            ('upsample6', nn.Upsample((256, 256))),
            ('up_batchnorm6', nn.BatchNorm2d(4)),
            ('up_dropout6', nn.Dropout(self.dropout_proba))
        ])) # (4, 256, 256)

        self.upblock7 = nn.Sequential(OrderedDict([
            ('up_conv7', nn.ConvTranspose2d(4, 1, kernel_size, conv_stride, padding)),
            ('up_activation7', nn.ReLU()),
            ('upsample7', nn.Upsample((512, 512))),
            # NO BATCHNORM
            # NO DROPOUT
        ])) # (1, 512, 512)


    def forward(self, X):

        X = self.downblock1(X)
        X = self.downblock2(X)
        X = self.downblock3(X)
        X = self.downblock4(X)
        X = self.downblock5(X)
        X = self.downblock6(X)
        X = self.downblock7(X)

        X = self.upblock1(X)
        X = self.upblock2(X)
        X = self.upblock3(X)
        X = self.upblock4(X)
        X = self.upblock5(X)
        X = self.upblock6(X)
        X = self.upblock7(X)
        
        return X

class Discriminator(nn.Module):
    def __init__(self, in_channels, kernel_size=(4, 4), padding=1, conv_stride=2):
        super(Discriminator, self).__init__()

        self.dropout_proba = 1
        self.leaky_alpha = 0.1

        # INPUT (1, 512, 512)

        self.convblock1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 2, kernel_size, conv_stride, padding)),
            ('activation1', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm1', nn.BatchNorm2d(2)),
            ('dropout1', nn.Dropout(self.dropout_proba))
        ])) # (2, 256, 256)

        self.convblock2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(2, 4, kernel_size, conv_stride, padding)),
            ('activation2', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm2', nn.BatchNorm2d(4)),
            ('dropout2', nn.Dropout(self.dropout_proba))
        ])) # (4, 128, 128)

        self.convblock3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(4, 8, kernel_size, conv_stride, padding)),
            ('activation3', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm3', nn.BatchNorm2d(8)),
            ('dropout3', nn.Dropout(self.dropout_proba))
        ])) # (8, 64, 64)

        self.convblock4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(8, 16, kernel_size, conv_stride, padding)),
            ('activation4', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm4', nn.BatchNorm2d(16)),
            ('dropout4', nn.Dropout(self.dropout_proba))
        ])) # (16, 32, 32)

        self.convblock5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(16, 32, kernel_size, conv_stride, padding)),
            ('activation5', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm5', nn.BatchNorm2d(32)),
            ('dropout5', nn.Dropout(self.dropout_proba))
        ])) # (32, 16, 16)

        self.convblock6 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(32, 64, kernel_size, conv_stride, padding)),
            ('activation6', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm6', nn.BatchNorm2d(64)),
            ('dropout6', nn.Dropout(self.dropout_proba))
        ])) # (64, 8, 8)

        self.convblock7 = nn.Sequential(OrderedDict([
            ('conv7', nn.Conv2d(64, 128, kernel_size, conv_stride, padding)),
            ('activation7', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm7', nn.BatchNorm2d(128)),
            ('dropout7', nn.Dropout(self.dropout_proba))
        ])) # (128, 4, 4)

        self.convblock8 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(128, 256, kernel_size, conv_stride, padding)),
            ('activation8', nn.LeakyReLU(self.leaky_alpha)),
            ('batchnorm8', nn.BatchNorm2d(256)),
            ('dropout8', nn.Dropout(self.dropout_proba))
        ])) # (256, 2, 2)

        self.convblock9 = nn.Sequential(OrderedDict([
            ('conv9', nn.Conv2d(256, 1, kernel_size, conv_stride, padding)),
            ('activation9', nn.Sigmoid()),
            # NO BATCHNORM
            # NO DROPOUT
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
    dummy = torch.rand(1, 1, 512, 512)
    gen = Generator(1)
    dis = Discriminator(1)
    gen_out = gen(dummy)
    dis_out = dis(gen_out)
    print(gen_out.shape)
    print(dis_out.shape)