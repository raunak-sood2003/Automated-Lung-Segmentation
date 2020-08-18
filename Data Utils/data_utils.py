import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Conditional_DCGAN import Generator, Discriminator
from tqdm import tqdm

lung_CT = np.load('D:\DATA\VESSEL_DATA\lung_CT.npy')
lung_masks = np.load('D:\DATA\VESSEL_DATA\lung_masks.npy')

class Lung_CT_DATA(Dataset):
    def __init__(self, lungs_np, masks_np, split_type, transforms=None):
        self.lungs_np = lungs_np
        self.masks_np = masks_np
        self.MAX_VAL = 2 ** 15 - 1
        self.MIN_VAL = -4579
        self.val_split = 0.1
        self.val_num = int(self.val_split * self.lungs_np.shape[0])
        self.split_type = split_type
        self.transforms = transforms

        if self.split_type == 'val':
            self.val_lungs = torch.Tensor([i for i in tqdm(self.lungs_np[:self.val_num])]).view(-1, 1, 512, 512)
            self.val_lungs -= self.MIN_VAL
            self.val_lungs /= (self.MAX_VAL - self.MIN_VAL)
            self.val_masks = torch.Tensor([i for i in tqdm(self.masks_np[:self.val_num])]).view(-1, 1, 512, 512)
        elif self.split_type == 'test':
            self.test_lungs = torch.Tensor([i for i in tqdm(self.lungs_np[self.val_num:self.val_num * 2])]).view(-1, 1,
                                                                                                                 512,
                                                                                                                 512)
            self.test_lungs -= self.MIN_VAL
            self.test_lungs /= (self.MAX_VAL - self.MIN_VAL)
            self.test_masks = torch.Tensor([i for i in tqdm(self.masks_np[self.val_num:self.val_num * 2])]).view(-1, 1,
                                                                                                                 512,
                                                                                                                 512)
        elif self.split_type == 'train':
            self.train_lungs = torch.Tensor([i for i in tqdm(self.lungs_np[self.val_num * 2:])]).view(-1, 1, 512, 512)
            self.train_lungs -= self.MIN_VAL
            self.train_lungs /= (self.MAX_VAL - self.MIN_VAL)
            self.train_masks = torch.Tensor([i for i in tqdm(self.masks_np[self.val_num * 2:])]).view(-1, 1, 512, 512)

    def __len__(self):
        if self.split_type == 'train':
            return self.train_lungs.shape[0]
        elif self.split_type == 'val':
            return self.val_lungs.shape[0]
        elif self.split_type == 'test':
            return self.test_lungs.shape[0]

    def __getitem__(self, idx):
        if self.split_type == 'train':
            return self.train_lungs[idx], self.train_masks[idx]
        elif self.split_type == 'val':
            return self.val_lungs[idx], self.val_masks[idx]
        elif self.split_type == 'test':
            return self.test_lungs[idx], self.test_masks[idx]
