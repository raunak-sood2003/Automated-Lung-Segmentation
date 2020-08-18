import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

def normalize(torch_Tensor, min_val, max_val):
    for img in torch_Tensor:
        img[img < min_val] = min_val
        img[img > max_val] = max_val
    torch_Tensor -= min_val
    torch_Tensor /= (max_val - min_val)
    return torch_Tensor

class lungCT_segmentation_DATA(Dataset):
    def __init__(self, classifier_pairs, split_type, transforms = None):
        self.classifier_pairs = classifier_pairs
        self.split_type = split_type
        self.transforms = transforms
        self.val_split = 0.1
        self.num_split = int(self.val_split * self.classifier_pairs.shape[0])
        self.min_val = -1024
        self.max_val = 100
        
        np.random.shuffle(self.classifier_pairs)
            
        if self.split_type == 'test':
            self.X_test = torch.Tensor([i[0] for i in tqdm(self.classifier_pairs[:self.num_split])]).view(-1, 1, 512, 512)
            self.X_test = normalize(self.X_test, self.min_val, self.max_val)
            self.y_test = torch.Tensor([i[1] for i in tqdm(self.classifier_pairs[:self.num_split])]).view(-1, 1, 512, 512)
        elif self.split_type == 'val':
            self.X_val = torch.Tensor([i[0] for i in tqdm(self.classifier_pairs[self.num_split:self.num_split*2])]).view(-1, 1, 512, 512)
            self.X_val = normalize(self.X_val, self.min_val, self.max_val)
            self.y_val = torch.Tensor([i[1] for i in tqdm(self.classifier_pairs[self.num_split:self.num_split*2])]).view(-1, 1, 512, 512)
        elif self.split_type == 'train':
            self.X_train = torch.Tensor([i[0] for i in tqdm(self.classifier_pairs[self.num_split*2:])]).view(-1, 1, 512, 512)
            self.X_train = normalize(self.X_train, self.min_val, self.max_val)
            self.y_train = torch.Tensor([i[1] for i in tqdm(self.classifier_pairs[self.num_split*2:])]).view(-1, 1, 512, 512)
    
    def __len__(self):
        if self.split_type == 'train':
            return self.y_train.shape[0]
        elif self.split_type == 'val':
            return self.y_val.shape[0]
        elif self.split_type == 'test':
            return self.y_test.shape[0]
    
    def __getitem__(self, idx):
        if self.split_type == 'train':
            return self.X_train[idx], self.y_train[idx]
        elif self.split_type == 'val':
            return self.X_val[idx], self.y_val[idx]
        elif self.split_type == 'test':
            return self.X_test[idx], self.y_test[idx]

if __name__ == '__main__':
    data_dir = 'D:/DATA/VESSEL_DATA/middle_lung.npy'
    data_npy = np.load(data_dir)

    train = lungCT_segmentation_DATA(data_npy, 'train')
    val = lungCT_segmentation_DATA(data_npy, 'val')
    test = lungCT_segmentation_DATA(data_npy, 'test')

    train_loader = DataLoader(train, batch_size = 8, shuffle = True)
    val_loader = DataLoader(val, batch_size = 8, shuffle = True)
    test_loader = DataLoader(test, batch_size = 8, shuffle = True)

    print("Train Loader Length: %s\nVal Loader Length: %s\nTest Loader Length: %s\n"
          % (train_loader.__len__(), val_loader.__len__(), test_loader.__len__()))

    out_img = -1
    out_mask = -1
    for img, mask in train_loader:
        out_img = img
        out_mask = mask
        break

    print("Img Batch Shape:", out_img.size())
    print("Mask Batch Shape:", out_mask.size())

    print("Max Value of img:", torch.max(out_img[0].view(512, 512)))
    print("Max Value of mask:", torch.max(out_mask[0].view(512, 512)))

    print("Min Value of img:", torch.min(out_img[0].view(512, 512)))
    print("Min Value of mask:", torch.min(out_mask[0].view(512, 512)))
