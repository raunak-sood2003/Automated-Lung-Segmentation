import torch
import torch.nn as nn
import numpy as np
from UNET import UNET
from data_loading import lungCT_segmentation_DATA
from torch.utils.data import DataLoader

# Hyperparameters

epochs = 1
batch_size = 8
lr = 0.01
device = torch.device("cuda")
# Data Loading

lung_CT = np.load('/home/rrsood003/DATA/middle_lung.npy')

train = lungCT_segmentation_DATA(lung_CT, 'train')
test = lungCT_segmentation_DATA(lung_CT, 'test')

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

# Training

model = UNET().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(1, epochs+1):
    train_loss = 0
    val_loss = 0
    for idx, data in enumerate(train_loader):
        model.train()
        X_train, y_train = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("{%s}/{%s}\t{%s}/{%s}\tLoss: %s"
              % (epoch, epochs, idx, len(train_loader), train_loss / len(train_loader)))
    for idx, data in enumerate(val_loader):
        model.eval()
        X_val, y_val = data[0].to(device), data[1].to(device)
        output_val = model(X_val)
        loss_val = criterion(output_val, y_val)
        val_loss += loss_val.item()
        print("{%s}/{%s}\tVal Loss: %s"
              % (epoch, epochs, val_loss / len(val_loader)))



