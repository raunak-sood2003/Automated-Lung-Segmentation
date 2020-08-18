import torch
import torch.nn as nn
import numpy as np
from UNET import UNET
from data_loading import lungCT_segmentation_DATA
from model_tools import iou_compute, DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

# Hyperparameters
epochs = 15
batch_size = 5
lr = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_save = True
# Data Loading
lung_CT = np.load('/home/rrsood003/DATA/middle_lung.npy')

train = lungCT_segmentation_DATA(lung_CT, 'train')
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)

val = lungCT_segmentation_DATA(lung_CT, 'val')
val_loader = DataLoader(val, batch_size = batch_size, shuffle = True)

# Training
model = UNET(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
val_losses = []
train_ious = []
val_ious = []
print("Starting Training...\nEpochs: %s\nBatch Size: %s\nLoss Function: %s\nMetrics: %s" 
        % (epochs, batch_size, "MSE Loss", ['train loss', 'val loss', 'train IOU', 'val IOU']))
for epoch in range(1, epochs+1):
    train_loss = 0
    val_loss = 0
    train_iou = 0
    val_iou = 0
    model.train()
    for idx, data in tqdm(enumerate(train_loader)):
        X_train, y_train = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(X_train)
        batch_iou = iou_compute(y_train, output)
        train_iou += batch_iou
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            X_val, y_val = batch[0].to(device), batch[1].to(device)
            output_val = model(X_val)
            loss_val = criterion(output_val, y_val)
            val_loss += loss_val.item()
            batch_iou_val = iou_compute(y_val, output_val)
            val_iou += batch_iou_val

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_iou /= (batch_size * len(train_loader))
    val_iou /= (batch_size * len(val_loader))

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_ious.append(train_iou)
    val_ious.append(val_iou)

    print("Epoch [%s]/[%s]\tTrain Loss: %s\tVal Loss: %s\tTrain IOU: %s\tVal IOU: %s"
            % (epoch, epochs, train_loss, val_loss, train_iou, val_iou))

if to_save:
    date = time.strftime("%D")
    save_dir = 'UNET_%s_' % round(val_ious[-1], 3) + 'BCE' + date.replace('/', '-')
    combined_dir = os.path.join('./saved_models', save_dir)
    os.mkdir(combined_dir)
    model_file = os.path.join(combined_dir, save_dir + '.pth')
    torch.save(model.state_dict(), model_file)
    np.save(combined_dir + '/train_losses.npy', np.array(train_losses))
    np.save(combined_dir + '/val_losses.npy', np.array(val_losses))
    np.save(combined_dir + '/train_ious.npy', np.array(train_ious))
    np.save(combined_dir + '/val_ious.npy', np.array(val_ious))



