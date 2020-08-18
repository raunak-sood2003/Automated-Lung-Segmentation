import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from UGAN import UGenerator, Discriminator
from data_loading import lungCT_segmentation_DATA
from tqdm import tqdm
from model_tools import iou_compute
import time
import os

lung_CT = np.load('/home/rrsood003/DATA/middle_lung.npy')

train = lungCT_segmentation_DATA(lung_CT, 'train')
val = lungCT_segmentation_DATA(lung_CT, 'val')

BATCH_SIZE = 5
train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_save = True
EPOCHS = 1
IN_CHANNELS = 1
OUT_CHANNELS = 1
net_G = UGenerator(IN_CHANNELS, OUT_CHANNELS).to(device)
net_D = Discriminator(IN_CHANNELS).to(device)
optimizer_G = torch.optim.Adam(net_G.parameters())
optimizer_D = torch.optim.Adam(net_D.parameters())
criterion = nn.BCELoss()
net_G.train()
net_D.train()
loss_discriminator_list = []
loss_generator_list = []
loss_discriminator_val_list = []
loss_generator_val_list = []
iou_train_list = []
iou_val_list = []
for epoch in range(1, EPOCHS + 1):
    loss_discriminator = 0
    loss_generator = 0
    iou_train = 0
    loss_discriminator_val = 0
    loss_generator_val = 0
    iou_val = 0
    for idx, batch in tqdm(enumerate(train_loader)):
        imgs, masks = batch[0].to(device), batch[1].to(device)
        # Discriminator
        labels_real = (torch.ones(BATCH_SIZE) * 0.9).reshape(BATCH_SIZE,).to(device)
        net_D.zero_grad()
        output_real = net_D(masks).reshape(BATCH_SIZE,)
        loss_real = criterion(output_real, labels_real)

        gen_out = net_G(imgs)
        labels_fake = (torch.ones(BATCH_SIZE) * 0.1).reshape(BATCH_SIZE,).to(device)
        output_fake = net_D(gen_out.detach()).reshape(BATCH_SIZE,)
        loss_fake = criterion(output_fake, labels_fake)
        loss_D = loss_fake + loss_real
        loss_D.backward()
        optimizer_D.step()
        loss_discriminator += loss_D.item()

        #Generator
        net_G.zero_grad()
        iou_gen = iou_compute(masks, gen_out)
        iou_train += iou_gen
        labels_gen = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE,).to(device)
        output_fake_gen = net_D(gen_out).reshape(BATCH_SIZE,)
        loss_G = criterion(output_fake_gen, labels_gen)
        loss_G.backward()
        optimizer_G.step()
        loss_generator += loss_G.item()

    for batch in tqdm(val_loader):
        imgs, masks = batch[0].to(device), batch[1].to(device)
        # Discriminator
        labels_real = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE,).to(device)
        output_real = net_D(masks).reshape(BATCH_SIZE,)
        loss_real = criterion(output_real, labels_real)

        gen_out = net_G(imgs)
        labels_fake = torch.zeros(BATCH_SIZE).reshape(BATCH_SIZE,).to(device)
        output_fake = net_D(gen_out.detach()).reshape(BATCH_SIZE,)
        loss_fake = criterion(output_fake, labels_fake)
        loss_D = loss_fake + loss_real
        loss_discriminator_val += loss_D.item()
        # Generator
        iou_gen = iou_compute(gen_out, masks)
        iou_val += iou_gen
        labels_gen = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE,).to(device)
        output_fake_gen = net_D(gen_out).reshape(BATCH_SIZE,)
        loss_G = criterion(output_fake_gen, labels_gen)
        loss_generator_val += loss_G.item()

    loss_generator /= len(train_loader)
    loss_discriminator /= len(train_loader)
    loss_generator_val /= len(val_loader)
    loss_discriminator_val /= len(val_loader)
    iou_train /= (BATCH_SIZE * len(train_loader))
    iou_val /= (BATCH_SIZE * len(val_loader))

    loss_discriminator_list.append(loss_discriminator)
    loss_generator_list.append(loss_generator)
    loss_discriminator_val_list.append(loss_discriminator_val)
    loss_generator_val_list.append(loss_generator_val)
    iou_train_list.append(iou_train)
    iou_val_list.append(iou_val)

    print("Epoch [%s]/[%s]\tTrain Generator Loss: %s\tTrain Discriminator Loss: %s\tTrain IOU: %s\tVal Generator Loss: %s\tVal Discriminator Loss: %s\tVal IOU: %s"
          % (epoch, EPOCHS, loss_generator, loss_discriminator, iou_train, loss_generator_val, loss_discriminator_val, iou_val))

if to_save:
    date = time.strftime("%D")
    save_dir = 'UGAN_%s_' % (iou_val_list[-1]) + date.replace("/", "-")
    save_path = os.path.join('/home/rrsood003/Segmentation/saved_models', save_dir)
    os.mkdir(save_path)
    model_file = save_dir + '.pth'
    model_dir = os.path.join(save_path, model_file)
    torch.save(net_G.state_dict(), model_dir)
    np.save(save_path + 'loss_discriminator_train.npy', np.array(loss_discriminator_list))
    np.save(save_path + 'loss_generator_train.npy', np.array(loss_generator_list))
    np.save(save_path + 'loss_discriminator_val.npy', np.array(loss_discriminator_val_list))
    np.save(save_path + 'iou_train.npy', np.array(iou_train_list))
    np.save(save_path + 'iou_val.npy', np.array(iou_val_list))
    print("Saved Model and Metrics Successfully")
