import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from UGAN import UGenerator, Discriminator
from UNET import UNET
from data_loading import lungCT_segmentation_DATA
from tqdm import tqdm
from model_tools import iou_compute
from config import config
import argparse
import time
import os

def unet_train(lung_CT_dir, in_c = 1, out_c = 1, epochs = 15, batch_size = 5, criterion = nn.BCELoss(), lr = 0.01, device = torch.device("cuda:0"), to_save = True):
    '''
    Training function for UNET model
    :param lung_CT_dir: Directory to numpy array containing paired lung CT scans and masks; shape: (n, 2, 512, 512)
    :param in_c: input channels to model
    :param out_c: output classes segmented
    :param epochs: number of epochs
    :param batch_size: batch size fed per iteration
    :param criterion: loss function used while training
    :param lr: learning rate for optimization
    :param device: cpu/gpu (cuda) devices
    :param to_save: boolean, if true, will save model (.pth) as well as loss and accuracies per epoch
    :return: Void
    '''
    lung_CT = np.load(lung_CT_dir)

    train = lungCT_segmentation_DATA(lung_CT, 'train')
    train_loader = DataLoader(train, batch_size = batch_size, shuffle=True)

    val = lungCT_segmentation_DATA(lung_CT, 'val')
    val_loader = DataLoader(val, batch_size = batch_size, shuffle=True)

    model = UNET(in_c, out_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    print("Starting Training...\nEpochs: %s\nBatch Size: %s\nLoss Function: %s\nMetrics: %s"
          % (epochs, batch_size, "MSE Loss", ['train loss', 'val loss', 'train IOU', 'val IOU']))

    for epoch in range(1, epochs + 1):
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
        save_dir = 'UNET_%s_' % round(val_ious[-1], 3) + 'BCE_' + date.replace('/', '-')
        if not os.path.isdir('./saved_models'):
            os.mkdir('./saved_models')
        combined_dir = os.path.join('./saved_models', save_dir)
        os.mkdir(combined_dir)
        model_file = os.path.join(combined_dir, save_dir + '.pth')
        torch.save(model.state_dict(), model_file)
        np.save(combined_dir + '/train_losses.npy', np.array(train_losses))
        np.save(combined_dir + '/val_losses.npy', np.array(val_losses))
        np.save(combined_dir + '/train_ious.npy', np.array(train_ious))
        np.save(combined_dir + '/val_ious.npy', np.array(val_ious))
        print("Saved Model and Metrics Successfully")

def UGAN_train(lung_CT_dir, in_c  = 1, out_c = 1, epochs = 15, batch_size = 5, criterion = nn.BCELoss(), lr = 0.01, device = torch.device("cuda:0"), to_save = True):
     '''
     Training function for UGAN model
    :param lung_CT_dir: Directory to numpy array containing paired lung CT scans and masks; shape: (n, 2, 512, 512)
    :param in_c: input channels to model
    :param out_c: output classes segmented
    :param epochs: number of epochs
    :param batch_size: batch size fed per iteration
    :param criterion: loss function used while training
    :param lr: learning rate for optimization
    :param device: cpu/gpu (cuda) devices
    :param to_save: boolean, if true, will save model (.pth) as well as loss and accuracies per epoch
    :return: Void
    '''
    lung_CT = np.load(lung_CT_dir)

    train = lungCT_segmentation_DATA(lung_CT, 'train')
    val = lungCT_segmentation_DATA(lung_CT, 'val')

    train_loader = DataLoader(train, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size = batch_size, shuffle=True)

    net_G = UGenerator(in_c, out_c).to(device)
    net_D = Discriminator(in_c).to(device)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr = lr)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr = lr)
    net_G.train()
    net_D.train()
    loss_discriminator_list = []
    loss_generator_list = []
    loss_discriminator_val_list = []
    loss_generator_val_list = []
    iou_train_list = []
    iou_val_list = []

    for epoch in range(1, epochs + 1):
        loss_discriminator = 0
        loss_generator = 0
        iou_train = 0
        loss_discriminator_val = 0
        loss_generator_val = 0
        iou_val = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            imgs, masks = batch[0].to(device), batch[1].to(device)
            # Discriminator
            labels_real = (torch.ones(batch_size) * 0.9).reshape(batch_size,).to(device)
            net_D.zero_grad()
            output_real = net_D(masks).reshape(batch_size,)
            loss_real = criterion(output_real, labels_real)

            gen_out = net_G(imgs)
            labels_fake = (torch.ones(batch_size) * 0.1).reshape(batch_size,).to(device)
            output_fake = net_D(gen_out.detach()).reshape(batch_size,)
            loss_fake = criterion(output_fake, labels_fake)
            loss_D = loss_fake + loss_real
            loss_D.backward()
            optimizer_D.step()
            loss_discriminator += loss_D.item()

            # Generator
            net_G.zero_grad()
            iou_gen = iou_compute(masks, gen_out)
            iou_train += iou_gen
            labels_gen = torch.ones(batch_size).reshape(batch_size,).to(device)
            output_fake_gen = net_D(gen_out).reshape(batch_size,)
            loss_G = criterion(output_fake_gen, labels_gen)
            loss_G.backward()
            optimizer_G.step()
            loss_generator += loss_G.item()

        for batch in tqdm(val_loader):
            imgs, masks = batch[0].to(device), batch[1].to(device)
            # Discriminator
            labels_real = torch.ones(batch_size).reshape(batch_size,).to(device)
            output_real = net_D(masks).reshape(batch_size,)
            loss_real = criterion(output_real, labels_real)

            gen_out = net_G(imgs)
            labels_fake = torch.zeros(batch_size).reshape(batch_size,).to(device)
            output_fake = net_D(gen_out.detach()).reshape(batch_size,)
            loss_fake = criterion(output_fake, labels_fake)
            loss_D = loss_fake + loss_real
            loss_discriminator_val += loss_D.item()
            # Generator
            iou_gen = iou_compute(gen_out, masks)
            iou_val += iou_gen
            labels_gen = torch.ones(batch_size).reshape(batch_size,).to(device)
            output_fake_gen = net_D(gen_out).reshape(batch_size,)
            loss_G = criterion(output_fake_gen, labels_gen)
            loss_generator_val += loss_G.item()

        loss_generator /= len(train_loader)
        loss_discriminator /= len(train_loader)
        loss_generator_val /= len(val_loader)
        loss_discriminator_val /= len(val_loader)
        iou_train /= (batch_size * len(train_loader))
        iou_val /= (batch_size * len(val_loader))

        loss_discriminator_list.append(loss_discriminator)
        loss_generator_list.append(loss_generator)
        loss_discriminator_val_list.append(loss_discriminator_val)
        loss_generator_val_list.append(loss_generator_val)
        iou_train_list.append(iou_train)
        iou_val_list.append(iou_val)

        print("Epoch [%s]/[%s]\tTrain Generator Loss: %s\tTrain Discriminator Loss: %s\tTrain IOU: %s\tVal Generator Loss: %s\tVal Discriminator Loss: %s\tVal IOU: %s"
            % (epoch, epochs, loss_generator, loss_discriminator, iou_train, loss_generator_val, loss_discriminator_val, iou_val))

    if to_save:
        date = time.strftime("%D")
        save_dir = 'UGAN_%s_' % (iou_val_list[-1]) + date.replace("/", "-")
        if not os.path.isdir('./saved_models'):
            os.mkdir('./saved_models')
        save_path = os.path.join('./saved_models', save_dir)
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

def test(model_path, model_object, lung_CT_path, criterion = nn.BCELoss(), device = torch.device("cuda:0")):
    '''
    Testing function for UNET/UGAN model
    :param model_path: path to saved model being tested (.pth)
    :param model_object: instantiated UNET/UGAN class
    :param lung_CT_path: Directory to numpy array containing paired lung CT scans and masks; shape: (n, 2, 512, 512)
    :param criterion: loss function being used
    :param device: cpu/gpu (cuda) devices
    :return: Void
    '''
    batch_size = 1
    lung_CT = np.load(lung_CT_path)
    test = lungCT_segmentation_DATA(lung_CT, 'test')
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    model_object.load_state_dict(torch.load(model_path))
    pred_images = []
    loss_total = 0
    iou_total = 0
    model_object.eval()
    for batch in tqdm(test_loader):
        img_batch, mask_batch = batch[0].to(device), batch[1].to(device)
        output = model_object(img_batch)
        pred_images.append([img_batch.cpu().detach().numpy().reshape(512, 512), output.cpu().detach().numpy().reshape(512, 512)])
        loss = criterion(output, mask_batch)
        loss_total += loss.item()
        iou = iou_compute(mask_batch, output)
        iou_total += iou

    loss_total /= len(test_loader)
    iou_total /= (batch_size * len(test_loader))

    print("Test Loss: %s\tTest IOU: %s" % (loss_total, iou_total))
    np.save('./DiceLoss_test_preds', np.array(pred_images))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train or Test UNET/UGAN model for lung CT Segmentation')
    parser.add_argument('--mode', type = str, default = 'train', help= 'either train or test UNET/UGAN model')
    parser.add_argument('--model', type=str, default='unet', help='either unet or ugan')
    args = parser.parse_args()

    if args.mode == 'train':
        if args.model == 'unet':
            unet_train(args.lungdir, in_c=config.in_c, out_c=config.out_c, epochs=config.epochs, batch_size=config.batch_size, criterion=config.criterion, lr=config.lr, device=config.device, to_save=config.save)
        elif args.model == 'ugan':
            UGAN_train(args.lungdir, in_c=config.in_c, out_c=config.out_c, epochs=config.epochs, batch_size=config.batch_size, criterion=config.criterion, lr=config.lr, device=config.device, to_save=config.save)
    elif args.mode == 'test':
            test(config.saved_model_path, config.model_object, args.lungdir, config.criterion, device=config.device)











