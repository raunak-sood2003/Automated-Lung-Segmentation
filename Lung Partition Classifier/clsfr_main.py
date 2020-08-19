import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clsfr_data_loading import lungCT_clsfr_DATA
import numpy as np
from tqdm as tqdm
import time
import os

pth = '/home/rrsood003/DATA/classifier_pairs.npy'
classifier_pairs = np.load(pth, allow_pickle = True)

batch_size = 16
train = lungCT_clsfr_DATA(classifier_pairs, 'train')
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
val = lungCT_clsfr_DATA(classifier_pairs, 'val')
val_loader = DataLoader(val, batch_size = batch_size, shuffle = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_save = True
epochs = 10
lr = 0.01
model = lungPartitionClsfr(1, 3).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    val_loss = 0
    train_total = 0
    train_correct = 0
    val_total = 0
    val_correct = 0
    for batch in tqdm(train_loader):
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output_train = model(X_batch)
        loss_train = criterion(output_train, y_batch)
        loss_train.backward()
        optimizer.step()
        train_loss += loss_train.item()
        
        for idx, y in enumerate(y_batch):
            if torch.argmax(y) == torch.argmax(output_train[idx]):
                train_correct += 1
            train_total += 1
    
    with torch.no_grad():
        model.eval()        
        for batch_val in tqdm(val_loader):
            X_val, y_val = batch_val[0].to(device), batch_val[1].to(device)
            output_val = model(X_val)
            loss_val = criterion(output_val, y_val)
            val_loss += loss_val.item()

            for idx, y in enumerate(y_val):
                if torch.argmax(y) == torch.argmax(output_val[idx]):
                    val_correct += 1
                val_total += 1
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print("Epoch [%s]/[%s]\tTrain Loss: %s\tVal Loss: %s\tTrain Acc: %s\tVal Acc: %s" 
          % (epoch, epochs, train_loss, val_loss, train_acc, val_acc))

if to_save:
    date = time.strftime("%D").replace('/', '-')
    clsfr_file = 'lungCT_clsfr_epoch%s_batch%s_%s' % (epochs, batch_size, date)
    save_folder = '/home/rrsood003/Classifier/saved_models/' + clsfr_file
    os.mkdir(save_folder)
    torch.save(model, save_folder + clsfr_file + '.pth')
    np.save(save_folder + 'train_losses.npy', np.array(train_losses))
    np.save(save_folder + 'val_losses.npy', np.array(val_losses))
    np.save(save_folder + 'train_accuracies.npy', np.array(train_accuracies))
    np.save(save_folder + 'val_accuracies.npy', np.array(val_accuracies))
    