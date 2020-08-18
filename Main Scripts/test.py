import torch
import torch.nn as nn
from UNET import UNET
from data_loading import lungCT_segmentation_DATA
from torch.utils.data import DataLoader
from model_tools import iou_compute, DiceLoss
import numpy as np
from tqdm import tqdm

batch_size = 1
device = torch.device("cuda:0")
lung_CT = np.load('/home/rrsood003/DATA/middle_lung.npy')
test = lungCT_segmentation_DATA(lung_CT, 'test')
test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

model = UNET(1, 1).to(device)
weights_dir = '/home/rrsood003/Segmentation/saved_models/UNET_0.981_Dice_08-12-20/UNET_0.981_08-12-20.pth'
model.load_state_dict(torch.load(weights_dir))
criterion = DiceLoss()
pred_images = []
loss_total = 0
iou_total = 0
model.eval()
for batch in tqdm(test_loader):
    img_batch, mask_batch = batch[0].to(device), batch[1].to(device)
    output = model(img_batch)
    pred_images.append([img_batch.cpu().detach().numpy().reshape(512, 512), output.cpu().detach().numpy().reshape(512, 512)])
    loss = criterion(output, mask_batch)
    loss_total += loss.item()
    iou = iou_compute(mask_batch, output)
    iou_total += iou

loss_total /= len(test_loader)
iou_total /= (batch_size * len(test_loader))

print("Test Loss: %s\tTest IOU: %s" % (loss_total, iou_total))
np.save('./DiceLoss_test_preds', np.array(pred_images))
