import numpy as np
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    '''
    Soft Dice Loss Function
    How to use:
        criterion = DiceLoss()
        ...
        loss = criterion(target, output)
    '''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):     
                                                    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
                                                                                    
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
                                                                                                            
        return 1 - dice

def iou_numpy(target_npy, pred_npy):
    '''
    Helper function for iou_compute
    :param target_npy: ground truth numpy mask
    :param pred_npy: predicted numpy mask
    :return: intersection over union between the two masks
    '''
    # (n, n) dim, rounded
    target_rounded = np.round(target_npy)
    pred_rounded = np.round(pred_npy)
    # Object iou
    intersection_obj = np.sum(target_rounded * pred_rounded)
    union_obj = np.sum(target_rounded) + np.sum(pred_rounded) - intersection_obj
    iou_obj = intersection_obj / union_obj
    # Background obj
    target_back = 1 - target_rounded
    pred_back = 1 - pred_rounded
    intersection_back = np.sum(target_back * pred_back)
    union_back = np.sum(target_back) + np.sum(pred_back) - intersection_back
    iou_back = intersection_back / union_back
                                                            
    iou_total = (iou_obj + iou_back) / 2
                                                                    
    return iou_total

def iou_compute(target_batch, pred_batch):
    '''
    Computes intersection over union between target and predicted PyTorch batches
    :param target_batch: Ground truth PyTorch batch of masks
    :param pred_batch: Predicted PyTorch batch of masks
    :return: sum of the IOUs for each target/pedicted pair
    '''
    target = np.squeeze(target_batch.cpu().numpy(), 1)
    pred = np.squeeze(pred_batch.cpu().detach().numpy(), 1) # (batch size, n, n)
    ious = []
    for i in range(pred.shape[0]):
        target_mask = target[i, :, :]
        pred_mask = pred[i, :, :]
        iou = iou_numpy(target_mask, pred_mask)
        ious.append(iou)
    return sum(ious)
