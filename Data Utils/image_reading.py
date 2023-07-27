import tarfile
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch

path_lungraw = 'D:/DATA/VESSEL_DATA/temp/LungCT'
path_maskraw = 'D:/DATA/VESSEL_DATA/temp/LungMasks'

def load_tarfile(data_path):
    for file in os.listdir(data_path):
        tar = tarfile.open(os.path.join(data_path, file), "r:bz2")
        tar.extractall()
        tar.close()

def load_img_npy(data_path):
    for file in os.listdir(data_path):
        if file.endswith(".mhd"):
            itk_image = sitk.ReadImage(os.path.join(data_path, file))
            img_array = sitk.GetArrayFromImage(itk_image)
            np.save(file[:-4] + ".npy", img_array)

def load_mask_npy(data_path):
    for file in os.listdir(data_path):
        if file.endswith(".mhd"):
            itk_image = sitk.ReadImage(os.path.join(data_path, file))
            img_array = sitk.GetArrayFromImage(itk_image)
            np.save(file[:-4] + "_MASK.npy", img_array)

def view_img_masks(scan_num, dir_num, path_lung, path_masks):
    dir_lung = os.listdir(path_lung)
    dir_mask = os.listdir(path_masks)
    image_arr = np.load(os.path.join(path_lung, dir_lung[dir_num]))
    mask_arr = np.load(os.path.join(path_masks, dir_mask[dir_num]))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_arr[scan_num, :, :], cmap = 'gray')
    ax[1].imshow(mask_arr[scan_num, :, :], cmap = 'gray')

def create_numpy(path_lung, path_masks, saved = True):
    path_list = []
    for i in range(0, len(os.listdir(path_lung))):
        path_list.append([os.path.join(path_lung, os.listdir(path_lung)[i]), os.path.join(path_masks, os.listdir(path_masks)[i])])
    
    volume_list = [[np.load(i[0]), np.load(i[1])] for i in path_list]
    paired_images = []
    
    for volumes in volume_list:
        scan_num = volumes[0].shape[0]
        for i in range(scan_num):
            paired_images.append([volumes[0][i, :, :], volumes[1][i, :, :]])
    
    lung_CT = np.array([i[0] for i in paired_images])
    lung_masks = np.array([i[1] for i in paired_images])
    
    return lung_CT, lung_masks
    
    if not saved:
        np.save('D:\DATA\VESSEL_DATA\lung_CT.npy', lung_CT)
        np.save('D:\DATA\VESSEL_DATA\lung_masks.npy', lung_masks)


def create_numpy(path_lung, path_masks, saved = False):
    path_list = []
    for i in range(0, len(os.listdir(path_lung))):
        path_list.append([os.path.join(path_lung, os.listdir(path_lung)[i]), os.path.join(path_masks, os.listdir(path_masks)[i])])
    
    volume_list = [[np.load(i[0]), np.load(i[1])] for i in path_list]
    paired_images = []
    
    for volumes in volume_list:
        scan_num = volumes[0].shape[0]
        for i in range(scan_num):
            paired_images.append([volumes[0][i, :, :], volumes[1][i, :, :]])
    
    lung_CT = np.array([i[0] for i in paired_images])
    lung_masks = np.array([i[1] for i in paired_images])
    
    return lung_CT, lung_masks
    
    if not saved:
        np.save('/home/rrsood003/DATA/lung_CT.npy', lung_CT)
        np.save('/home/rrsood003/DATA/lung_masks.npy', lung_masks)
