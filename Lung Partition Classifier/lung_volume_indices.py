import os
import numpy as np

def lung_volume(vessel_mask_volume):
    pixels = 0
    for idx in range(vessel_mask_volume.shape[0]):
        mask = vessel_mask_volume[idx, :, :]
        lung_pixels = np.sum(mask == 1)
        pixels += lung_pixels
    return pixels

def lower_third_slice(vessel_mask_volume):
    vessel_1_vol = lung_volume(vessel_mask_volume)
    lower_third_slice = -1
    one_third_volume = vessel_1_vol // 3
    volume_counter = 0
    for idx in range(vessel_mask_volume.shape[0]):
        if volume_counter <= one_third_volume:
            mask = vessel_mask_volume[idx, :, :]
            lung_pixels = np.sum(mask == 1)
            volume_counter += lung_pixels
            lower_third_slice = idx
    return lower_third_slice

def upper_third_slice(vessel_mask_volume):
    vessel_volume = lung_volume(vessel_mask_volume)
    vessel_lower = lower_third_slice(vessel_mask_volume)
    upper_slice = -1
    one_third_volume = vessel_volume // 3
    volume_counter = 0
    for idx in range(vessel_lower, vessel_mask_volume.shape[0]):
        if volume_counter <= one_third_volume:
            mask = vessel_mask_volume[idx, :, :]
            lung_pixels = np.sum(mask == 1)
            volume_counter += lung_pixels
            upper_slice = idx
    return upper_slice

def get_slices(mask_dir):
    slices_list = []
    for mask_vol in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_vol)
        mask_arr = np.load(mask_path)
        lower_third = lower_third_slice(mask_arr)
        upper_third = upper_third_slice(mask_arr)
        slices_list.append([lower_third, upper_third])
    return slices_list
'''
slices_list = [[148, 212], [160, 242],[214, 318],[183, 262],[170, 260],[188, 244],
               [217, 301],[166, 264],[246, 339],[174, 255],[177, 251],[153, 239],
               [230, 308],[168, 235],[214, 293],[176, 258],[254, 303],[132, 222],[174, 249],[171, 244]]
'''
def make_lung_partitions(lung_dirs, mask_dirs, slices_list):
    lower_lung = []
    middle_lung = []
    upper_lung = []
    for idx in range(len(slices_list)):
        path_lung = os.path.join(lung_dirs, os.listdir(lung_dirs)[idx])
        path_mask = os.path.join(mask_dirs, os.listdir(mask_dirs)[idx])
        lung_npy = np.load(path_lung)
        mask_npy = np.load(path_mask)
        for i in range(slices_list[idx][0]):
            lower_lung.append([lung_npy[i, :, :], mask_npy[i, :, :]])
        for j in range(slices_list[idx][0], slices_list[idx][1]):
            middle_lung.append([lung_npy[j, :, :], mask_npy[j, :, :]])
        for k in range(slices_list[idx][1], lung_npy.shape[0]):
            upper_lung.append([lung_npy[k, :, :], mask_npy[k, :, :]])
    return lower_lung, middle_lung, upper_lung

