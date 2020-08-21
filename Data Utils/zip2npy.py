import tarfile
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def load_tarfile(data_path):
    '''
    Unzipping zipped lung volumes
    :param data_path: folder containing zipped (bz2) lung CT volumes
    :return: Void
    '''
    for file in os.listdir(data_path):
        tar = tarfile.open(os.path.join(data_path, file), "r:bz2")
        tar.extractall()
        tar.close()

def load_img_npy(data_path):
    '''
    Converting mhd files (lung CT) to numpy arrays
    :param data_path: folder containing unzipped mhd/raw files
    :return: Void (saves numpy lung volumes locally)
    '''
    for file in os.listdir(data_path):
        if file.endswith(".mhd"):
            itk_image = sitk.ReadImage(os.path.join(data_path, file))
            img_array = sitk.GetArrayFromImage(itk_image)
            np.save(file[:-4] + ".npy", img_array)

def load_mask_npy(data_path):
    '''
    Converting mhd files (lung masks) to numpy arrays
    :param data_path: folder containing unzipped mask files (mhd) files
    :return: Void (saves numpy masks locally)
    '''
    for file in os.listdir(data_path):
        if file.endswith(".mhd"):
            itk_image = sitk.ReadImage(os.path.join(data_path, file))
            img_array = sitk.GetArrayFromImage(itk_image)
            np.save(file[:-4] + "_MASK.npy", img_array)

def view_img_masks(scan_num, dir_num, path_lung, path_masks):
    '''
    Tool for viewing CT images and masks
    :param scan_num: slice number in CT volume
    :param dir_num: folder number in lung CT volume volder
    :param path_lung: path to lung volumes
    :param path_masks: path to lung masks
    :return: Void (shows images and masks using matplotlib)
    '''
    dir_lung = os.listdir(path_lung)
    dir_mask = os.listdir(path_masks)
    image_arr = np.load(os.path.join(path_lung, dir_lung[dir_num]))
    mask_arr = np.load(os.path.join(path_masks, dir_mask[dir_num]))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_arr[scan_num, :, :], cmap = 'gray')
    ax[1].imshow(mask_arr[scan_num, :, :], cmap = 'gray')


def create_numpy(path_lung, path_masks, dir_lung, dir_mask, saved=True):
    '''
    Creating numpy arrays for lung CT scans and masks
    :param path_lung: path to folder containing lung volumes
    :param path_masks: path to folder containing masks
    :param dir_lung: path to save lung numpy array
    :param dir_mask: path to save lung numpy masks
    :param saved: boolean to save or not
    :return: Void (saves numpy arrays to specified path)
    '''
    path_list = []
    for i in range(0, len(os.listdir(path_lung))):
        path_list.append(
            [os.path.join(path_lung, os.listdir(path_lung)[i]), os.path.join(path_masks, os.listdir(path_masks)[i])])

    volume_list = [[np.load(i[0]), np.load(i[1])] for i in path_list]
    paired_images = []

    for volumes in volume_list:
        scan_num = volumes[0].shape[0]
        for i in range(scan_num):
            paired_images.append([volumes[0][i, :, :], volumes[1][i, :, :]])

    lung_CT = np.array([i[0] for i in paired_images])
    lung_masks = np.array([i[1] for i in paired_images])

    if not saved:
        np.save(dir_lung, lung_CT)
        np.save(dir_mask, lung_masks)

