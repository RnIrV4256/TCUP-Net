import numpy as np
from scipy.ndimage import gaussian_filter
import random
import os
import SimpleITK as sitk
import sys, time
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import argparse
import os
from patchify import patchify, unpatchify


def get_patches(img, patches_size):
    # This will split the image into small images of shape [48,6,5]
    patches = patchify(img, patches_size, step=1)
    return patches


def get_distance(f, spacing):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f, sampling=spacing)),
                        dist_func(1 - f, sampling=spacing))

    return distance


def get_head(img_path):
    temp = sitk.ReadImage(img_path)
    spacing = temp.GetSpacing()
    direction = temp.GetDirection()
    origin = temp.GetOrigin()

    return spacing, direction, origin


def copy_head_and_right_xyz(data, spacing, direction, origin):
    TrainData_new = data.astype('float32')
    TrainData_new = TrainData_new.transpose(2, 1, 0)
    TrainData_new = sitk.GetImageFromArray(TrainData_new)
    TrainData_new.SetSpacing(spacing)
    TrainData_new.SetOrigin(origin)
    TrainData_new.SetDirection(direction)

    return TrainData_new


def normalization(np_array):
    mean = np.mean(np_array)
    std = np.std(np_array)
    normed_array = (np_array - mean) / (std + 1e-8)
    return normed_array, mean, std


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


"""
==========================================
The input must be nii.gz which contains 
import header information such as spacing.
Spacing will affect the generation of the
signed distance.
=========================================
"""


def generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a[0])
    target_a = []
    target_b = []
    Mean = []
    Std = []
    mod_num = len(image_a)
    temp_resize_nii_path = 'temp_image.nii.gz'
    temp_resize_nii_label_path = 'temp_label.nii.gz'
    label1 = sitk.ReadImage(label_b)
    flag = 0
    for i in range(mod_num):
        data2 = nib.load(image_a[i]).get_fdata()  # mixed cases
        data1 = sitk.ReadImage(image_b[i])  # carved cases

        mod_i_image_path = image_b[i]

        if nib.load(image_a[i]).get_fdata().shape != nib.load(image_b[i]).get_fdata().shape:
            new_data1 = resize_image_itk(data1, data2.shape, resamplemethod=sitk.sitkLinear)
            sitk.WriteImage(new_data1, temp_resize_nii_path)
            mod_i_image_path = temp_resize_nii_path
            label_b = temp_resize_nii_label_path
            flag = 1

        if flag == 1:
            new_label1 = resize_image_itk(label1, data2.shape, resamplemethod=sitk.sitkNearestNeighbor)
            sitk.WriteImage(new_label1, label_b)

        img = nib.load(image_a[i]).get_fdata()
        normed_array, mean, std = normalization(img)
        target_a.append(normed_array)
        Mean.append(mean)
        Std.append(std)
        img = nib.load(mod_i_image_path).get_fdata()
        normed_array, mean, std = normalization(img)
        target_b.append(normed_array)

    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()
    label = np.copy(label_b)

    target_a = np.array(target_a)
    target_b = np.array(target_b)
    Mean = np.array(Mean)
    Std = np.array(Std)

    dis_array = get_distance(label, spacing)  # creat signed distance
    c = np.random.beta(1, 1)  # [0,1]             #creat distance

    c = (c - 0.5) * 2  # [-1.1]

    if c > 0:
        lam = c * np.min(dis_array) / 2  # λl = -1/2|min(dis_array)|
    else:
        lam = c * np.min(dis_array)

    mask = (dis_array < lam).astype('float32')  # creat M

    new_target = target_a * (mask == 0) + target_b * mask
    new_label = label_a * (mask == 0) + label_b * mask
    for i in range(mod_num):
        new_target[i, :, :, :] = new_target[i, :, :, :] * Std[i] + Mean[i]

    if len(new_target.shape) < 4:
        new_target.reshape(1, new_target.shape[0], new_target.shape[1], new_target.shape[2])
    target = []
    for i in range(len(image_a)):
        target.append(copy_head_and_right_xyz(new_target[i], spacing, direction, origin))
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    mask = copy_head_and_right_xyz(mask, spacing, direction, origin)

    return target, new_label, mask, lam