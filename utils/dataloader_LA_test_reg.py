from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np
import h5py

# from utils.augmentation_cpu import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, num_classes, shot=10):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(labeled_file_dir)]
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes
        self.labeled_filenames = self.labeled_filenames[shot:]
        # print(len(self.labeled_filenames), self.labeled_filenames)

    def __getitem__(self, index):
        labed_name1 = self.labeled_filenames[index//len(self.labeled_filenames)]
        labed_h5f1 = h5py.File(self.labeled_file_dir + "/" + labed_name1 + "/mri_norm2.h5", 'r')
        labed_img1 = labed_h5f1['image'][:]
        labed_lab1 = labed_h5f1['label'][:]
        # labed_img1 = self.pad(labed_img1)
        labed_img1, labed_lab1 = self.RandomCrop(labed_img1, labed_lab1)
        labed_img1 = labed_img1.astype(np.float32)
        labed_img1 = labed_img1[np.newaxis, :, :, :]
        # labed_lab1 = self.pad(labed_lab1)
        labed_lab1 = self.to_categorical(labed_lab1, self.num_classes)
        labed_lab1 = labed_lab1.astype(np.float32)

        labed_name2 = self.labeled_filenames[index % len(self.labeled_filenames)]
        labed_h5f2 = h5py.File(self.labeled_file_dir + "/" + labed_name2 + "/mri_norm2.h5", 'r')
        labed_img2 = labed_h5f2['image'][:]
        labed_lab1 = labed_h5f1['label'][:]
        # labed_img2 = self.pad(labed_img2)
        labed_img2, labed_lab2 = self.RandomCrop(labed_img2, labed_lab2)
        labed_img2 = labed_img2.astype(np.float32)
        labed_img2 = labed_img2[np.newaxis, :, :, :]
        labed_lab2 = labed_h5f2['label'][:]
        # labed_lab2 = self.pad(labed_lab2)
        labed_lab2 = self.to_categorical(labed_lab2, self.num_classes)
        labed_lab2 = labed_lab2.astype(np.float32)

        return labed_img1, labed_lab1, labed_img2, labed_lab2, labed_name1, labed_name2

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.labeled_filenames) * len(self.labeled_filenames)

    def pad(self, image):
        patch_size = (112, 112, 80)
        w, h, d = image.shape

        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)

        return image

    def RandomCrop(self, image, label):
        output_size = (112, 112, 80)
        # pad the sample if necessary
        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])

        label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        return image, label