#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, inspect

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import numpy as np
import random
from libs.Utils import Utils
from DeepLearningBatchGeneratorUtils.DataGeneratorBase import BatchGeneratorBase
import nibabel as nib
from os.path import join
from libs.Config import Config as C
from libs.ImgUtils import ImgUtils
from libs.DatasetUtils import DatasetUtils

'''
Info:
Dimensions order for DeepLearningBatchGenerator:

(batch_size, channels, x, y, [z])
'''

class SlicesBatchGeneratorNpyImg_fusion(BatchGeneratorBase):
    '''
    Returns 2D slices ordered way. Takes data in form of a npy file for each image. Npy file is already cropped to right size.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None
        self.global_idx = 0

    def generate_train_batch(self):

        subject = self._data[0]
        data = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subject, self.HP.FEATURES_FILENAME + ".npy"), mmap_mode="r")
        seg = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subject, self.HP.LABELS_FILENAME + ".npy"), mmap_mode="r")

        if self.HP.SLICE_DIRECTION == "x":
            end = data.shape[0]
        elif self.HP.SLICE_DIRECTION == "y":
            end = data.shape[1]
        elif self.HP.SLICE_DIRECTION == "z":
            end = data.shape[2]

        # Stop iterating if we reached end of data
        if self.global_idx >= end:
            # print("Stopped because end of file")
            self.global_idx = 0
            raise StopIteration

        new_global_idx = self.global_idx + self.BATCH_SIZE

        # If we reach end, make last batch smaller, so it fits exactly into rest
        if new_global_idx >= end:
            new_global_idx = end  # not end-1, because this goes into range, and there automatically -1

        idxs = range(self.global_idx, new_global_idx)

        if self.HP.SLICE_DIRECTION == "x":
            x = data[idxs,:,:].astype(np.float32)
            y = seg[idxs,:,:].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = x.transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif self.HP.SLICE_DIRECTION == "y":
            x = data[:,idxs,:].astype(np.float32)
            y = seg[:,idxs,:].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = x.transpose(1, 3, 0, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(1, 3, 0, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif self.HP.SLICE_DIRECTION == "z":
            x = data[:,:,idxs].astype(np.float32)
            y = seg[:,:,idxs].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = x.transpose(2, 3, 0, 1)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(2, 3, 0, 1)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)

        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        #If we want only CA Binary
        #Bundles together Order
        # x = x[:, (0, 75, 150, 5, 80, 155), :, :]
        # y = y[:, (0, 5), :, :]
        #Mixed Order
        # x = x[:, (0, 5, 75, 80, 150, 155), :, :]
        # y = y[:, (0, 5), :, :]

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        self.global_idx = new_global_idx
        return data_dict


class SlicesBatchGeneratorRandomNpyImg_fusion(BatchGeneratorBase):
    '''
    Randomly sample 2D slices from a npy file for each subject.

    About 4s per 54-batch 75 bundles 1.25mm.
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        # data = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".npy"), mmap_mode="r")
        if np.random.random() < 0.5:
            data = np.load(join(C.HOME, "HCP_fusion_npy_270g_125mm", subjects[subject_idx], "270g_125mm_xyz.npy"), mmap_mode="r")
        else:
            data = np.load(join(C.HOME, "HCP_fusion_npy_32g_25mm", subjects[subject_idx], "32g_25mm_xyz.npy"), mmap_mode="r")

        seg = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".npy"), mmap_mode="r")

        # print("data 1: {}".format(data.shape))
        # print("seg 1: {}".format(seg.shape))

        slice_idxs = np.random.choice(data.shape[0], self.BATCH_SIZE, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            x = data[slice_idxs, :, :].astype(np.float32)      # (batch_size, y, z, channels, xyz)
            y = seg[slice_idxs, :, :].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = np.array(x).transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)

        elif slice_direction == 1:
            x = data[:, slice_idxs, :].astype(np.float32)      # (x, batch_size, z, channels, xyz)
            y = seg[:, slice_idxs, :].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = np.array(x).transpose(1, 3, 0, 2)
            y = np.array(y).transpose(1, 3, 0, 2)

        elif slice_direction == 2:
            x = data[:, :, slice_idxs].astype(np.float32)      # (x, y, batch_size, channels, xyz)
            y = seg[:, :, slice_idxs].astype(self.HP.LABELS_TYPE)

            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))

            x = np.array(x).transpose(2, 3, 0, 1)
            y = np.array(y).transpose(2, 3, 0, 1)


        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        # If we want only CA Binary
        #Bundles together Order
        # x = x[:, (0, 75, 150, 5, 80, 155), :, :]
        # y = y[:, (0, 5), :, :]
        #Mixed Order
        # x = x[:, (0, 5, 75, 80, 150, 155), :, :]
        # y = y[:, (0, 5), :, :]

        # print("data 2: {}".format(x.shape))
        # print("seg 2: {}".format(y.shape))

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class SlicesBatchGeneratorRandomNpyImg_fusionMean(BatchGeneratorBase):
    '''
    take mean of xyz channel and return slices (x,y,nrBundles)
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        data = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".npy"), mmap_mode="r")
        seg = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".npy"), mmap_mode="r")

        # print("data 1: {}".format(data.shape))
        # print("seg 1: {}".format(seg.shape))

        slice_idxs = np.random.choice(data.shape[0], self.BATCH_SIZE, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            x = data[slice_idxs, :, :].astype(np.float32)      # (batch_size, y, z, channels, xyz)
            y = seg[slice_idxs, :, :].astype(self.HP.LABELS_TYPE)

            x = x.mean(axis=4)

            x = np.array(x).transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)

        elif slice_direction == 1:
            x = data[:, slice_idxs, :].astype(np.float32)      # (x, batch_size, z, channels, xyz)
            y = seg[:, slice_idxs, :].astype(self.HP.LABELS_TYPE)

            x = x.mean(axis=4)

            x = np.array(x).transpose(1, 3, 0, 2)
            y = np.array(y).transpose(1, 3, 0, 2)

        elif slice_direction == 2:
            x = data[:, :, slice_idxs].astype(np.float32)      # (x, y, batch_size, channels, xyz)
            y = seg[:, :, slice_idxs].astype(self.HP.LABELS_TYPE)

            x = x.mean(axis=4)

            x = np.array(x).transpose(2, 3, 0, 1)
            y = np.array(y).transpose(2, 3, 0, 1)


        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        # print("data 2: {}".format(x.shape))
        # print("seg 2: {}".format(y.shape))

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict

