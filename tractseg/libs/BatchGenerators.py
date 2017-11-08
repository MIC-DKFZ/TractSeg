#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from tractseg.libs.DatasetUtils import DatasetUtils

'''
Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
'''

class SlicesBatchGenerator(BatchGeneratorBase):
    '''
    Returns 2D slices in ordered way.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None
        self.global_idx = 0

    def generate_train_batch(self):
        if self.HP.SLICE_DIRECTION == "x":
            end = self._data[0].shape[0]
        elif self.HP.SLICE_DIRECTION == "y":
            end = self._data[0].shape[1]
        elif self.HP.SLICE_DIRECTION == "z":
            end = self._data[0].shape[2]

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
            x = np.array(self._data[0][idxs,:,:,:]).astype(np.float32)
            y = np.array(self._data[1][idxs,:,:,:]).astype(self.HP.LABELS_TYPE)
            x = x.transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif self.HP.SLICE_DIRECTION == "y":
            x = np.array(self._data[0][:,idxs,:,:]).astype(np.float32)
            y = np.array(self._data[1][:,idxs,:,:]).astype(self.HP.LABELS_TYPE)
            x = x.transpose(1, 3, 0, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(1, 3, 0, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif self.HP.SLICE_DIRECTION == "z":
            x = np.array(self._data[0][:,:,idxs,:]).astype(np.float32)
            y = np.array(self._data[1][:,:,idxs,:]).astype(self.HP.LABELS_TYPE)
            x = x.transpose(2, 3, 0, 1)  # depth-channel has to be before width and height for Unet (but after batches)
            y = y.transpose(2, 3, 0, 1)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        self.global_idx = new_global_idx
        return data_dict


class SlicesBatchGeneratorRandom(BatchGeneratorBase):
    '''
    Randomly sample 2D slices from list of 2D slices.

    About 1.6-2s per 54-batch 45 bundles 1.25mm.
    ( a bit faster than SlicesBatchGeneratorRandomNiftiImg but not much)
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):
        # np.random.seed(1337) #Global random seed not working here; have to seed again

        # If shape of data is (x, y, z, channels) -> we are iterating over x
        idxs = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE, False, None)
        x = np.array(self._data[0][idxs]).astype(np.float32)
        x = x.transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
        y = np.array(self._data[1][idxs]).astype(self.HP.LABELS_TYPE)
        y = y.transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class SlicesBatchGeneratorRandomNiftiImg(BatchGeneratorBase):
    '''
    Randomly sample 2D slices from a .nii.gz image.

    About 2.5s per 54-batch 75 bundles 1.25mm. ?
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        # data = nib.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".nii.gz")).get_data()
        if np.random.random() < 0.5:
            data = nib.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
        else:
            data = nib.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()
        seg = nib.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".nii.gz")).get_data()

        data = np.nan_to_num(data)    # Needed otherwise not working
        seg = np.nan_to_num(seg)

        data = DatasetUtils.scale_input_to_unet_shape(data, self.HP.DATASET, self.HP.RESOLUTION)    # (x, y, z, channels)
        if self.HP.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
            # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask to the lower resolution
            seg = DatasetUtils.scale_input_to_unet_shape(seg, "HCP", self.HP.RESOLUTION)
        else:
            seg = DatasetUtils.scale_input_to_unet_shape(seg, self.HP.DATASET, self.HP.RESOLUTION)  # (x, y, z, classes)

        slice_idxs = np.random.choice(data.shape[0], self.BATCH_SIZE, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            x = data[slice_idxs, :, :].astype(np.float32)      # (batch_size, y, z, channels)
            y = seg[slice_idxs, :, :].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif slice_direction == 1:
            x = data[:, slice_idxs, :].astype(np.float32)      # (x, batch_size, z, channels)
            y = seg[:, slice_idxs, :].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(1, 3, 0, 2)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            x = data[:, :, slice_idxs].astype(np.float32)      # (x, y, batch_size, channels)
            y = seg[:, :, slice_idxs].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(2, 3, 0, 1)
            y = np.array(y).transpose(2, 3, 0, 1)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class SlicesBatchGeneratorRandomNpyImg(BatchGeneratorBase):
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

        data = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".npy"), mmap_mode="r")
        seg = np.load(join(C.HOME, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".npy"), mmap_mode="r")

        slice_idxs = np.random.choice(data.shape[0], self.BATCH_SIZE, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            x = data[slice_idxs, :, :].astype(np.float32)      # (batch_size, y, z, channels)
            y = seg[slice_idxs, :, :].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(0, 3, 1, 2)  # depth-channel has to be before width and height for Unet (but after batches)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif slice_direction == 1:
            x = data[:, slice_idxs, :].astype(np.float32)      # (x, batch_size, z, channels)
            y = seg[:, slice_idxs, :].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(1, 3, 0, 2)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            x = data[:, :, slice_idxs].astype(np.float32)      # (x, y, batch_size, channels)
            y = seg[:, :, slice_idxs].astype(self.HP.LABELS_TYPE)
            x = np.array(x).transpose(2, 3, 0, 1)
            y = np.array(y).transpose(2, 3, 0, 1)

        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


