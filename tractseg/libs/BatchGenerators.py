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

import numpy as np
import random
# from batchgenerators.dataloading.data_loader import DataLoaderBase
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from time import sleep
import os

import nibabel as nib
from os.path import join
from tractseg.libs.Config import Config as C
from tractseg.libs.DatasetUtils import DatasetUtils
from tractseg.libs import exp_utils
from tractseg.libs import img_utils

'''
Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
'''

class SlicesBatchGenerator(SlimDataLoaderBase):
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
            self.global_idx = 0
            raise StopIteration

        new_global_idx = self.global_idx + self.batch_size

        # If we reach end, make last batch smaller, so it fits exactly into rest
        if new_global_idx >= end:
            new_global_idx = end  # not end-1, because this goes into range, and there automatically -1

        idxs = list(range(self.global_idx, new_global_idx))

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


class SlicesBatchGeneratorRandomNiftiImg(SlimDataLoaderBase):
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

        for i in range(20):
            try:
                if self.HP.FEATURES_FILENAME == "12g90g270g":
                    # if np.random.random() < 0.5:
                    #     data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                    # else:
                    #     data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()

                    rnd_choice = np.random.random()
                    if rnd_choice < 0.33:
                        data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                    elif rnd_choice < 0.66:
                        data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()
                    else:
                        data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "12g_125mm_peaks.nii.gz")).get_data()
                elif self.HP.FEATURES_FILENAME == "T1_Peaks270g":
                    peaks = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                    t1 = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "T1.nii.gz")).get_data()
                    data = np.concatenate((peaks, t1), axis=3)
                elif self.HP.FEATURES_FILENAME == "T1_Peaks12g90g270g":
                    rnd_choice = np.random.random()
                    if rnd_choice < 0.33:
                        peaks = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                    elif rnd_choice < 0.66:
                        peaks = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()
                    else:
                        peaks = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "12g_125mm_peaks.nii.gz")).get_data()
                    t1 = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "T1.nii.gz")).get_data()
                    data = np.concatenate((peaks, t1), axis=3)
                else:
                    data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".nii.gz")).get_data()

                seg = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".nii.gz")).get_data()
                break
            except IOError:
                exp_utils.print_and_save(self.HP, "\n\nWARNING: Could not load file. Trying again in 20s (Try number: " + str(i) + ").\n\n")
            exp_utils.print_and_save(self.HP, "Sleeping 20s")
            sleep(20)
        # exp_utils.print_and_save(self.HP, "Successfully loaded input.")

        data = np.nan_to_num(data)    # Needed otherwise not working
        seg = np.nan_to_num(seg)

        data = DatasetUtils.scale_input_to_unet_shape(data, self.HP.DATASET, self.HP.RESOLUTION)    # (x, y, z, channels)

        if self.HP.LABELS_FILENAME not in ["bundle_peaks_11_808080", "bundle_peaks_20_808080", "bundle_peaks_808080",
                                           "bundle_masks_20_808080", "bundle_masks_72_808080", "bundle_peaks_Part1_808080",
                                           "bundle_peaks_Part2_808080", "bundle_peaks_Part3_808080", "bundle_peaks_Part4_808080"]:
            if self.HP.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
                # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask to the lower resolution
                seg = DatasetUtils.scale_input_to_unet_shape(seg, "HCP", self.HP.RESOLUTION)
            else:
                seg = DatasetUtils.scale_input_to_unet_shape(seg, self.HP.DATASET, self.HP.RESOLUTION)  # (x, y, z, classes)

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)

        # Randomly sample slice orientation
        if self.HP.TRAINING_SLICE_DIRECTION == "xyz":
            slice_direction = int(round(random.uniform(0,2)))
        else:
            slice_direction = 1 #always use Y

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






############################################################################################################
# Backup
############################################################################################################

class SlicesBatchGeneratorRandomNiftiImg_5slices(SlimDataLoaderBase):
    '''
    Randomly sample 2D slices from a .nii.gz image.
    Always 2 slices above and bellow.

    About 2.5s per 54-batch 75 bundles 1.25mm. ?
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        for i in range(20):
            try:
                if np.random.random() < 0.5:
                    data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                else:
                    data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()

                # rnd_choice = np.random.random()
                # if rnd_choice < 0.33:
                #     data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                # elif rnd_choice < 0.66:
                #     data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()
                # else:
                #     data = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], "12g_125mm_peaks.nii.gz")).get_data()

                seg = nib.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".nii.gz")).get_data()
                break
            except IOError:
                exp_utils.print_and_save(self.HP, "\n\nWARNING: Could not load file. Trying again in 20s (Try number: " + str(i) + ").\n\n")
            exp_utils.print_and_save(self.HP, "Sleeping 20s")
            sleep(20)
        # exp_utils.print_and_save(self.HP, "Successfully loaded input.")

        data = np.nan_to_num(data)    # Needed otherwise not working
        seg = np.nan_to_num(seg)

        data = DatasetUtils.scale_input_to_unet_shape(data, self.HP.DATASET, self.HP.RESOLUTION)    # (x, y, z, channels)
        if self.HP.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
            # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask to the lower resolution
            seg = DatasetUtils.scale_input_to_unet_shape(seg, "HCP", self.HP.RESOLUTION)
        else:
            seg = DatasetUtils.scale_input_to_unet_shape(seg, self.HP.DATASET, self.HP.RESOLUTION)  # (x, y, z, classes)

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            y = seg[slice_idxs, :, :].astype(self.HP.LABELS_TYPE)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif slice_direction == 1:
            y = seg[:, slice_idxs, :].astype(self.HP.LABELS_TYPE)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            y = seg[:, :, slice_idxs].astype(self.HP.LABELS_TYPE)
            y = np.array(y).transpose(2, 3, 0, 1)


        sw = 5 #slice_window (only odd numbers allowed)
        pad = int((sw-1) / 2)

        data_pad = np.zeros((data.shape[0]+sw-1, data.shape[1]+sw-1, data.shape[2]+sw-1, data.shape[3])).astype(data.dtype)
        data_pad[pad:-pad, pad:-pad, pad:-pad, :] = data   #padded with two slices of zeros on all sides
        batch=[]
        for s_idx in slice_idxs:
            if slice_direction == 0:
                #(s_idx+2)-2:(s_idx+2)+3 = s_idx:s_idx+5
                x = data_pad[s_idx:s_idx+sw:, pad:-pad, pad:-pad, :].astype(np.float32)      # (5, y, z, channels)
                x = np.array(x).transpose(0, 3, 1, 2)  # channels dim has to be before width and height for Unet (but after batches)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)
            elif slice_direction == 1:
                x = data_pad[pad:-pad, s_idx:s_idx+sw, pad:-pad, :].astype(np.float32)  # (5, y, z, channels)
                x = np.array(x).transpose(1, 3, 0, 2)  # channels dim has to be before width and height for Unet (but after batches)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)
            elif slice_direction == 2:
                x = data_pad[pad:-pad, pad:-pad, s_idx:s_idx+sw, :].astype(np.float32)  # (5, y, z, channels)
                x = np.array(x).transpose(2, 3, 0, 1)  # channels dim has to be before width and height for Unet (but after batches)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)
        data_dict = {"data": np.array(batch),     # (batch_size, channels, x, y, [z])
                     "seg": y}                    # (batch_size, channels, x, y, [z])

        return data_dict


class SlicesBatchGeneratorRandomNpyImg(SlimDataLoaderBase):
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

        data = np.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.FEATURES_FILENAME + ".npy"), mmap_mode="r")
        seg = np.load(join(C.DATA_PATH, self.HP.DATASET_FOLDER, subjects[subject_idx], self.HP.LABELS_FILENAME + ".npy"), mmap_mode="r")

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)

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


class SlicesBatchGeneratorPrecomputedBatches(SlimDataLoaderBase):
    '''
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.HP = None

    def generate_train_batch(self):

        type = self._data[0]
        path = join(C.DATA_PATH, self.HP.DATASET_FOLDER, type)
        # do not use last batch, because might be corrupted if aborted batch precompution early
        nr_of_files = len([name for name in os.listdir(path) if os.path.isfile(join(path, name))]) - 1
        idx = int(random.uniform(0, int(nr_of_files / 2.)))

        data = nib.load(join(path, "batch_" + str(idx) + "_data.nii.gz")).get_data()
        seg = nib.load(join(path, "batch_" + str(idx) + "_seg.nii.gz")).get_data()

        return {"data": data, "seg": seg}
