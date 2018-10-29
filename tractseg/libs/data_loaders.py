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

"""
Code to load data and to create batches of 2D slices from 3D images.

Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
import random
from time import sleep
import numpy as np
import nibabel as nib

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import dataset_utils
from tractseg.libs import exp_utils


def load_training_data(Config, subject, use_gt_mask=True):
    """
    Load data and labels for one subject from the training set. Cut and scale to make them have
    correct size.

    :param Config: config class
    :param subject: subject id (string)
    :return:
    """
    for i in range(20):
        try:
            if Config.FEATURES_FILENAME == "12g90g270g":
                # if np.random.random() < 0.5:
                #     data = nib.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx], "270g_125mm_peaks.nii.gz")).get_data()
                # else:
                #     data = nib.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx], "90g_125mm_peaks.nii.gz")).get_data()

                rnd_choice = np.random.random()
                if rnd_choice < 0.33:
                    data = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks.nii.gz")).get_data()
                elif rnd_choice < 0.66:
                    data = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks.nii.gz")).get_data()
                else:
                    data = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks.nii.gz")).get_data()
            elif Config.FEATURES_FILENAME == "T1_Peaks270g":
                peaks = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks.nii.gz")).get_data()
                t1 = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1.nii.gz")).get_data()
                data = np.concatenate((peaks, t1), axis=3)
            elif Config.FEATURES_FILENAME == "T1_Peaks12g90g270g":
                rnd_choice = np.random.random()
                if rnd_choice < 0.33:
                    peaks = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks.nii.gz")).get_data()
                elif rnd_choice < 0.66:
                    peaks = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks.nii.gz")).get_data()
                else:
                    peaks = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks.nii.gz")).get_data()
                t1 = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1.nii.gz")).get_data()
                data = np.concatenate((peaks, t1), axis=3)
            else:
                data = nib.load(
                    join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.FEATURES_FILENAME + ".nii.gz")).get_data()

            break
        except IOError:
            exp_utils.print_and_save(Config, "\n\nWARNING: Could not load file. Trying again in 20s (Try number: " + str(i) + ").\n\n")
        exp_utils.print_and_save(Config, "Sleeping 20s")
        sleep(20)
    data = np.nan_to_num(data)  # Needed otherwise not working
    data = dataset_utils.scale_input_to_unet_shape(data, Config.DATASET, Config.RESOLUTION)  # (x, y, z, channels)

    if use_gt_mask:
        seg = nib.load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.LABELS_FILENAME + ".nii.gz")).get_data()
        seg = np.nan_to_num(seg)
        if Config.LABELS_FILENAME not in ["bundle_peaks_11_808080", "bundle_peaks_20_808080", "bundle_peaks_808080",
                                               "bundle_masks_20_808080", "bundle_masks_72_808080", "bundle_peaks_Part1_808080",
                                               "bundle_peaks_Part2_808080", "bundle_peaks_Part3_808080", "bundle_peaks_Part4_808080"]:
            if Config.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
                # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask to the lower resolution
                seg = dataset_utils.scale_input_to_unet_shape(seg, "HCP", Config.RESOLUTION)
            else:
                seg = dataset_utils.scale_input_to_unet_shape(seg, Config.DATASET, Config.RESOLUTION)  # (x, y, z, classes)
    else:
        # Use dummy mask in case we only want to predict on some data (where we do not have Ground Truth))
        seg = np.zeros((Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.NR_OF_CLASSES)).astype(Config.LABELS_TYPE)

    return data, seg


class DataLoader2D_Nifti(SlimDataLoaderBase):
    '''
    Randomly selects subjects and slices and creates batch of 2D slices.

    Takes image IDs provided via self._data, randomly selects one ID,
    loads the nifti image and randomly samples 2D slices from it.

    Timing:
    About 2.5s per 54-batch 75 bundles 1.25mm. ?
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        data, seg = load_training_data(self.Config, subjects[subject_idx])

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)
        x, y = dataset_utils.sample_slices(data, seg, slice_idxs,
                              training_slice_direction=self.Config.TRAINING_SLICE_DIRECTION,
                              labels_type=self.Config.LABELS_TYPE)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict



############################################################################################################
# Backup
############################################################################################################

class DataLoader2D_Nifti_5slices(SlimDataLoaderBase):
    '''
    Randomly selects subjects and slices and creates batch of 2D slices (+2 slices above and below).

    Takes image ID provided via self._data, loads the nifti image and randomly samples 2D slices
    from it. Always adds 2 slices above and below.

    Timing:
    About 2.5s per 54-batch 75 bundles 1.25mm. ?
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        data, seg = load_training_data(self.Config, subjects[subject_idx])

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)

        # Randomly sample slice orientation
        slice_direction = int(round(random.uniform(0,2)))

        if slice_direction == 0:
            y = seg[slice_idxs, :, :].astype(self.Config.LABELS_TYPE)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif slice_direction == 1:
            y = seg[:, slice_idxs, :].astype(self.Config.LABELS_TYPE)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            y = seg[:, :, slice_idxs].astype(self.Config.LABELS_TYPE)
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


class DataLoader2D_Npy(SlimDataLoaderBase):
    '''
    Takes image ID provided via self._data, loads the Npy (numpy array) image and randomly samples 2D slices from it.

    Timing:
    About 4s per 54-batch 75 bundles 1.25mm.
    About 2s per 54-batch 45 bundles 1.25mm.
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def generate_train_batch(self):

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        if self.Config.TYPE == "combined":
            if np.random.random() < 0.5:
                data = np.load(join(C.DATA_PATH, "HCP_fusion_npy_270g_125mm", subjects[subject_idx], "270g_125mm_xyz.npy"), mmap_mode="r")
            else:
                data = np.load(join(C.DATA_PATH, "HCP_fusion_npy_32g_25mm", subjects[subject_idx], "32g_25mm_xyz.npy"), mmap_mode="r")
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx], self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")
        else:
            data = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx], self.Config.FEATURES_FILENAME + ".npy"), mmap_mode="r")
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx], self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")

        data = np.nan_to_num(data)
        seg = np.nan_to_num(seg)

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)
        x, y = dataset_utils.sample_slices(data, seg, slice_idxs,
                                           training_slice_direction=self.Config.TRAINING_SLICE_DIRECTION,
                                           labels_type=self.Config.LABELS_TYPE)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoader2D_PrecomputedBatches(SlimDataLoaderBase):
    '''
    Loads precomputed batches
    '''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def generate_train_batch(self):

        type = self._data[0]
        path = join(C.DATA_PATH, self.Config.DATASET_FOLDER, type)
        # do not use last batch, because might be corrupted if aborted batch precompution early
        nr_of_files = len([name for name in os.listdir(path) if os.path.isfile(join(path, name))]) - 1
        idx = int(random.uniform(0, int(nr_of_files / 2.)))

        data = nib.load(join(path, "batch_" + str(idx) + "_data.nii.gz")).get_data()
        seg = nib.load(join(path, "batch_" + str(idx) + "_seg.nii.gz")).get_data()

        return {"data": data, "seg": seg}
