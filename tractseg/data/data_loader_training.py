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
import warnings
import multiprocessing
from time import sleep
import numpy as np
import nibabel as nib

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, FlipVectorAxisTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_2D_image_batched
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from tractseg.data.DLDABG_standalone import ResampleTransformLegacy

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import dataset_utils
from tractseg.libs import exp_utils
from tractseg.libs import img_utils

# warnings.simplefilter("ignore", UserWarning)  # hide batchgenerator warnings

def load_training_data(Config, subject):
    """
    Load data and labels for one subject from the training set. Cut and scale to make them have
    correct size.

    :param Config: config class
    :param subject: subject id (string)
    :return:
    """

    def load(filepath):
        data = nib.load(filepath + ".nii.gz").get_data()
        # data = np.load(filepath + ".npy", mmap_mode="r")
        return data

    if Config.FEATURES_FILENAME == "12g90g270g":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))

    elif Config.FEATURES_FILENAME == "12g90g270g_BX":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_bedpostx_peaks_scaled"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_bedpostx_peaks_scaled"))

    elif Config.FEATURES_FILENAME == "12g90g270g_CSD_BX":
        rnd_choice_1 = np.random.random()
        rnd_choice_2 = np.random.random()
        if rnd_choice_1 < 0.5:  # CSD
            if rnd_choice_2 < 0.33:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
            elif rnd_choice_2 < 0.66:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))
        else:  # BX
            if rnd_choice_2 < 0.33:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
            elif rnd_choice_2 < 0.66:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_bedpostx_peaks_scaled"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_bedpostx_peaks_scaled"))
            # Flip x axis to make BedpostX compatible with mrtrix CSD
            data[:, :, :, 0] *= -1
            data[:, :, :, 3] *= -1
            data[:, :, :, 6] *= -1

    elif Config.FEATURES_FILENAME == "32g270g_BX":
        rnd_choice = np.random.random()
        path_32g = join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "32g_125mm_bedpostx_peaks_scaled")
        if rnd_choice < 0.5:  # and os.path.exists(path_32g + ".nii.gz"):
            data = load(path_32g)
            rnd_choice_2 = np.random.random()
            if rnd_choice_2 < 0.5:
                data[:, :, :, 6:9] = 0  # set third peak to 0
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
    elif Config.FEATURES_FILENAME == "T1_Peaks270g":
        peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        t1 = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1"))
        data = np.concatenate((peaks, t1), axis=3)
    elif Config.FEATURES_FILENAME == "T1_Peaks12g90g270g":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        elif rnd_choice < 0.66:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
        else:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))
        t1 = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1"))
        data = np.concatenate((peaks, t1), axis=3)
    else:
        data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.FEATURES_FILENAME))


    seg = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.LABELS_FILENAME))

    # Not needed anymore when using preprocessed input data
    # data = np.nan_to_num(data)
    # seg = np.nan_to_num(seg)

    # This no needed anymore because padding/cropping is done automatically. Only downsampling is not done automatically
    # if Config.LABELS_FILENAME not in ["bundle_peaks_11_808080", "bundle_peaks_20_808080", "bundle_peaks_808080",
    #                                        "bundle_masks_20_808080", "bundle_masks_72_808080", "bundle_peaks_Part1_808080",
    #                                        "bundle_peaks_Part2_808080", "bundle_peaks_Part3_808080", "bundle_peaks_Part4_808080"]:
    #     if Config.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
    #         # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask to the lower resolution
    #         seg = dataset_utils.scale_input_to_unet_shape(seg, "HCP", Config.RESOLUTION)
    #     else:
    #         seg = dataset_utils.scale_input_to_unet_shape(seg, Config.DATASET, Config.RESOLUTION)  # (x, y, z, classes)

    return data, seg


class BatchGenerator2D_Nifti_random(SlimDataLoaderBase):
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

        # np.random.seed(1234)

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor

        data, seg = load_training_data(self.Config, subjects[subject_idx])

        #Convert peaks to tensors if tensor model
        if self.Config.NR_OF_GRADIENTS == 18:
            data = img_utils.peak_image_to_tensor_image(data)

        slice_direction = dataset_utils.slice_dir_to_int(self.Config.TRAINING_SLICE_DIRECTION)
        slice_idxs = np.random.choice(data.shape[slice_direction], self.batch_size, False, None)
        x, y = dataset_utils.sample_slices(data, seg, slice_idxs,
                            slice_direction=slice_direction,
                            labels_type=self.Config.LABELS_TYPE)


        # Can be replaced by crop
        # x = pad_nd_image(x, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # y = pad_nd_image(y, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # x = center_crop_2D_image_batched(x, self.Config.INPUT_DIM)
        # y = center_crop_2D_image_batched(y, self.Config.INPUT_DIM)

        #Crop and pad to input size
        x, y = crop(x, y, crop_size=self.Config.INPUT_DIM)  # does not work with img with batches and channels

        # Works -> results as good? -> todo: make the same way for inference!
        # This is needed for Schizo dataset
        # x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})
        # y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})

        # Does not make it slower
        x = x.astype(np.float32)
        y = y.astype(np.float32)  # if not doing this: during validation: ConnectionResetError: [Errno 104] Connection
                                  # reset by peer

        #possible optimization: sample slices from different patients and pad all to same size (size of biggest)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class BatchGenerator2D_Npy_random(SlimDataLoaderBase):
    '''
    Takes image ID provided via self._data, loads the Npy (numpy array) image and randomly samples 2D slices from it.

    Needed for fusion training.

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
        slice_direction = dataset_utils.slice_dir_to_int(self.Config.TRAINING_SLICE_DIRECTION)
        x, y = dataset_utils.sample_slices(data, seg, slice_idxs,
                                           slice_direction=slice_direction,
                                           labels_type=self.Config.LABELS_TYPE)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderTraining:

    def __init__(self, Config):
        self.Config = Config

    def _augment_data(self, batch_generator, type=None):

        if self.Config.DATA_AUGMENTATION:
            num_processes = 15  # 15 is a bit faster than 8 on cluster
            # num_processes = multiprocessing.cpu_count()  # on cluster: gives all cores, not only assigned cores
        else:
            num_processes = 6

        tfs = []  #transforms

        if self.Config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        if self.Config.DATA_AUGMENTATION:
            if type == "train":
                # scale: inverted: 0.5 -> bigger; 2 -> smaller
                # patch_center_dist_from_border: if 144/2=72 -> always exactly centered; otherwise a bit off center
                # (brain can get off image and will be cut then)
                if self.Config.DAUG_SCALE:
                    # spatial transform automatically crops/pads to correct size
                    center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                    tfs.append(SpatialTransform(self.Config.INPUT_DIM,
                                                patch_center_dist_from_border=center_dist_from_border,
                                                do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM,
                                                alpha=self.Config.DAUG_ALPHA, sigma=self.Config.DAUG_SIGMA,
                                                do_rotation=self.Config.DAUG_ROTATE,
                                                angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8), angle_z=(-0.8, 0.8),
                                                do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                border_cval_data=0,
                                                order_data=3,
                                                border_mode_seg='constant', border_cval_seg=0,
                                                order_seg=0, random_crop=True,
                                                p_el_per_sample=self.Config.P_SAMP,
                                                p_rot_per_sample=self.Config.P_SAMP,
                                                p_scale_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

                if self.Config.DAUG_RESAMPLE_LEGACY:
                    tfs.append(ResampleTransformLegacy(zoom_range=(0.5, 1)))

                if self.Config.DAUG_GAUSSIAN_BLUR:
                    tfs.append(GaussianBlurTransform(blur_sigma=self.Config.DAUG_BLUR_SIGMA,
                                                     different_sigma_per_channel=False,
                                                     p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=self.Config.DAUG_NOISE_VARIANCE,
                                                      p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_generator, Compose(tfs), num_processes=num_processes,
                                           num_cached_per_queue=1, seeds=None, pin_memory=True)
        return batch_gen    # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


    def get_batch_generator(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        if self.Config.TYPE == "combined":
            batch_gen = BatchGenerator2D_Npy_random((data, seg), batch_size=batch_size)
        else:
            batch_gen = BatchGenerator2D_Nifti_random((data, seg), batch_size=batch_size)
            # batch_gen = SlicesBatchGeneratorRandomNiftiImg_5slices((data, seg), batch_size=batch_size)

        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)

        return batch_gen



############################################################################################################
# Backup
############################################################################################################

class BatchGenerator2D_Nifti_random_5slices(SlimDataLoaderBase):
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


