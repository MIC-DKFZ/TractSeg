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
Code to load data and to specify data augmentation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import nibabel as nib
import numpy as np
import random
import os

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, FlipVectorAxisTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import PadToMultipleTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from tractseg.libs import img_utils
from tractseg.libs.data_loaders import DataLoader2D_Nifti
from tractseg.libs.data_loaders import DataLoader2D_Npy
from tractseg.libs.data_loaders import DataLoader2D_PrecomputedBatches
from tractseg.libs.data_loaders import DataLoader2D_Nifti_5slices
from tractseg.libs.data_managers_inference import DataLoader2D_data_ordered_standalone
from tractseg.libs import dataset_utils
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import exp_utils
from tractseg.libs.DLDABG_standalone import ReorderSegTransform
from tractseg.libs.data_loaders import load_training_data

np.random.seed(1337)  # for reproducibility


class DataManagerSingleSubjectById:
    def __init__(self, Config, subject=None, use_gt_mask=True):
        self.subject = subject
        self.Config = Config
        self.use_gt_mask = use_gt_mask

    def get_batches(self, batch_size=1):

        if self.Config.TYPE == "combined":
            # Load from Npy file for Fusion
            data = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject, self.Config.FEATURES_FILENAME + ".npy"), mmap_mode="r")
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject, self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")
            data = np.nan_to_num(data)
            seg = np.nan_to_num(seg)
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            batch_gen = DataLoader2D_data_ordered_standalone((data, seg), batch_size)
        else:
            data, seg = load_training_data(self.Config, self.subject, use_gt_mask=self.use_gt_mask)
            batch_gen = DataLoader2D_data_ordered_standalone((data, seg), batch_size)

        batch_gen.Config = self.Config
        tfs = []  # transforms

        if self.Config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=False))

        if self.Config.TEST_TIME_DAUG:
            center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
            tfs.append(SpatialTransform(self.Config.INPUT_DIM,
                                        patch_center_dist_from_border=center_dist_from_border,
                                        do_elastic_deform=True, alpha=(90., 120.), sigma=(9., 11.),
                                        do_rotation=True, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
                                        angle_z=(-0.8, 0.8),
                                        do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                        border_cval_data=0,
                                        order_data=3,
                                        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True))
            # tfs.append(ResampleTransform(zoom_range=(0.5, 1)))
            # tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05)))
            tfs.append(ContrastAugmentationTransform(contrast_range=(0.7, 1.3), preserve_range=True, per_channel=False))
            tfs.append(BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), per_channel=False))


        tfs.append(ReorderSegTransform())
        batch_gen = SingleThreadedAugmenter(batch_gen, Compose(tfs))
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, x, y, channels)


class DataManagerTrainingNiftiImgs:
    '''
    This is the DataManager used during training
    '''

    def __init__(self, Config):
        self.Config = Config
        print("Loading data from: " + join(C.DATA_PATH, self.Config.DATASET_FOLDER))

    def get_batches(self, batch_size=128, type=None, subjects=None, num_batches=None):
        data = subjects
        seg = []

        #6 -> >30GB RAM
        if self.Config.DATA_AUGMENTATION:
            num_processes = 8  # 6 is a bit faster than 16
        else:
            num_processes = 6

        if self.HP.TYPE == "combined":
            batch_gen = DataLoader2D_Npy((data, seg), batch_size=batch_size)
        else:
            batch_gen = DataLoader2D_Nifti((data, seg), batch_size=batch_size)
            # batch_gen = SlicesBatchGeneratorRandomNiftiImg_5slices((data, seg), batch_size=batch_size)

        batch_gen.Config = self.Config
        tfs = []  #transforms

        if self.Config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        if self.Config.DATASET == "Schizo" and self.Config.RESOLUTION == "2mm":
            tfs.append(PadToMultipleTransform(16))

        if self.Config.DATA_AUGMENTATION:
            if type == "train":
                # scale: inverted: 0.5 -> bigger; 2 -> smaller
                # patch_center_dist_from_border: if 144/2=72 -> always exactly centered; otherwise a bit off center (brain can get off image and will be cut then)

                if self.Config.DAUG_SCALE:
                    center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                    tfs.append(SpatialTransform(self.Config.INPUT_DIM,
                                                        patch_center_dist_from_border=center_dist_from_border,
                                                        do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM, alpha=(90., 120.), sigma=(9., 11.),
                                                        do_rotation=self.Config.DAUG_ROTATE, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
                                                        angle_z=(-0.8, 0.8),
                                                        do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                        border_cval_data=0,
                                                        order_data=3,
                                                        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(ResampleTransform(zoom_range=(0.5, 1)))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05)))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_gen, Compose(tfs), num_processes=num_processes, num_cached_per_queue=1, seeds=None)
        return batch_gen    # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


class DataManagerPrecomputedBatches:
    def __init__(self, Config):
        self.Config = Config
        print("Loading data from: " + join(C.DATA_PATH, self.Config.DATASET_FOLDER))

    def get_batches(self, batch_size=128, type=None, subjects=None, num_batches=None):
        data = type
        seg = []

        num_processes = 1  # 6 is a bit faster than 16

        batch_gen = DataLoader2D_PrecomputedBatches((data, seg), batch_size=batch_size)
        batch_gen.Config = self.Config

        batch_gen = MultiThreadedAugmenter(batch_gen, Compose([]), num_processes=num_processes, num_cached_per_queue=1, seeds=None)
        return batch_gen


class DataManagerPrecomputedBatches_noDLBG:
    '''
    Somehow MultiThreadedAugmenter (with num_processes=1 and num_cached_per_queue=1) in ep1 fast (7s) but after
    that slower (10s). With this manual Iterator time is always the same (7.5s).
    '''
    def __init__(self, Config):
        self.Config = Config
        print("Loading data from: " + join(C.DATA_PATH, self.Config.DATASET_FOLDER))

    def get_batches(self, batch_size=None, type=None, subjects=None, num_batches=None):
        num_processes = 1

        nr_of_samples = len(subjects) * self.Config.INPUT_DIM[0]
        if num_batches is None:
            num_batches_multithr = int(nr_of_samples / batch_size / num_processes)   #number of batches for exactly one epoch
        else:
            num_batches_multithr = int(num_batches / num_processes)

        for i in range(num_batches_multithr):
            path = join(C.DATA_PATH, self.Config.DATASET_FOLDER, type)
            nr_of_files = len([name for name in os.listdir(path) if os.path.isfile(join(path, name))]) - 2
            idx = int(random.uniform(0, int(nr_of_files / 2.)))

            # data = nib.load(join(path, "batch_" + str(idx) + "_data.nii.gz")).get_data()
            # seg = nib.load(join(path, "batch_" + str(idx) + "_seg.nii.gz")).get_data()
            data = nib.load(join(path, "batch_" + str(idx) + "_data.nii.gz")).get_data()[:self.Config.BATCH_SIZE]
            seg = nib.load(join(path, "batch_" + str(idx) + "_seg.nii.gz")).get_data()[:self.Config.BATCH_SIZE]
            yield {"data": data, "seg": seg}