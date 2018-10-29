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

from tractseg.data.data_loader_training import BatchGenerator2D_Nifti_random
from tractseg.data.data_loader_training import BatchGenerator2D_Npy_random
from tractseg.data.data_loader_inference import BatchGenerator2D_data_ordered_standalone
from tractseg.libs.system_config import SystemConfig as C
from tractseg.data.DLDABG_standalone import ReorderSegTransform
from tractseg.data.data_loader_training import load_training_data

np.random.seed(1337)  # for reproducibility


class DataLoaderTrainingSingle:

    def __init__(self, Config, subject=None):
        self.Config = Config
        self.subject = subject

    def _augment_data(self, batch_generator, type=None):
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
        batch_gen = SingleThreadedAugmenter(batch_generator, Compose(tfs))
        return batch_gen

    def get_batch_generator(self, batch_size=1):
        if self.Config.TYPE == "combined":
            # Load from Npy file for Fusion
            data = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject,
                                self.Config.FEATURES_FILENAME + ".npy"), mmap_mode="r")
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject,
                               self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")
            data = np.nan_to_num(data)
            seg = np.nan_to_num(seg)
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            batch_gen = BatchGenerator2D_data_ordered_standalone((data, seg), batch_size)
        else:
            data, seg = load_training_data(self.Config, self.subject)
            batch_gen = BatchGenerator2D_data_ordered_standalone((data, seg), batch_size)
        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)
        return batch_gen

