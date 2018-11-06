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

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, FlipVectorAxisTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import PadToMultipleTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import dataset_utils
from tractseg.libs import exp_utils
from tractseg.data.data_loader_training import load_training_data



class BatchGenerator3D_Nifti_random(SlimDataLoaderBase):
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
        # subject_idx = int(random.uniform(0, len(subjects)))     # len(subjects)-1 not needed because int always rounds to floor
        subject_idxs = np.random.choice(len(subjects), self.batch_size, False, None)

        x = []
        y = []
        for subject_idx in subject_idxs:
            data, seg = load_training_data(self.Config, subjects[subject_idx])  # (x, y, z, channels)
            data = data.transpose(3, 0, 1, 2)  # channels have to be first
            seg = seg.transpose(3, 0, 1, 2)
            x.append(data)
            y.append(seg)

        data_dict = {"data": np.array(x),     # (batch_size, channels, x, y, [z])
                     "seg": np.array(y)}      # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderTraining:

    def __init__(self, Config):
        self.Config = Config

    def _augment_data(self, batch_generator, type=None):

        if self.Config.DATA_AUGMENTATION:
            num_processes = 8  # 6 is a bit faster than 16
        else:
            num_processes = 6

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
                                                do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM,
                                                alpha=(90., 120.), sigma=(9., 11.),
                                                do_rotation=self.Config.DAUG_ROTATE,
                                                angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8), angle_z=(-0.8, 0.8),
                                                do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                border_cval_data=0,
                                                order_data=3,
                                                border_mode_seg='constant', border_cval_seg=0,
                                                order_seg=0, random_crop=True))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(ResampleTransform(zoom_range=(0.5, 1)))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05)))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_generator, Compose(tfs), num_processes=num_processes,
                                           num_cached_per_queue=1, seeds=None)
        return batch_gen    # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


    def get_batch_generator(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        if self.Config.TYPE == "combined":
            raise NotImplementedError("Not implemented yet")
        else:
            batch_gen = BatchGenerator3D_Nifti_random((data, seg), batch_size=batch_size)

        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)

        return batch_gen

