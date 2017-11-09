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

import inspect
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

from os.path import join
import nibabel as nib
from libs.ImgUtils import ImgUtils
from libs.BatchGenerators import SlicesBatchGeneratorRandomNiftiImg
from libs.BatchGenerators import SlicesBatchGenerator
from libs.BatchGenerators_fusion import SlicesBatchGeneratorRandomNpyImg_fusion
from libs.BatchGenerators_fusion import SlicesBatchGeneratorNpyImg_fusion
from tractseg.libs.DatasetUtils import DatasetUtils
from libs.Config import Config as C
from libs.ExpUtils import ExpUtils
from tractseg.libs.AugmentationGenerators import *

from DeepLearningBatchGeneratorUtils.MultiThreadedGenerator import MultiThreadedGenerator
from DeepLearningBatchGeneratorUtils.SpatialTransformGenerators import *
from DeepLearningBatchGeneratorUtils.ResamplingAugmentationGenerators import linear_downsampling_generator_scipy
from DeepLearningBatchGeneratorUtils.SampleNormalizationGenerators import zero_one_normalization_generator
from DeepLearningBatchGeneratorUtils.ColorAugmentationGenerators import *
from DeepLearningBatchGeneratorUtils.NoiseGenerators import gaussian_noise_generator

np.random.seed(1337)  # for reproducibility

class DataManagerSingleSubjectById:
    def __init__(self, HP, subject=None, use_gt_mask=True):
        self.subject = subject
        self.HP = HP
        self.use_gt_mask = use_gt_mask

        if self.HP.TYPE == "single_direction":
            self.data_dir = join(C.HOME, self.HP.DATASET_FOLDER, subject)
        elif self.HP.TYPE == "combined":
            self.data_dir = join(C.HOME, self.HP.DATASET_FOLDER, subject, self.HP.FEATURES_FILENAME + ".npy") #data_dir not used when doing fusion

        print("Loading data from: " + self.data_dir)

    def get_batches(self, batch_size=1):

        num_processes = 1   # not not use more than 1 if you want to keep original slice order (Threads do return in random order)

        if self.HP.TYPE == "combined":
            # Load from Npy file for Fusion
            data = self.subject
            seg = []
            nr_of_samples = len([self.subject]) * self.HP.INPUT_DIM[0]
            num_batches = int(nr_of_samples / batch_size / num_processes)
            batch_gen = SlicesBatchGeneratorNpyImg_fusion((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches, seed=None)
        else:
            # Load Features
            data_img = nib.load(join(self.data_dir, self.HP.FEATURES_FILENAME + ".nii.gz"))
            data = data_img.get_data()
            data = np.nan_to_num(data)
            data = DatasetUtils.scale_input_to_unet_shape(data, self.HP.DATASET, self.HP.RESOLUTION)
            # data = DatasetUtils.scale_input_to_unet_shape(data, "HCP_32g", "1.25mm")  #If we want to test HCP_32g on HighRes net

            #Load Segmentation
            if self.use_gt_mask:
                seg = nib.load(join(self.data_dir, self.HP.LABELS_FILENAME + ".nii.gz")).get_data()
            else:
                # Use dummy mask in case we only want to predict on some data (where we do not have Ground Truth))
                spacing = ImgUtils.get_dwi_spacing("HCP", "1.25mm")
                seg = np.zeros((spacing[0], spacing[1], spacing[2], self.HP.NR_OF_CLASSES)).astype(self.HP.LABELS_TYPE)

            if self.HP.DATASET in ["HCP_2mm", "HCP_2.5mm", "HCP_32g"]:
                # By using "HCP" but lower resolution scale_input_to_unet_shape will automatically downsample the HCP sized seg_mask
                seg = DatasetUtils.scale_input_to_unet_shape(seg, "HCP", self.HP.RESOLUTION)
            else:
                seg = DatasetUtils.scale_input_to_unet_shape(seg, self.HP.DATASET, self.HP.RESOLUTION) # Seg has same resolution as probmaps -> we can use same resizing
            batch_gen = SlicesBatchGenerator((data, seg), BATCH_SIZE=batch_size)

        batch_gen.HP = self.HP

        if self.HP.NORMALIZE_DATA:
            batch_gen = zero_one_normalization_generator(batch_gen)

        if self.HP.TEST_TIME_DAUG:
            center_dist_from_border = int(self.HP.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
            batch_gen = ultimate_transform_generator_v2(batch_gen, self.HP.INPUT_DIM,
                                                        patch_center_dist_from_border=center_dist_from_border,
                                                        do_elastic_deform=True, alpha=(90., 120.), sigma=(9., 11.),
                                                        do_rotation=True, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
                                                        angle_z=(-0.8, 0.8),
                                                        do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                        border_cval_data=0,
                                                        order_data=3,
                                                        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True)
            # batch_gen = linear_downsampling_generator_scipy(batch_gen, zoom_range=(0.5, 1))
            # batch_gen = gaussian_noise_generator(batch_gen, noise_variance=(0, 0.05))
            batch_gen = contrast_augmentation_generator(batch_gen, contrast_range=(0.7, 1.3), per_channel=False)
            batch_gen = brightness_augmentation_by_multiplication_generator(batch_gen, multiplier_range=(0.7, 1.3), per_channel=False)

        batch_gen = reorder_seg_generator(batch_gen)    #reorder seg so we can easily compute F1 manually
        batch_gen = MultiThreadedGenerator(batch_gen, num_processes=num_processes, num_cached_per_queue=2) # Only use num_processes=1, otherwise global_idx of SlicesBatchGenerator not working
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, x, y, channels)


class DataManagerSingleSubjectByFile:
    def __init__(self, HP, data):
        self.data = data
        self.HP = HP
        ExpUtils.print_verbose(self.HP, "Loading data from PREDICT_IMG input file")

    def get_batches(self, batch_size=1):
        data = np.nan_to_num(self.data)
        # Use dummy mask in case we only want to predict on some data (where we do not have Ground Truth))
        seg = np.zeros((144, 144, 144, self.HP.NR_OF_CLASSES)).astype(self.HP.LABELS_TYPE)

        num_processes = 1  # not not use more than 1 if you want to keep original slice order (Threads do return in random order)
        batch_gen = SlicesBatchGenerator((data, seg), BATCH_SIZE=batch_size)
        batch_gen.HP = self.HP

        if self.HP.NORMALIZE_DATA:
            batch_gen = zero_one_normalization_generator(batch_gen)
        # batch_gen = linear_downsampling_generator(batch_gen, max_downsampling_factor=2, isotropic=True)
        batch_gen = reorder_seg_generator(batch_gen)    #reorder seg so we can easily compute F1 manually

        batch_gen = MultiThreadedGenerator(batch_gen, num_processes=num_processes, num_cached_per_queue=2) # Only use num_processes=1, otherwise global_idx of SlicesBatchGenerator not working
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, x, y, channels)


class DataManagerTrainingNiftiImgs:
    def __init__(self, HP):
        self.HP = HP
        print("Loading data from: " + join(C.HOME, self.HP.DATASET_FOLDER))

    def get_batches(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        num_processes = 16
        nr_of_samples = len(subjects) * self.HP.INPUT_DIM[0]
        num_batches = int(nr_of_samples / batch_size / num_processes)

        if self.HP.TYPE == "combined":
            # Simple with .npy  -> just a little bit faster than Nifti (<10%) and f1 not better => use Nifti
            batch_gen = SlicesBatchGeneratorRandomNpyImg_fusion((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches, seed=None)
            # batch_gen = SlicesBatchGeneratorRandomNpyImg_fusionMean((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches, seed=None)
        else:
            batch_gen = SlicesBatchGeneratorRandomNiftiImg((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches, seed=None)

        batch_gen.HP = self.HP

        if self.HP.NORMALIZE_DATA:
            batch_gen = zero_one_normalization_generator(batch_gen)

        if self.HP.DATA_AUGMENTATION:
            if type == "train":
                #scale: inverted: 0.5 -> bigger; 2 -> smaller
                #patch_center_dist_from_border: if 144/2=72 -> always exactly centered; otherwise a bit off center (brain can get off image and will be cut then)
                center_dist_from_border = int(self.HP.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                batch_gen = ultimate_transform_generator_v2(batch_gen, self.HP.INPUT_DIM,
                                                            patch_center_dist_from_border=center_dist_from_border,
                                                            do_elastic_deform=True, alpha=(90., 120.), sigma=(9., 11.),
                                                            do_rotation=True, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
                                                            angle_z=(-0.8, 0.8),
                                                            do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                            border_cval_data=0,
                                                            order_data=3,
                                                            border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True)

                batch_gen = linear_downsampling_generator_scipy(batch_gen, zoom_range=(0.5, 1))                 #UNet results the same for Scipy and Nilearn. But Scipy about 30% faster.
                batch_gen = contrast_augmentation_generator(batch_gen, contrast_range=(0.7, 1.3), per_channel=False)
                batch_gen = gaussian_noise_generator(batch_gen, noise_variance=(0, 0.05))
                batch_gen = brightness_augmentation_by_multiplication_generator(batch_gen, multiplier_range=(0.7, 1.3), per_channel=False)

                batch_gen = mirror_axis_generator(batch_gen)
                # batch_gen = gamma_augmentation_generator(batch_gen, gamma_range=(0.75, 1.5))  # produces Loss=NaN; maybe because data not in 0-1



        batch_gen = MultiThreadedGenerator(batch_gen, num_processes=num_processes, num_cached_per_queue=1)
        return batch_gen    # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)