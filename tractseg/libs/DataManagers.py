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

from os.path import join
import nibabel as nib
import numpy as np
import random
import os

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessMultiplicativeTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import Mirror, SpatialTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from tractseg.libs.ImgUtils import ImgUtils
from tractseg.libs.BatchGenerators import SlicesBatchGeneratorRandomNiftiImg
from tractseg.libs.BatchGenerators import SlicesBatchGeneratorPrecomputedBatches
from tractseg.libs.BatchGenerators import SlicesBatchGeneratorRandomNiftiImg_5slices
from tractseg.libs.BatchGenerators import SlicesBatchGenerator
from tractseg.libs.BatchGenerators_fusion import SlicesBatchGeneratorRandomNpyImg_fusion
from tractseg.libs.BatchGenerators_fusion import SlicesBatchGeneratorNpyImg_fusion
from tractseg.libs.DatasetUtils import DatasetUtils
from tractseg.libs.Config import Config as C
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.AugmentationGenerators import *

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
        tfs = []  # transforms

        if self.HP.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=False))

        if self.HP.TEST_TIME_DAUG:
            center_dist_from_border = int(self.HP.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
            tfs.append(SpatialTransform(self.HP.INPUT_DIM,
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
        batch_gen = MultiThreadedAugmenter(batch_gen, Compose(tfs), num_processes=num_processes, num_cached_per_queue=2, seeds=None) # Only use num_processes=1, otherwise global_idx of SlicesBatchGenerator not working
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
        tfs = []  # transforms

        if self.HP.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=False))
        tfs.append(ReorderSegTransform())
        batch_gen = MultiThreadedAugmenter(batch_gen, Compose(tfs), num_processes=num_processes, num_cached_per_queue=2, seeds=None)  # Only use num_processes=1, otherwise global_idx of SlicesBatchGenerator not working
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, x, y, channels)


class DataManagerTrainingNiftiImgs:
    def __init__(self, HP):
        self.HP = HP
        print("Loading data from: " + join(C.HOME, self.HP.DATASET_FOLDER))

    def get_batches(self, batch_size=128, type=None, subjects=None, num_batches=None):
        data = subjects
        seg = []

        num_processes = 1  # 6 is a bit faster than 16
        nr_of_samples = len(subjects) * self.HP.INPUT_DIM[0]
        if num_batches is None:
            num_batches_multithr = int(nr_of_samples / batch_size / num_processes)   #number of batches for exactly one epoch
        else:
            num_batches_multithr = int(num_batches / num_processes)

        if self.HP.TYPE == "combined":
            # Simple with .npy  -> just a little bit faster than Nifti (<10%) and f1 not better => use Nifti
            batch_gen = SlicesBatchGeneratorRandomNpyImg_fusion((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches_multithr, seed=None)
            # batch_gen = SlicesBatchGeneratorRandomNpyImg_fusionMean((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches_multithr, seed=None)
        else:
            batch_gen = SlicesBatchGeneratorRandomNiftiImg((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches_multithr, seed=None)
            # batch_gen = SlicesBatchGeneratorRandomNiftiImg_5slices((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches_multithr, seed=None)

        batch_gen.HP = self.HP
        tfs = []  #transforms

        if self.HP.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=False))

        if self.HP.DATA_AUGMENTATION:
            if type == "train":
                # scale: inverted: 0.5 -> bigger; 2 -> smaller
                # patch_center_dist_from_border: if 144/2=72 -> always exactly centered; otherwise a bit off center (brain can get off image and will be cut then)

                center_dist_from_border = int(self.HP.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                tfs.append(SpatialTransform(self.HP.INPUT_DIM,
                                                    patch_center_dist_from_border=center_dist_from_border,
                                                    do_elastic_deform=True, alpha=(90., 120.), sigma=(9., 11.),
                                                    do_rotation=True, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
                                                    angle_z=(-0.8, 0.8),
                                                    do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                    border_cval_data=0,
                                                    order_data=3,
                                                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True))
                tfs.append(ResampleTransform(zoom_range=(0.5, 1)))
                tfs.append(ContrastAugmentationTransform(contrast_range=(0.7, 1.3), preserve_range=True, per_channel=False))
                tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05)))
                tfs.append(BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), per_channel=False))

                # tfs.append(Mirror())
                # tfs.append(GammaTransform(gamma_range=(0.75, 1.5))) # produces Loss=NaN; maybe because data not in 0-1

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_gen, Compose(tfs), num_processes=num_processes, num_cached_per_queue=1, seeds=None)
        return batch_gen    # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


class DataManagerPrecomputedBatches:
    def __init__(self, HP):
        self.HP = HP
        print("Loading data from: " + join(C.HOME, self.HP.DATASET_FOLDER))

    def get_batches(self, batch_size=128, type=None, subjects=None, num_batches=None):
        data = type
        seg = []

        num_processes = 1  # 6 is a bit faster than 16
        nr_of_samples = len(subjects) * self.HP.INPUT_DIM[0]
        if num_batches is None:
            num_batches_multithr = int(nr_of_samples / batch_size / num_processes)   #number of batches for exactly one epoch
        else:
            num_batches_multithr = int(num_batches / num_processes)


        batch_gen = SlicesBatchGeneratorPrecomputedBatches((data, seg), BATCH_SIZE=batch_size, num_batches=num_batches_multithr, seed=None)
        batch_gen.HP = self.HP

        batch_gen = MultiThreadedAugmenter(batch_gen, Compose([]), num_processes=num_processes, num_cached_per_queue=1, seeds=None)
        return batch_gen


class DataManagerPrecomputedBatches_noDLBG:
    '''
    Somehow MultiThreadedAugmenter (with num_processes=1 and num_cached_per_queue=1) in ep1 fast (7s) but after
    that slower (10s). With this manual Iterator time is always the same (7.5s).
    '''
    def __init__(self, HP):
        self.HP = HP
        print("Loading data from: " + join(C.HOME, self.HP.DATASET_FOLDER))

    def get_batches(self, batch_size=None, type=None, subjects=None, num_batches=None):
        num_processes = 1

        #todo important: change (use all test and validate again)
        if type == "train":
            nr_of_samples = len(subjects) * self.HP.INPUT_DIM[0]
            if num_batches is None:
                num_batches_multithr = int(nr_of_samples / batch_size / num_processes)   #number of batches for exactly one epoch
            else:
                num_batches_multithr = int(num_batches / num_processes)
        else:
            num_batches_multithr = 11

        #todo important: change
        num_batches_multithr = 11

        for i in range(num_batches_multithr):
            path = join(C.HOME, self.HP.DATASET_FOLDER, type)
            nr_of_files = len([name for name in os.listdir(path) if os.path.isfile(join(path, name))]) - 1
            idx = int(random.uniform(0, int(nr_of_files / 2.)))

            data = nib.load(join(path, "batch_" + str(idx) + "_data.nii.gz")).get_data()
            seg = nib.load(join(path, "batch_" + str(idx) + "_seg.nii.gz")).get_data()
            yield {"data": data, "seg": seg}