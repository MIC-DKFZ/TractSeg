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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import nibabel as nib
import numpy as np
import random
import os

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from tractseg.libs.system_config import SystemConfig as C

np.random.seed(1337)  # for reproducibility


class BatchGenerator2D_PrecomputedBatches(SlimDataLoaderBase):
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


def _augment_data(Config, batch_generator, type=None):
    batch_gen = MultiThreadedAugmenter(batch_generator, Compose([]), num_processes=1, num_cached_per_queue=1, seeds=None)
    return batch_gen


class DataLoaderPrecomputed():

    def __init__(self, Config):
        self.Config = Config

    def get_batch_generator(self, batch_size=128, type=None):
        data = type
        seg = []

        batch_gen = BatchGenerator2D_PrecomputedBatches((data, seg), batch_size=batch_size)
        batch_gen.Config = self.Config

        batch_gen = _augment_data(self.Config, batch_gen, type=type)
        return batch_gen


    def get_batch_generator_noDLBG(self, batch_size=None, type=None, subjects=None, num_batches=None):
        '''
        Somehow MultiThreadedAugmenter (with num_processes=1 and num_cached_per_queue=1) in ep1 fast (7s) but after
        that slower (10s). With this manual Iterator time is always the same (7.5s).
        '''
        print("Loading data from: " + join(C.DATA_PATH, self.Config.DATASET_FOLDER))

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

