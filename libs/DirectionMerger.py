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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import importlib
import numpy as np

class DirectionMerger:

    @staticmethod
    def get_seg_single_img_3_directions(HP, model, subject=None, data=None, scale_to_world_shape=True):
        '''
        Returns probs

        :param HP:
        :param model:
        :param subject:
        :param data:
        :param scale_to_world_shape:
        :return:
        '''
        from libs.Trainer import Trainer

        #todo important: change
        #Adapt for PredictByFile
        # DataManagerSingleSubject = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectById")
        DataManagerSingleSubject = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectByFile")

        prob_slices = []
        directions = ["x", "y", "z"]
        # directions = ["x"]
        for direction in directions:
            HP.SLICE_DIRECTION = direction
            print("Using direction: " + HP.SLICE_DIRECTION)

            #todo important: change
            # Adapt for PredictByFile
            # dataManagerSingle = DataManagerSingleSubject(HP, subject=subject)
            dataManagerSingle = DataManagerSingleSubject(HP, data=data)

            trainerSingle = Trainer(model, dataManagerSingle)
            img_probs, img_y = trainerSingle.get_seg_single_img(HP, probs=True, scale_to_world_shape=scale_to_world_shape)    # (x, y, z, nrClasses)
            prob_slices.append(img_probs)
            # img_probs = np.reshape(img_probs, (-1, img_probs.shape[-1]))  # Flatten all dims except nrClasses dim
            # img_y = np.reshape(img_y, (-1, img_y.shape[-1]))

        probs_x, probs_y, probs_z = prob_slices
        new_shape = probs_x.shape + (1,)  # (146, 174, 146, 45)  -> (146, 174, 146, 45, 1)
        probs_x = np.reshape(probs_x, new_shape)
        probs_y = np.reshape(probs_y, new_shape)
        probs_z = np.reshape(probs_z, new_shape)

        probs_combined = np.concatenate((probs_x, probs_y, probs_z), axis=4)    # (146, 174, 146, 45, 3)
        return probs_combined, img_y

    @staticmethod
    def mean_fusion(threshold, img, probs=True):
        '''
        :param img: 5D Image with probability per direction, shape: (x, y, z, nr_classes, 3)
        :return: 4D image, shape (x, y, z, nr_classes)
        '''
        print("Taking Mean")
        probs_mean = img.mean(axis=4)
        if not probs:
            probs_mean[probs_mean >= threshold] = 1
            probs_mean[probs_mean < threshold] = 0
            probs_mean = probs_mean.astype(np.int16)
        return probs_mean

    @staticmethod
    def majority_fusion(threshold, img, probs=None):
        #Combine with Majority Voting
        #  -> no so good because lose probability information (only binary afterwards)
        #  -> with mean slightly better results (+0.002)
        #  => Use mean
        print("Majority Voting")
        img[img >= threshold] = 1
        img[img < threshold] = 0
        probs_combined = img.astype(np.int16)
        probs_sum = probs_combined.sum(axis=4)
        probs_result = np.zeros(probs_sum.shape)
        probs_result[probs_sum >= 2] = 1   #majority is at least 2 of 3
        probs_result[probs_sum < 2] = 0
        return probs_result.astype(np.int16)
