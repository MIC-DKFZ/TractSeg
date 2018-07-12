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

import warnings
warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)    #hide h5py warnings

import importlib
import numpy as np
import time
import nibabel as nib
from os.path import join

from tractseg.libs.Config import Config as C
from tractseg.libs.Config import get_config_name
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.Utils import Utils
from tractseg.libs.DatasetUtils import DatasetUtils
from tractseg.libs.DirectionMerger import DirectionMerger
from tractseg.libs.ImgUtils import ImgUtils
from tractseg.libs.DataManagers import DataManagerSingleSubjectByFile
from tractseg.libs.Trainer import Trainer

def run_tractseg(data, output_type="tract_segmentation", input_type="peaks",
                 single_orientation=False, verbose=False, dropout_sampling=False):
    '''
    Run TractSeg

    :param data: input peaks (4D numpy array with shape [x,y,z,9])
    :param brain_mask: 3D numpy array with shape [x,y,z]
    :param output_type: "tract_segmentation" | "endings_segmentation" | "TOM"
    :param input_type: "peaks"
    :param verbose: show debugging infos
    :return: 4D numpy array with the output of tractseg
        for tract_segmentation:     [x,y,z,nr_of_bundles]
        for endings_segmentation:   [x,y,z,2*nr_of_bundles]
        for TOM:                    [x,y,z,3*nr_of_bundles]
    '''
    start_time = time.time()

    config = get_config_name(input_type, output_type)
    HP = getattr(importlib.import_module("tractseg.config.PretrainedModels." + config), "HP")()
    HP.VERBOSE = verbose
    HP.TRAIN = False
    HP.TEST = False
    HP.SEGMENT = False
    HP.GET_PROBS = False
    HP.LOAD_WEIGHTS = True
    HP.DROPOUT_SAMPLING = dropout_sampling

    if input_type == "peaks":
        if HP.EXPERIMENT_TYPE == "tract_segmentation":
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_tract_segmentation_v1.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg_270g_125mm_run2", "best_weights_ep136.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg_12g90g270g_125mm_DAugAll_Dropout", "best_weights_ep114.npz")
        elif HP.EXPERIMENT_TYPE == "endings_segmentation":
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_endings_segmentation_v1.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "EndingsSeg_12g90g270g_125mm_DAugAll", "best_weights_ep16.npz")
        elif HP.EXPERIMENT_TYPE == "peak_regression":
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_peak_regression_v1.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "Peaks20_270g_125mm_LW5", "best_weights_ep144.npz")
    elif input_type == "T1":
        if HP.EXPERIMENT_TYPE == "tract_segmentation":
            # HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_tract_segmentation_v1.npz")
            HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg_T1_125mm_DAugAll", "best_weights_ep142.npz")
        elif HP.EXPERIMENT_TYPE == "endings_segmentation":
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_endings_segmentation_v1.npz")
        elif HP.EXPERIMENT_TYPE == "peak_regression":
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights_peak_regression_v1.npz")
    print("Loading weights from: {}".format(HP.WEIGHTS_PATH))

    ModelClass = getattr(importlib.import_module("tractseg.models." + HP.MODEL), HP.MODEL)   # run early before code changes in background

    if HP.EXPERIMENT_TYPE == "peak_regression":
        HP.NR_OF_CLASSES = 3*len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])
    else:
        HP.NR_OF_CLASSES = len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])

    if HP.VERBOSE:
        print("Hyperparameters:")
        ExpUtils.print_HPs(HP)

    Utils.download_pretrained_weights(experiment_type=HP.EXPERIMENT_TYPE)

    data = np.nan_to_num(data)
    # brain_mask = ImgUtils.simple_brain_mask(data)
    # if HP.VERBOSE:
    #     nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), "otsu_brain_mask_DEBUG.nii.gz")

    if input_type == "T1":
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    data, seg_None, bbox, original_shape = DatasetUtils.crop_to_nonzero(data)
    data, transformation = DatasetUtils.pad_and_scale_img_to_square_img(data, target_size=HP.INPUT_DIM[0])

    model = ModelClass(HP)

    if HP.EXPERIMENT_TYPE == "tract_segmentation" or HP.EXPERIMENT_TYPE == "endings_segmentation":
        if single_orientation:     # mainly needed for testing because of less RAM requirements
            dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
            trainerSingle = Trainer(model, dataManagerSingle)
            if HP.DROPOUT_SAMPLING:
                seg, img_y = trainerSingle.get_seg_single_img(HP, probs=True, scale_to_world_shape=False)
            else:
                seg, img_y = trainerSingle.get_seg_single_img(HP, probs=False, scale_to_world_shape=False)
        else:
            seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False)
            if HP.DROPOUT_SAMPLING:
                seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=True)
            else:
                seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=False)

    elif HP.EXPERIMENT_TYPE == "peak_regression":
        dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
        trainerSingle = Trainer(model, dataManagerSingle)
        seg, img_y = trainerSingle.get_seg_single_img(HP, probs=True, scale_to_world_shape=False)
        seg = ImgUtils.remove_small_peaks(seg, len_thr=0.3)
        #3 dir for Peaks -> not working (?)
        # seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False)
        # seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=True)

    seg = DatasetUtils.cut_and_scale_img_back_to_original_img(seg, transformation)
    seg = DatasetUtils.add_original_zero_padding_again(seg, bbox, original_shape, HP.NR_OF_CLASSES)
    ExpUtils.print_verbose(HP, "Took {}s".format(round(time.time() - start_time, 2)))

    return seg

