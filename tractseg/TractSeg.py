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
from tractseg.libs.DataManagersInference import DataManagerSingleSubjectByFile
from tractseg.libs.Trainer import Trainer
from tractseg.models.BaseModel import BaseModel

def run_tractseg(data, output_type="tract_segmentation", input_type="peaks",
                 single_orientation=False, verbose=False, dropout_sampling=False, threshold=0.5,
                 bundle_specific_threshold=False, get_probs=False, peak_threshold=0.1,
                 postprocess=False, peak_regression_part="All"):
    '''
    Run TractSeg

    :param data: input peaks (4D numpy array with shape [x,y,z,9])
    :param output_type: "tract_segmentation" | "endings_segmentation" | "TOM" | "dm_regression"
    :param input_type: "peaks"
    :param verbose: show debugging infos
    :param dropout_sampling: create uncertainty map by monte carlo dropout (https://arxiv.org/abs/1506.02142)
    :param threshold: Threshold for converting probability map to binary map
    :param bundle_specific_threshold: Threshold is lower for some bundles which need more sensitivity (CA, CST, FX)
    :param get_probs: Output raw probability map instead of binary map
    :param peak_threshold: all peaks shorter than peak_threshold will be set to zero
    :return: 4D numpy array with the output of tractseg
        for tract_segmentation:     [x,y,z,nr_of_bundles]
        for endings_segmentation:   [x,y,z,2*nr_of_bundles]
        for TOM:                    [x,y,z,3*nr_of_bundles]
    '''
    start_time = time.time()

    config = get_config_name(input_type, output_type, dropout_sampling=dropout_sampling)
    HP = getattr(importlib.import_module("tractseg.config.PretrainedModels." + config), "HP")()
    HP.VERBOSE = verbose
    HP.TRAIN = False
    HP.TEST = False
    HP.SEGMENT = False
    HP.GET_PROBS = get_probs
    HP.LOAD_WEIGHTS = True
    HP.DROPOUT_SAMPLING = dropout_sampling
    HP.THRESHOLD = threshold

    if bundle_specific_threshold:
        HP.GET_PROBS = True

    if input_type == "peaks":
        if HP.EXPERIMENT_TYPE == "tract_segmentation" and HP.DROPOUT_SAMPLING:
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_dropout_v2.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "x_Pretrained_TractSeg_Models/TractSeg_12g90g270g_125mm_DS_DAugAll_Dropout", "best_weights_ep407.npz")
        elif HP.EXPERIMENT_TYPE == "tract_segmentation":
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_v2.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "x_Pretrained_TractSeg_Models/TractSeg_T1_12g90g270g_125mm_DAugAll", "best_weights_ep392.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg72_888", "best_weights_ep247.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg72_888_SchizoFineT_lr001", "best_weights_ep186.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg_12g90g270g_125mm_DS_DAugAll_RotMir", "best_weights_ep200.npz")
        elif HP.EXPERIMENT_TYPE == "endings_segmentation":
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_endings_segmentation_v3.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "EndingsSeg_12g90g270g_125mm_DS_DAugAll", "best_weights_ep234.npz")
        # elif HP.EXPERIMENT_TYPE == "peak_regression":
        #     HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_peak_regression_v1.npz")
        #     # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "x_Pretrained_TractSeg_Models/Peaks20_12g90g270g_125mm_DAugSimp_constW5", "best_weights_ep441.npz")  #more oversegmentation with DAug
        elif HP.EXPERIMENT_TYPE == "dm_regression":
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_dm_regression_v1.npz")
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "DmReg_12g90g270g_125mm_DAugAll_Ubuntu", "best_weights_ep80.npz")
    elif input_type == "T1":
        if HP.EXPERIMENT_TYPE == "tract_segmentation":
            # HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_v1.npz")
            HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes/x_Pretrained_TractSeg_Models", "TractSeg_T1_125mm_DAugAll", "best_weights_ep142.npz")
        elif HP.EXPERIMENT_TYPE == "endings_segmentation":
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_endings_segmentation_v1.npz")
        elif HP.EXPERIMENT_TYPE == "peak_regression":
            HP.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_peak_regression_v1.npz")

    if HP.VERBOSE:
        print("Hyperparameters:")
        ExpUtils.print_HPs(HP)

    data = np.nan_to_num(data)
    # brain_mask = ImgUtils.simple_brain_mask(data)
    # if HP.VERBOSE:
    #     nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), "otsu_brain_mask_DEBUG.nii.gz")

    if input_type == "T1":
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    data, seg_None, bbox, original_shape = DatasetUtils.crop_to_nonzero(data)
    data, transformation = DatasetUtils.pad_and_scale_img_to_square_img(data, target_size=HP.INPUT_DIM[0])

    if HP.EXPERIMENT_TYPE == "tract_segmentation" or HP.EXPERIMENT_TYPE == "endings_segmentation" or HP.EXPERIMENT_TYPE == "dm_regression":
        print("Loading weights from: {}".format(HP.WEIGHTS_PATH))
        HP.NR_OF_CLASSES = len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])
        Utils.download_pretrained_weights(experiment_type=HP.EXPERIMENT_TYPE, dropout_sampling=HP.DROPOUT_SAMPLING)
        model = BaseModel(HP)
        if single_orientation:     # mainly needed for testing because of less RAM requirements
            dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
            trainerSingle = Trainer(model, dataManagerSingle)
            if HP.DROPOUT_SAMPLING or HP.EXPERIMENT_TYPE == "dm_regression" or HP.GET_PROBS:
                seg, img_y = trainerSingle.get_seg_single_img(HP, probs=True, scale_to_world_shape=False, only_prediction=True)
            else:
                seg, img_y = trainerSingle.get_seg_single_img(HP, probs=False, scale_to_world_shape=False, only_prediction=True)
        else:
            seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False, only_prediction=True)
            if HP.DROPOUT_SAMPLING or HP.EXPERIMENT_TYPE == "dm_regression" or HP.GET_PROBS:
                seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=True)
            else:
                seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=False)

    elif HP.EXPERIMENT_TYPE == "peak_regression":
        weights = {
            "Part1": "pretrained_weights_peak_regression_part1_v1.npz",
            "Part2": "pretrained_weights_peak_regression_part2_v1.npz",
            "Part3": "pretrained_weights_peak_regression_part3_v1.npz",
            "Part4": "pretrained_weights_peak_regression_part4_v1.npz",
        }
        if peak_regression_part == "All":
            parts = ["Part1", "Part2", "Part3", "Part4"]
            seg_all = np.zeros((data.shape[0], data.shape[1], data.shape[2], HP.NR_OF_CLASSES * 3))
        else:
            parts = [peak_regression_part]
            HP.CLASSES = "All_" + peak_regression_part
            HP.NR_OF_CLASSES = 3 * len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])

        for idx, part in enumerate(parts):
            # HP.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "x_Pretrained_TractSeg_Models/" + weights[part])
            HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, weights[part])
            print("Loading weights from: {}".format(HP.WEIGHTS_PATH))
            HP.CLASSES = "All_" + part
            HP.NR_OF_CLASSES = 3 * len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])
            Utils.download_pretrained_weights(experiment_type=HP.EXPERIMENT_TYPE, dropout_sampling=HP.DROPOUT_SAMPLING, part=part)
            dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
            model = BaseModel(HP)
            trainerSingle = Trainer(model, dataManagerSingle)
            seg, img_y = trainerSingle.get_seg_single_img(HP, probs=True, scale_to_world_shape=False, only_prediction=True)

            if peak_regression_part == "All":
                seg_all[:, :, :, (idx*HP.NR_OF_CLASSES) : (idx*HP.NR_OF_CLASSES+HP.NR_OF_CLASSES)] = seg

        if peak_regression_part == "All":
            HP.CLASSES = "All"
            HP.NR_OF_CLASSES = 3 * len(ExpUtils.get_bundle_names(HP.CLASSES)[1:])
            seg = seg_all

        #quite fast
        if bundle_specific_threshold:
            seg = ImgUtils.remove_small_peaks_bundle_specific(seg, ExpUtils.get_bundle_names(HP.CLASSES)[1:], len_thr=0.3)
        else:
            seg = ImgUtils.remove_small_peaks(seg, len_thr=peak_threshold)

        #3 dir for Peaks -> not working (?)
        # seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False, only_prediction=True)
        # seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=True)

    if bundle_specific_threshold and HP.EXPERIMENT_TYPE == "tract_segmentation":
        seg = ImgUtils.probs_to_binary_bundle_specific(seg, ExpUtils.get_bundle_names(HP.CLASSES)[1:])

    #remove following two lines to keep super resolution
    seg = DatasetUtils.cut_and_scale_img_back_to_original_img(seg, transformation)  #quite slow
    seg = DatasetUtils.add_original_zero_padding_again(seg, bbox, original_shape, HP.NR_OF_CLASSES) #quite slow

    if postprocess:
        seg = ImgUtils.postprocess_segmentations(seg, blob_thr=50, hole_closing=2)

    ExpUtils.print_verbose(HP, "Took {}s".format(round(time.time() - start_time, 2)))
    return seg

