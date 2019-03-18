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
import numpy as np

from tractseg.libs import exp_utils
from tractseg.libs.system_config import SystemConfig as C


class Config:
    """Settings and Hyperparameters"""
    EXP_MULTI_NAME = ""  #CV Parent Dir name; leave empty for Single Bundle Experiment
    EXP_NAME = "HCP_TEST"
    MODEL = "UNet_Pytorch_DeepSup"
    # tract_segmentation / endings_segmentation / dm_regression / peak_regression
    EXPERIMENT_TYPE = "tract_segmentation"

    DIM = "2D"  # 2D / 3D
    NUM_EPOCHS = 250
    EPOCH_MULTIPLIER = 1 #2D: 1, 3D: 12 for lowRes, 3 for highRes
    DATA_AUGMENTATION = True
    DAUG_SCALE = True
    DAUG_NOISE = True
    DAUG_NOISE_VARIANCE = (0, 0.05)
    DAUG_ELASTIC_DEFORM = True
    DAUG_ALPHA = (90., 120.)
    DAUG_SIGMA = (9., 11.)
    DAUG_RESAMPLE = False   # does not change validation dice (if using Gaussian_blur) -> deactivate
    DAUG_RESAMPLE_LEGACY = False    # does not change validation dice (at least on AutoPTX) -> deactivate
    DAUG_GAUSSIAN_BLUR = True
    DAUG_BLUR_SIGMA = (0, 1)
    DAUG_ROTATE = False
    DAUG_MIRROR = False
    DAUG_FLIP_PEAKS = False
    P_SAMP = 1.0    # 1.0 slightly less overfitting than 0.4 but not much ("break-even" 20epochs later)
    DAUG_INFO = "Elastic(90,120)(9,11) - Scale(0.9, 1.5) - CenterDist60 - " \
                "DownsampScipy(0.5,1) - Gaussian(0,0.05) - Rotate(-0.8,0.8)"
    DATASET = "HCP"  # HCP / HCP_32g / Schizo
    RESOLUTION = "1.25mm" # 1.25mm (/ 2.5mm)
    # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    FEATURES_FILENAME = "12g90g270g"

    LABELS_FILENAME = ""        # autofilled
    LOSS_FUNCTION = "default"   # default / soft_batch_dice
    OPTIMIZER = "Adamax"
    CLASSES = "All"             # All / 11 / 20 / CST_right
    NR_OF_GRADIENTS = 9
    NR_OF_CLASSES = len(exp_utils.get_bundle_names(CLASSES)[1:])
    # NR_OF_CLASSES = 3 * len(exp_utils.get_bundle_names(CLASSES)[1:])

    INPUT_DIM = None  # (80, 80) / (144, 144)
    LOSS_WEIGHT = 1  # 1: no weighting
    LOSS_WEIGHT_LEN = -1  # -1: constant over all epochs
    SLICE_DIRECTION = "y"  # x, y, z  (combined needs z)
    TRAINING_SLICE_DIRECTION = "xyz"    # y / xyz
    INFO = "-"
    BATCH_NORM = False
    WEIGHT_DECAY = 0
    USE_DROPOUT = False
    DROPOUT_SAMPLING = False
    # DATASET_FOLDER = "HCP_batches/270g_125mm_bundle_peaks_Y_subset"
    DATASET_FOLDER = "HCP_preproc" # HCP / Schizo
    LABELS_FOLDER = "bundle_masks"  # bundle_masks / bundle_masks_dm
    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    BATCH_SIZE = 47  #Peak Prediction: 44 #Pytorch: 50  #Lasagne: 56  #Lasagne combined: 42  #Pytorch UpSample: 56
    LEARNING_RATE = 0.001  # 0.002 #LR find: 0.000143 ?  # 0.001
    LR_SCHEDULE = False
    UNET_NR_FILT = 64
    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "HCP100_45B_UNet_x_DM_lr002_slope2_dec992_ep800/best_weights_ep64.npz")
    WEIGHTS_PATH = ""  # if empty string: autoloading the best_weights in get_best_weights_path()
    TYPE = "single_direction"  # single_direction / combined
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []
    TRAIN = True
    TEST = True
    SEGMENT = False
    GET_PROBS = False
    OUTPUT_MULTIPLE_FILES = False
    RESET_LAST_LAYER = False
    UPSAMPLE_TYPE = "bilinear"  # bilinear / nearest

    # Peak_regression specific
    PEAK_DICE_THR = [0.95]
    PEAK_DICE_LEN_THR = 0.05
    FLIP_OUTPUT_PEAKS = True    # flip peaks along z axis to make them compatible with MITK

    # For TractSeg.py application
    PREDICT_IMG = False
    PREDICT_IMG_OUTPUT = None
    TRACTSEG_DIR = "tractseg_output"
    KEEP_INTERMEDIATE_FILES = False
    CSD_RESOLUTION = "LOW"  # HIGH / LOW
    NR_CPUS = -1

    # Rarly changed:
    LABELS_TYPE = np.int16  # Binary: np.int16, Regression: np.float32
    THRESHOLD = 0.5  # Binary: 0.5, Regression: 0.01 ?
    TEST_TIME_DAUG = False
    USE_VISLOGGER = False  #only works with Python 3
    SAVE_WEIGHTS = True
    SEG_INPUT = "Peaks"  # Gradients/ Peaks
    NR_SLICES = 1  # adapt manually: NR_OF_GRADIENTS in UNet.py and get_batch... in train() and in get_seg_prediction()
    PRINT_FREQ = 20  #20
    NORMALIZE_DATA = True
    NORMALIZE_PER_CHANNEL = False
    BEST_EPOCH = 0
    VERBOSE = True
    CALC_F1 = True
