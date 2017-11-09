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

import argparse
import distutils.util

import importlib
import numpy as np
import os
import warnings
import time
import nibabel as nib
from os.path import join

from tractseg.libs.Config import Config as C
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.Utils import Utils
from tractseg.libs.DatasetUtils import DatasetUtils
from tractseg.libs.DirectionMerger import DirectionMerger
from tractseg.libs.ImgUtils import ImgUtils
from tractseg.libs.Mrtrix import Mrtrix

warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings

#Settings and Hyperparameters
class HP:
    EXP_MULTI_NAME = ""              #CV Parent Dir name # leave empty for Single Bundle Experiment
    EXP_NAME = "HCP_Pytorch_Mir"       # HCP_normAfter
    MODEL = "UNet_Pytorch"     # UNet_Lasagne / UNet_Pytorch
    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True
    DAUG_INFO = "Elastic(90,120)(9,11) - Scale(0.9, 1.5) - CenterDist60 - DownsampScipy(0.5,1) - Contrast(0.7,1.3) - Gaussian(0,0.05) - BrightnessMult(0.7,1.3) - RotateUltimate(-0.8,0.8) - Mirror"
    DATASET = "HCP"  # HCP / HCP_32g
    RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "270g_125mm_peaks"  # 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    LABELS_FILENAME = "bundle_masks"     # bundle_masks / bundle_masks_45       #Only used when using DataManagerNifti
    DATASET_FOLDER = "HCP"  # HCP / TRACED / HCP_fusion_npy_270g_125mm / HCP_fusion_npy_32g_25mm
    LABELS_FOLDER = "bundle_masks"  # bundle_masks / bundle_masks_dm
    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    BATCH_SIZE = 46  # Lasagne: 56  # Lasagne combined: 42   #Pytorch: 46
    LEARNING_RATE = 0.002
    UNET_NR_FILT = 64
    LOAD_WEIGHTS = False
    WEIGHTS_PATH = ""
    TYPE = "single_direction"       # single_direction / combined
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []
    TRAIN = True
    TEST = True
    SEGMENT = False
    GET_PROBS = False

    PREDICT_IMG = False
    PREDICT_IMG_OUTPUT = None
    OUTPUT_MULTIPLE_FILES = False
    TRACTSEG_DIR = "tractseg_output"
    KEEP_INTERMEDIATE_FILES = False
    CSD_RESOLUTION = "LOW"  # HIGH / LOW
    SKIP_PEAK_EXTRACTION = False

    #Rarly changed:
    LABELS_TYPE = np.int16  # Binary: np.int16, Regression: np.float32
    THRESHOLD = 0.5         # Binary: 0.5, Regression: 0.01 ?
    TEST_TIME_DAUG = False
    SLICE_DIRECTION = "x"   #no effect at the moment     # x, y, z  (combined needs z)
    USE_VISLOGGER = False
    INFO = "74 BNew, DMNifti, newSplit, 90gAnd270g, NormBeforeDAug, Fusion: 32gAnd270g"
    SAVE_WEIGHTS = True
    NR_OF_CLASSES = len(ExpUtils.get_bundle_names())
    SEG_INPUT = "Peaks"     # Gradients/ Peaks
    NR_SLICES = 1           # adapt manually: NR_OF_GRADIENTS in UNet.py and get_batch... in train() and in get_seg_prediction()
    PRINT_FREQ = 20
    NORMALIZE_DATA = True
    BEST_EPOCH = 0
    INPUT_DIM = (144, 144)
    VERBOSE = True

parser = argparse.ArgumentParser(description="Segment white matter bundles in a Diffusion MRI image.",
                                    epilog="Written by Jakob Wasserthal. Please reference TODO")
parser.add_argument("-i", metavar="filename", dest="input", help="Diffusion Input image (Nifti image)", required=True)
#https://stackoverflow.com/questions/20048048/argparse-default-option-based-on-another-option
parser.add_argument("-o", metavar="directory", dest="output", help="Output directory")
parser.add_argument("--output_multiple_files", action="store_true", help="Create extra output file for each bundle", default=False)
parser.add_argument("--bvals", metavar="filename", help="bvals file. Default is 'Diffusion.bvals'")  #todo: change default
parser.add_argument("--bvecs", metavar="filename", help="bvecs file. Default is 'Diffusion.bvecs'")
parser.add_argument("--verbose", action="store_true", help="Show more intermediate output", default=False)
parser.add_argument("--skip_peak_extraction", action="store_true", help="Do not calculate input peaks. You have to provide them yourself then.", default=False)
parser.add_argument("--keep_intermediate_files", action="store_true", help="Do not remove intermediate files like CSD output and peaks", default=False)
parser.add_argument('--version', action='version', version='TractSeg 0.5')
#todo: optionally supply brain mask (must have same dimensions as dwi)
args = parser.parse_args()

# args.input = "/mnt/jakob/E130-Personal/Wasserthal/data/SoftSigns/subject01/test/Diffusion.nii.gz"

HP.PREDICT_IMG = args.input is not None
if args.output:
    HP.PREDICT_IMG_OUTPUT = join(args.output, HP.TRACTSEG_DIR)
elif HP.PREDICT_IMG:
    HP.PREDICT_IMG_OUTPUT = join(os.path.dirname(args.input), HP.TRACTSEG_DIR)
HP.OUTPUT_MULTIPLE_FILES = args.output_multiple_files
HP.VERBOSE = args.verbose
HP.KEEP_INTERMEDIATE_FILES = args.keep_intermediate_files
HP.TRAIN = False
HP.TEST = False
HP.SEGMENT = False
HP.GET_PROBS = False
HP.LOAD_WEIGHTS = True
HP.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, "pretrained_weights.npz")

if HP.VERBOSE:
    print("Hyperparameters:")
    ExpUtils.print_HPs(HP)

print("Segmenting bundles...")

Utils.download_pretrained_weights()
bvals, bvecs = ExpUtils.get_bvals_bvecs_path(args)
ExpUtils.make_dir(HP.PREDICT_IMG_OUTPUT)

if not HP.SKIP_PEAK_EXTRACTION:
    Mrtrix.create_brain_mask(args.input, HP.PREDICT_IMG_OUTPUT)
    Mrtrix.create_fods(args.input, HP.PREDICT_IMG_OUTPUT, bvals, bvecs, HP.CSD_RESOLUTION)

start_time = time.time()
data_img = nib.load(join(HP.PREDICT_IMG_OUTPUT, "peaks.nii.gz"))
data, transformation = DatasetUtils.pad_and_scale_img_to_square_img(data_img.get_data(), target_size=144)

ModelClass = getattr(importlib.import_module("tractseg.models." + HP.MODEL), HP.MODEL)
model = ModelClass(HP)
seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False)
seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=False)

seg = DatasetUtils.cut_and_scale_img_back_to_original_img(seg, transformation)
ExpUtils.print_verbose(HP, "Took {}s".format(round(time.time() - start_time, 2)))

if HP.OUTPUT_MULTIPLE_FILES:
    ImgUtils.save_multilabel_img_as_multiple_files(seg, data_img.get_affine(), HP.PREDICT_IMG_OUTPUT)  # Save as several files
else:
    img = nib.Nifti1Image(seg, data_img.get_affine())
    nib.save(img, join(HP.PREDICT_IMG_OUTPUT, "bundle_segmentations.nii.gz"))

Mrtrix.clean_up(HP)