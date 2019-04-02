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

import glob
import os
import re
from os.path import join
from pprint import pprint
import sys
import numpy as np

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs.subjects import get_all_subjects
from tractseg.libs import utils


def create_experiment_folder(experiment_name, multi_parent_path, train):
    '''
    Create a new experiment folder. If it already exist, create new one with increasing number at the end.

    If not training model (only predicting): Use existing folder
    '''

    if multi_parent_path != "":
        dir = join(multi_parent_path, experiment_name)
    else:
        dir = join(C.EXP_PATH, experiment_name)

    if not train:
        if os.path.exists(dir):
            return dir
        else:
            sys.exit('Testing target directory does not exist!')
    else:

        for i in range(40):
            if os.path.exists(dir):
                # tailing_numbers = re.findall('x.*?([0-9]+)$', experiment_name) #not correct
                tailing_numbers = re.findall('x([0-9]+)$', experiment_name) #find tailing numbers that start with a x
                if len(tailing_numbers) > 0:
                    num = int(tailing_numbers[0])
                    if num < 10:
                        experiment_name = experiment_name[:-1] + str(num+1)
                    else:
                        experiment_name = experiment_name[:-2] + str(num+1)
                else:
                    experiment_name += "_x2"

                if multi_parent_path != "":
                    dir = join(multi_parent_path, experiment_name)
                else:
                    dir = join(C.EXP_PATH, experiment_name)
            else:
                os.makedirs(dir)
                break
        return dir


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_Configs(Config):
    # dict = copy.deepcopy(Config.__dict__)
    dict = {attr: getattr(Config, attr) for attr in dir(Config)
            if not callable(getattr(Config, attr)) and not attr.startswith("__")}
    dict.pop("TRAIN_SUBJECTS", None)
    dict.pop("TEST_SUBJECTS", None)
    dict.pop("VALIDATE_SUBJECTS", None)
    pprint(dict)


def get_best_weights_path(exp_path, load_weights):
    if load_weights:
        return glob.glob(exp_path + "/best_weights_ep*.npz")[0]
    else:
        return ""


def get_bvals_bvecs_path(args):
    input_file_without_ending = os.path.basename(args.input).split(".")[0]
    if args.bvals:
        bvals = args.bvals
    else:
        bvals = join(os.path.dirname(args.input), input_file_without_ending + ".bvals")
    if args.bvecs:
        bvecs = args.bvecs
    else:
        bvecs = join(os.path.dirname(args.input), input_file_without_ending + ".bvecs")
    return bvals, bvecs


def get_brain_mask_path(Config, args):
    if args.brain_mask:
        return args.brain_mask

    brain_mask_path = join(Config.PREDICT_IMG_OUTPUT, "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        return brain_mask_path

    brain_mask_path = join(os.path.dirname(args.input), "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        print("Loading brain mask from: {}".format(brain_mask_path))
        return brain_mask_path

    # raise ValueError("no brainmask available")
    return None


def get_bundle_names(CLASSES):

    if CLASSES == "All":
        # 72 Tracts
        bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                   'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                   'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    elif CLASSES == "All_Part1":
        # 18 Tracts
        bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right']
    elif CLASSES == "All_Part2":
        # 18 Tracts
        bundles = ['FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right',
                   'ILF_left', 'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left',
                   'SCP_right', 'SLF_I_left']
    elif CLASSES == "All_Part3":
        # 18 Tracts
        bundles = ['SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right', 'STR_left',
                   'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right']
    elif CLASSES == "All_Part4":
        # 18 Tracts
        bundles = ['T_PAR_left', 'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right',
                   'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right',
                   'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    elif CLASSES == "11":
        # 11 Major tracts
        bundles = ["CST_left", "CST_right", "IFO_left", "IFO_right", "CA", "CG_left", "CG_right",
                   "FX_left", "FX_right", "UF_left", "UF_right"]

    elif CLASSES == "20":
        # 20 Major tracts
        bundles = ["AF_left", "AF_right", "CA", "CST_left", "CST_right", "CG_left", "CG_right",
                   "ICP_left", "ICP_right", "MCP", "SCP_left", "SCP_right", "ILF_left", "ILF_right",
                   "IFO_left", "IFO_right", "OR_left", "OR_right", "UF_left", "UF_right"]

    elif CLASSES == "20_endpoints_combined":
        # endpoints for "20"; beginnings and endings combined
        bundles = ["AF_left", "AF_right", "CA", "CST_left", "CST_right", "CG_left", "CG_right",
                   "ICP_left", "ICP_right", "MCP", "SCP_left", "SCP_right", "ILF_left", "ILF_right",
                   "IFO_left", "IFO_right", "OR_left", "OR_right", "UF_left", "UF_right"]

    elif CLASSES == "20_endpoints":
        #endpoints for "20"
        bundles = ['AF_left_b', 'AF_left_e', 'AF_right_b', 'AF_right_e', 'CA_b', 'CA_e',
                     'CST_left_b', 'CST_left_e', 'CST_right_b', 'CST_right_e', 'CG_left_b',
                     'CG_left_e', 'CG_right_b', 'CG_right_e', 'ICP_left_b', 'ICP_left_e',
                     'ICP_right_b', 'ICP_right_e', 'MCP_b', 'MCP_e', 'SCP_left_b', 'SCP_left_e',
                     'SCP_right_b', 'SCP_right_e', 'ILF_left_b', 'ILF_left_e', 'ILF_right_b',
                     'ILF_right_e', 'IFO_left_b', 'IFO_left_e', 'IFO_right_b', 'IFO_right_e',
                     'OR_left_b', 'OR_left_e', 'OR_right_b', 'OR_right_e', 'UF_left_b', 'UF_left_e',
                     'UF_right_b', 'UF_right_e'] #40

    elif CLASSES == "20_bundles_endpoints":
        #endpoints for "20"
        bundles = ['AF_left', 'AF_left_b', 'AF_left_e', 'AF_right', 'AF_right_b', 'AF_right_e',
                   'CA', 'CA_b', 'CA_e', 'CST_left', 'CST_left_b', 'CST_left_e', 'CST_right', 'CST_right_b', 'CST_right_e',
                   'CG_left', 'CG_left_b', 'CG_left_e', 'CG_right', 'CG_right_b', 'CG_right_e',
                   'ICP_left', 'ICP_left_b', 'ICP_left_e', 'ICP_right', 'ICP_right_b', 'ICP_right_e',
                   'MCP', 'MCP_b', 'MCP_e', 'SCP_left', 'SCP_left_b', 'SCP_left_e',
                   'SCP_right', 'SCP_right_b', 'SCP_right_e', 'ILF_left', 'ILF_left_b', 'ILF_left_e',
                   'ILF_right', 'ILF_right_b', 'ILF_right_e', 'IFO_left', 'IFO_left_b', 'IFO_left_e',
                   'IFO_right', 'IFO_right_b', 'IFO_right_e',
                   'OR_left', 'OR_left_b', 'OR_left_e', 'OR_right', 'OR_right_b', 'OR_right_e',
                   'UF_left', 'UF_left_b', 'UF_left_e', 'UF_right', 'UF_right_b', 'UF_right_e'] #60

    elif CLASSES == "All_endpoints":
        #endpoints for "All"
        bundles = ['AF_left_b', 'AF_left_e', 'AF_right_b', 'AF_right_e', 'ATR_left_b', 'ATR_left_e', 'ATR_right_b',
         'ATR_right_e', 'CA_b', 'CA_e', 'CC_1_b', 'CC_1_e', 'CC_2_b', 'CC_2_e', 'CC_3_b', 'CC_3_e', 'CC_4_b',
         'CC_4_e', 'CC_5_b', 'CC_5_e', 'CC_6_b', 'CC_6_e', 'CC_7_b', 'CC_7_e', 'CG_left_b', 'CG_left_e',
         'CG_right_b', 'CG_right_e', 'CST_left_b', 'CST_left_e', 'CST_right_b', 'CST_right_e', 'MLF_left_b',
         'MLF_left_e', 'MLF_right_b', 'MLF_right_e', 'FPT_left_b', 'FPT_left_e', 'FPT_right_b', 'FPT_right_e',
         'FX_left_b', 'FX_left_e', 'FX_right_b', 'FX_right_e', 'ICP_left_b', 'ICP_left_e', 'ICP_right_b',
         'ICP_right_e', 'IFO_left_b', 'IFO_left_e', 'IFO_right_b', 'IFO_right_e', 'ILF_left_b', 'ILF_left_e',
         'ILF_right_b', 'ILF_right_e', 'MCP_b', 'MCP_e', 'OR_left_b', 'OR_left_e', 'OR_right_b', 'OR_right_e',
         'POPT_left_b', 'POPT_left_e', 'POPT_right_b', 'POPT_right_e', 'SCP_left_b', 'SCP_left_e', 'SCP_right_b',
         'SCP_right_e', 'SLF_I_left_b', 'SLF_I_left_e', 'SLF_I_right_b', 'SLF_I_right_e', 'SLF_II_left_b',
         'SLF_II_left_e', 'SLF_II_right_b', 'SLF_II_right_e', 'SLF_III_left_b', 'SLF_III_left_e', 'SLF_III_right_b',
         'SLF_III_right_e', 'STR_left_b', 'STR_left_e', 'STR_right_b', 'STR_right_e', 'UF_left_b', 'UF_left_e',
         'UF_right_b', 'UF_right_e', 'CC_b', 'CC_e', 'T_PREF_left_b', 'T_PREF_left_e', 'T_PREF_right_b',
         'T_PREF_right_e', 'T_PREM_left_b', 'T_PREM_left_e', 'T_PREM_right_b', 'T_PREM_right_e', 'T_PREC_left_b',
         'T_PREC_left_e', 'T_PREC_right_b', 'T_PREC_right_e', 'T_POSTC_left_b', 'T_POSTC_left_e', 'T_POSTC_right_b',
         'T_POSTC_right_e', 'T_PAR_left_b', 'T_PAR_left_e', 'T_PAR_right_b', 'T_PAR_right_e', 'T_OCC_left_b',
         'T_OCC_left_e', 'T_OCC_right_b', 'T_OCC_right_e', 'ST_FO_left_b', 'ST_FO_left_e', 'ST_FO_right_b',
         'ST_FO_right_e', 'ST_PREF_left_b', 'ST_PREF_left_e', 'ST_PREF_right_b', 'ST_PREF_right_e',
         'ST_PREM_left_b', 'ST_PREM_left_e', 'ST_PREM_right_b', 'ST_PREM_right_e', 'ST_PREC_left_b',
         'ST_PREC_left_e', 'ST_PREC_right_b', 'ST_PREC_right_e', 'ST_POSTC_left_b', 'ST_POSTC_left_e',
         'ST_POSTC_right_b', 'ST_POSTC_right_e', 'ST_PAR_left_b', 'ST_PAR_left_e', 'ST_PAR_right_b',
         'ST_PAR_right_e', 'ST_OCC_left_b', 'ST_OCC_left_e', 'ST_OCC_right_b', 'ST_OCC_right_e'] #144

    elif CLASSES == "AutoPTX":
        bundles = ["af_l", "af_r", "ar_l", "ar_r", "atr_l", "atr_r", "cbd_l", "cbd_r", "cbp_l", "cbp_r", "cbt_l",
                   "cbt_r", "cing_l", "cing_r", "cst_l", "cst_r", "fa_l", "fa_r", "fma", "fmi", "fx_l", "fx_r",
                   "ifo_l", "ifo_r", "ilf_l", "ilf_r", "mcp", "mdlf_l", "mdlf_r", "MG_ac", "MG_unc_l", "MG_unc_r",
                   "or_l", "or_r", "slf1_l_kattest2_symm", "slf1_r_kattest2_symm", "slf2_l_kattest2_symm",
                   "slf2_r_kattest2_symm", "slf3_l_kattest2_symm", "slf3_r_kattest2_symm", "str_l", "str_r",
                   "unc_l", "unc_r"]

    elif CLASSES == "AutoPTX_42":
        bundles = ["af_l", "af_r", "ar_l", "ar_r", "atr_l", "atr_r", "cbd_l", "cbd_r", "cbp_l", "cbp_r", "cbt_l",
                   "cbt_r", "cing_l", "cing_r", "cst_l", "cst_r", "fa_l", "fa_r", "fma", "fmi", "fx_l", "fx_r",
                   "ifo_l", "ifo_r", "ilf_l", "ilf_r", "mcp", "mdlf_l", "mdlf_r", "MG_ac", "MG_unc_l", "MG_unc_r",
                   "or_l", "or_r", "slf1_l_kattest2_symm", "slf1_r_kattest2_symm", "slf2_l_kattest2_symm",
                   "slf2_r_kattest2_symm", "slf3_l_kattest2_symm", "slf3_r_kattest2_symm", "str_l", "str_r"]

    elif CLASSES == "AutoPTX_CST":
        bundles = ["cst_l", "cst_r"]

    elif CLASSES == "test":
        # Only use subset of classes for unit testing because of runtime
        bundles = ["CST_right", "CA", "UF_left"]

    else:
        #1 tract
        # bundles = ["CST_right"]
        bundles = [CLASSES]

    return ["BG"] + bundles    #Add Background label (is always beginning of list)


def get_ACT_noACT_bundle_names():
    ACT = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7',
           'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left', 'FPT_right', 'FX_left',
           'FX_right', 'ICP_left', 'ICP_right', 'ILF_left', 'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left',
           'POPT_right', 'SCP_left', 'SCP_right', 'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right',
           'SLF_III_left', 'SLF_III_right', 'STR_left', 'STR_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
           'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left', 'T_PAR_right',
           'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left',
           'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left',
           'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    noACT = ["CA", "IFO_left", "IFO_right", "UF_left", "UF_right"]

    return ACT, noACT


def get_labels_filename(Config):

    if Config.CLASSES == "All" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_808080"

    elif Config.CLASSES == "11" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_11"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_11_808080"

    elif Config.CLASSES == "20" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_20"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_20_808080"

    elif Config.CLASSES == "All_Part1" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_Part1"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_Part1_808080"

    elif Config.CLASSES == "All_Part2" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_Part2"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_Part2_808080"

    elif Config.CLASSES == "All_Part3" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_Part3"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_Part3_808080"

    elif Config.CLASSES == "All_Part4" and Config.EXPERIMENT_TYPE == "peak_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_peaks_Part4"
        else:
            Config.LABELS_FILENAME = "bundle_peaks_Part5_808080"

    elif Config.CLASSES == "All_endpoints" and Config.EXPERIMENT_TYPE == "endings_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "endpoints_72_ordered"
        else:
            Config.LABELS_FILENAME = "endpoints_72_ordered"

    elif Config.CLASSES == "20_endpoints" and Config.EXPERIMENT_TYPE == "endings_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "endpoints_20_ordered"
        else:
            Config.LABELS_FILENAME = "endpoints_20_ordered"

    elif Config.CLASSES == "20_endpoints_combined" and Config.EXPERIMENT_TYPE == "endings_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "endpoints_20_combined"
        else:
            Config.LABELS_FILENAME = "endpoints_20_combined"

    elif Config.CLASSES == "20_bundles_endpoints" and Config.EXPERIMENT_TYPE == "endings_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_endpoints_20"
        else:
            Config.LABELS_FILENAME = "bundle_endpoints_20"

    elif Config.CLASSES == "All" and Config.EXPERIMENT_TYPE == "tract_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_72"
        elif Config.RESOLUTION == "2mm" and Config.DATASET == "Schizo":
            Config.LABELS_FILENAME = "bundle_masks_72"
        else:
            Config.LABELS_FILENAME = "bundle_masks_72_808080"

    elif (Config.CLASSES == "AutoPTX" or Config.CLASSES == "AutoPTX_42") and \
            Config.EXPERIMENT_TYPE == "tract_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_thr001"
        elif Config.RESOLUTION == "2mm" and Config.DATASET == "Schizo":
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_thr001"
        else:
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_thr001_808080"

    elif Config.CLASSES == "AutoPTX_CST" and Config.EXPERIMENT_TYPE == "tract_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_thr001_CST"
        elif Config.RESOLUTION == "2mm" and Config.DATASET == "Schizo":
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_thr001_CST"
        else:
            Config.LABELS_FILENAME = "NOT_AVAILABLE"

    elif Config.CLASSES == "20" and Config.EXPERIMENT_TYPE == "tract_segmentation":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_20"
        else:
            Config.LABELS_FILENAME = "bundle_masks_20_808080"

    elif Config.CLASSES == "All" and Config.EXPERIMENT_TYPE == "dm_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_dm"
        else:
            Config.LABELS_FILENAME = "NOT_AVAILABLE"

    elif (Config.CLASSES == "AutoPTX" or Config.CLASSES == "AutoPTX_42") and \
            Config.EXPERIMENT_TYPE == "dm_regression":
        if Config.RESOLUTION == "1.25mm":
            Config.LABELS_FILENAME = "bundle_masks_autoPTX_dm"
        else:
            Config.LABELS_FILENAME = "NOT_AVAILABLE"

    else:
        Config.LABELS_FILENAME = "bundle_peaks/" + Config.CLASSES

    return Config



def add_background_class(data):
    '''
    List of 3D Array with bundle masks; shape: (nr_bundles, x,y,z)
    Calculate BG class (where no other class is 1) and add it at idx=0 to array.
    Returns with nr_bundles shape in last dim: (x,y,z,nr_bundles)
    :param data:
    :return:
    '''
    s = data[0].shape
    mask_ml = np.zeros((s[0], s[1], s[2], len(data) + 1))
    background = np.ones((s[0], s[1], s[2]))  # everything that contains no bundle

    for idx in range(len(data)):
        mask = data[idx]
        mask_ml[:, :, :, idx + 1] = mask
        background[mask == 1] = 0  # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml


def get_cv_fold(fold, dataset="HCP"):
    '''
    Brauche train-validate-test wegen Best-model selection und wegen training von combined net
    :return:
    '''

    if dataset == "HCP_all":
        subjects = get_all_subjects(dataset)
        cut_point = int(len(subjects) * 0.9)
        #todo important: change
        # return subjects[:cut_point], subjects[cut_point:], []
        return subjects[:cut_point], subjects[cut_point:], ["599469", "599469"]
    else:
        #For CV
        if fold == 0:
            train, validate, test = [0, 1, 2], [3], [4]
            # train, validate, test = [0, 1, 2, 3, 4], [3], [4]
        elif fold == 1:
            train, validate, test = [1, 2, 3], [4], [0]
        elif fold == 2:
            train, validate, test = [2, 3, 4], [0], [1]
        elif fold == 3:
            train, validate, test = [3, 4, 0], [1], [2]
        elif fold == 4:
            train, validate, test = [4, 0, 1], [2], [3]

        subjects = get_all_subjects(dataset)

        if dataset.startswith("HCP"):
            # subjects = list(Utils.chunks(subjects[:100], 10))   #10 folds
            subjects = list(utils.chunks(subjects, 21))   #5 folds a 21 subjects
            # => 5 fold CV ok (score only 1%-point worse than 10 folds (80 vs 60 train subjects) (10 Fold CV impractical!)
        elif dataset.startswith("Schizo"):
            # 410 subjects
            subjects = list(utils.chunks(subjects, 82))  # 5 folds a 82 subjects
        else:
            raise ValueError("Invalid dataset name")

        subjects = np.array(subjects)
        return list(subjects[train].flatten()), list(subjects[validate].flatten()), list(subjects[test].flatten())


def print_and_save(Config, text, only_log=False):
    if not only_log:
        print(text)
    try:
        with open(join(Config.EXP_PATH, "Log.txt"), "a") as f:  # a for append
            f.write(text)
            f.write("\n")
    except IOError:
        print("WARNING: Could not write to Log.txt file")


def print_verbose(Config, text):
    if Config.VERBOSE:
        print(text)


def get_correct_input_dim(Config):
    if Config.DIM == "2D":
        if Config.RESOLUTION == "1.25mm":
            input_dim = (144, 144)
        elif Config.RESOLUTION == "2mm":
            input_dim = (96, 96)
        elif Config.RESOLUTION == "2.5mm":
            input_dim = (80, 80)
    else:  # 3D
        if Config.RESOLUTION == "1.25mm":
            input_dim = (144, 144, 144)
        elif Config.RESOLUTION == "2mm":
            input_dim = (96, 96, 96)
        elif Config.RESOLUTION == "2.5mm":
            input_dim = (80, 80, 80)
    return input_dim


def get_manual_exp_name_peaks(manual_exp_name, part):
    """
    If want to use manual experiment name for peak regression, replace part nr by X:
    e.g. PeaksPartX_HR_DAug_fold2
    -> will find correct part then automatically

    Args:
        manual_exp_name:
        part:

    Returns:

    """
    manual_exp_name_parts = manual_exp_name.split("X")
    return manual_exp_name_parts[0] + part[-1] + manual_exp_name_parts[1]