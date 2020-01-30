
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tractseg.libs import img_utils
from tractseg.data.subjects import get_all_subjects
from tractseg.libs import utils


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

    elif CLASSES == "All_tractometry":
        # 50 Tracts
        bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left',
                   'FPT_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'T_PREM_left', 'T_PREM_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREM_left',
                   'ST_PREM_right']

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

    elif CLASSES == "AutoPTX_27":
        bundles = ["ar_l", "ar_r", "atr_l", "atr_r", "cgc_l", "cgc_r", "cgh_l", "cgh_r", "cst_l", "cst_r", "fma",
                   "fmi", "ifo_l", "ifo_r", "ilf_l", "ilf_r", "mcp", "ml_l", "ml_r", "ptr_l", "ptr_r", "slf_l",
                   "slf_r", "str_l", "str_r", "unc_l", "unc_r"]  # 27

    elif CLASSES == "xtract":
        bundles = ["ac", "af_l", "af_r", "ar_l", "ar_r", "atr_l", "atr_r", "cbd_l", "cbd_r", "cbp_l", "cbp_r",
                   "cbt_l", "cbt_r", "cst_l", "cst_r", "fa_l", "fa_r", "fma", "fmi", "fx_l", "fx_r", "ifo_l",
                   "ifo_r", "ilf_l", "ilf_r", "mcp", "mdlf_l", "mdlf_r", "or_l", "or_r", "slf1_l", "slf1_r",
                   "slf2_l", "slf2_r", "slf3_l", "slf3_r", "str_l", "str_r", "uf_l", "uf_r", "vof_l",
                   "vof_r"]  # 42

    elif CLASSES == "AutoPTX_CST":
        bundles = ["cst_l", "cst_r"]

    elif CLASSES == "test":
        # Only use subset of classes for unit testing because of runtime
        bundles = ["CST_right", "CA", "IFO_right"]

    elif CLASSES == "test_single":
        # Only use subset of classes for unit testing because of runtime
        bundles = ["CST_right"]

    else:
        #1 tract
        bundles = [CLASSES]

    return ["BG"] + bundles  # Add Background label (is always beginning of list)


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
    """
    Returns name of labels file (without file ending (.nii.gz automatically added)) depending on config settings.
    """
    if Config.LABELS_FILENAME != "":
        print("INFO: LABELS_FILENAME manually set")
        return Config

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
        # else:
        #     Config.LABELS_FILENAME = "bundle_masks_72_808080"
        else:
            Config.LABELS_FILENAME = "bundle_masks_72"

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


def get_dwi_affine(dataset, resolution):

    if dataset == "HCP" and resolution == "1.25mm":
        # shape (145,174,145)
        return np.array([[-1.25, 0.,  0.,   90.],
                         [0., 1.25,   0.,  -126.],
                         [0.,    0., 1.25, -72.],
                         [0.,    0.,  0.,   1.]])

    elif dataset == "HCP_32g" and resolution == "1.25mm":
        # shape (145,174,145)
        return np.array([[-1.25, 0.,  0.,   90.],
                         [0., 1.25,   0.,  -126.],
                         [0.,    0., 1.25, -72.],
                         [0.,    0.,  0.,   1.]])

    elif (dataset == "HCP_32g" or dataset == "HCP_2mm") and resolution == "2mm":
        # shape (90,108,90)
        return np.array([[-2., 0.,  0.,   90.],
                         [0.,  2.,  0.,  -126.],
                         [0.,  0.,  2.,  -72.],
                         [0.,  0.,  0.,   1.]])

    elif (dataset == "HCP" or dataset == "HCP_32g" or dataset == "HCP_2.5mm") and resolution == "2.5mm":
        # shape (73,87,73)
        return np.array([[-2.5, 0.,  0.,   90.],
                         [0.,  2.5,  0.,  -126.],
                         [0.,  0.,  2.5,  -72.],
                         [0.,  0.,  0.,    1.]])

    else:
        raise ValueError("No Affine defined for this dataset and resolution")


def get_cv_fold(fold, dataset="HCP"):
    if dataset == "HCP_all":
        subjects = get_all_subjects(dataset)
        cut_point = int(len(subjects) * 0.9)
        return subjects[:cut_point], subjects[cut_point:], ["599671", "599469"]
    elif dataset == "HCP_90g":
        subjects = get_all_subjects(dataset)
        cut_point = int(len(subjects) * 0.7)
        return subjects[:cut_point], subjects[cut_point:], ["599671", "599469"]
    elif dataset == "biobank_20k" or dataset == "biobank_10":
        subjects = get_all_subjects(dataset)
        cut_point = int(len(subjects) * 0.9)
        return subjects[:cut_point], subjects[cut_point:], ["1000013", "1000013"]
    else:
        if fold == 0:
            train, validate, test = [0, 1, 2], [3], [4]
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
            subjects = list(utils.chunks(subjects, 21))   #5 folds a 21 subjects
            # 5 fold CV ok (score only 1%-point worse than 10 folds (80 vs 60 train subjects) (10 Fold CV impractical!)
        elif dataset.startswith("Schizo"):
            # ~410 subjects
            subjects = list(utils.chunks(subjects, 82))  # 5 folds a 82 subjects
        else:
            raise ValueError("Invalid dataset name")

        subjects = np.array(subjects)
        return list(subjects[train].flatten()), list(subjects[validate].flatten()), list(subjects[test].flatten())


def scale_input_to_unet_shape(img4d, dataset, resolution="1.25mm"):
    """
    Scale input image to right isotropic resolution and pad/cut image to make it square to fit UNet input shape.
    This is not generic but optimised for some specific datasets.

    Args:
        img4d: (x, y, z, classes)
        dataset: HCP|HCP_32g|TRACED|Schizo
        resolution: 1.25mm|2mm|2.5mm

    Returns:
        img with dim 1mm: (144,144,144,none) or 2mm: (80,80,80,none) or 2.5mm: (80,80,80,none)
        (note: 2.5mm padded with more zeros to reach 80,80,80)
    """
    if resolution == "1.25mm":
        if dataset == "HCP":  # (145,174,145)
            # no resize needed
            return img4d[1:, 15:159, 1:]  # (144,144,144)
        elif dataset == "HCP_32g":  # (73,87,73)
            img4d = img_utils.resize_first_three_dims(img4d, zoom=2)  # (146,174,146,none)
            img4d = img4d[:-1,:,:-1]  # remove one voxel that came from upsampling   # (145,174,145)
            return img4d[1:, 15:159, 1:]  # (144,144,144)
        elif dataset == "TRACED":  # (78,93,75)
            raise ValueError("resolution '1.25mm' not supported for dataset 'TRACED'")
        elif dataset == "Schizo":  # (91,109,91)
            img4d = img_utils.resize_first_three_dims(img4d, zoom=1.60)  # (146,174,146)
            return img4d[1:145, 15:159, 1:145]                                # (144,144,144)

    elif resolution == "2mm":
        if dataset == "HCP":  # (145,174,145)
            img4d = img_utils.resize_first_three_dims(img4d, zoom=0.62)  # (90,108,90)
            return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
        elif dataset == "HCP_32g":  # (145,174,145)
            img4d = img_utils.resize_first_three_dims(img4d, zoom=0.62)  # (90,108,90)
            return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
        elif dataset == "HCP_2mm":  # (90,108,90)
            # no resize needed
            return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
        elif dataset == "TRACED":  # (78,93,75)
            raise ValueError("resolution '2mm' not supported for dataset 'TRACED'")
        elif dataset == "Schizo":  # (91,109,91)
            return img4d[:, 9:100, :]                                # (91,91,91)

    elif resolution == "2.5mm":
        if dataset == "HCP":  # (145,174,145)
            img4d = img_utils.resize_first_three_dims(img4d, zoom=0.5)  # (73,87,73,none)
            bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
            # make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
            bg = bg + img4d[0,0,0,:]
            bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
            return bg  # (80,80,80)
        elif dataset == "HCP_2.5mm":  # (73,87,73,none)
            # no resize needed
            bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
            # make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
            bg = bg + img4d[0,0,0,:]
            bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
            return bg  # (80,80,80)
        elif dataset == "HCP_32g":  # (73,87,73,none)
            bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
            # make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
            bg = bg + img4d[0, 0, 0, :]
            bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
            return bg  # (80,80,80)
        elif dataset == "TRACED":  # (78,93,75)
            # no resize needed
            bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
            bg = bg + img4d[0, 0, 0, :]  # make bg have same value as bg from original img
            bg[1:79, :, 3:78, :] = img4d[:, 7:87, :, :]
            return bg  # (80,80,80)


def scale_input_to_original_shape(img4d, dataset, resolution="1.25mm"):
    """
    Scale input image to original resolution and pad/cut image to make it original size.
    This is not generic but optimised for some specific datasets.

    Args:
        img4d:  (x, y, z, classes)
        dataset: HCP|HCP_32g|TRACED|Schizo
        resolution: 1.25mm|2mm|2.5mm

    Returns:
        (x_original, y_original, z_original, classes)
    """
    if resolution == "1.25mm":
        if dataset == "HCP":  # (144,144,144)
            # no resize needed
            return img_utils.pad_4d_image_left(img4d, np.array([1, 15, 1, 0]),
                                               [146, 174, 146, img4d.shape[3]],
                                               pad_value=0)[:-1, :, :-1, :]  # (145, 174, 145, none)
        elif dataset == "HCP_32g":  # (144,144,144)
            # no resize needed
            return img_utils.pad_4d_image_left(img4d, np.array([1, 15, 1, 0]),
                                               [146, 174, 146, img4d.shape[3]],
                                               pad_value=0)[:-1, :, :-1, :]  # (145, 174, 145, none)
        elif dataset == "TRACED":  # (78,93,75)
            raise ValueError("resolution '1.25mm' not supported for dataset 'TRACED'")
        elif dataset == "Schizo":  # (144,144,144)
            img4d = img_utils.pad_4d_image_left(img4d, np.array([1, 15, 1, 0]),
                                                [145, 174, 145, img4d.shape[3]], pad_value=0)  # (145, 174, 145, none)
            return img_utils.resize_first_three_dims(img4d, zoom=0.62)  # (91,109,91)

    elif resolution == "2mm":
        if dataset == "HCP":  # (80,80,80)
            return img_utils.pad_4d_image_left(img4d, np.array([5, 14, 5, 0]),
                                               [90, 108, 90, img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
        elif dataset == "HCP_32g":  # (80,80,80)
            return img_utils.pad_4d_image_left(img4d, np.array([5, 14, 5, 0]),
                                               [90, 108, 90, img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
        elif dataset == "HCP_2mm":  # (80,80,80)
            return img_utils.pad_4d_image_left(img4d, np.array([5, 14, 5, 0]),
                                               [90, 108, 90, img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
        elif dataset == "TRACED":  # (78,93,75)
            raise ValueError("resolution '2mm' not supported for dataset 'TRACED'")

    elif resolution == "2.5mm":
        if dataset == "HCP":  # (80,80,80)
            img4d = img_utils.pad_4d_image_left(img4d, np.array([0, 4, 0, 0]),
                                                [80, 87, 80, img4d.shape[3]], pad_value=0) # (80,87,80,none)
            return img4d[4:77,:,4:77, :] # (73, 87, 73, none)
        elif dataset == "HCP_2.5mm":  # (80,80,80)
            img4d = img_utils.pad_4d_image_left(img4d, np.array([0, 4, 0, 0]),
                                                [80, 87, 80, img4d.shape[3]], pad_value=0)  # (80,87,80,none)
            return img4d[4:77,:,4:77,:]  # (73, 87, 73, none)
        elif dataset == "HCP_32g":  # ((80,80,80)
            img4d = img_utils.pad_4d_image_left(img4d, np.array([0, 4, 0, 0]),
                                                [80, 87, 80, img4d.shape[3]], pad_value=0)  # (80,87,80,none)
            return img4d[4:77, :, 4:77, :]  # (73, 87, 73, none)
        elif dataset == "TRACED":  # (80,80,80)
            img4d = img_utils.pad_4d_image_left(img4d, np.array([0, 7, 0, 0]),
                                                [80, 93, 80, img4d.shape[3]], pad_value=0)  # (80,93,80,none)
            return img4d[1:79, :, 3:78, :]  # (78,93,75,none)


def get_optimal_orientation_for_bundle(bundle):
    """
    Get optimal orientation if want to plot the respective bundle.
    """
    bundles_orientation = {'AF_left': 'sagittal',
                           'AF_right': 'sagittal',
                           'ATR_left': 'sagittal',
                           'ATR_right': 'sagittal',
                           'CA': 'coronal',
                           'CC_1': 'axial',
                           'CC_2': 'axial',
                           'CC_3': 'coronal',
                           'CC_4': 'coronal',
                           'CC_5': 'coronal',
                           'CC_6': 'coronal',
                           'CC_7': 'axial',
                           'CG_left': 'sagittal',
                           'CG_right': 'sagittal',
                           'CST_left': 'coronal',
                           'CST_right': 'coronal',
                           'MLF_left': 'sagittal',
                           'MLF_right': 'sagittal',
                           'FPT_left': 'sagittal',
                           'FPT_right': 'sagittal',
                           'FX_left': 'sagittal',
                           'FX_right': 'sagittal',
                           'ICP_left': 'sagittal',
                           'ICP_right': 'sagittal',
                           'IFO_left': 'sagittal',
                           'IFO_right': 'sagittal',
                           'ILF_left': 'sagittal',
                           'ILF_right': 'sagittal',
                           'MCP': 'axial',
                           'OR_left': 'axial',
                           'OR_right': 'axial',
                           'POPT_left': 'sagittal',
                           'POPT_right': 'sagittal',
                           'SCP_left': 'sagittal',
                           'SCP_right': 'sagittal',
                           'SLF_I_left': 'sagittal',
                           'SLF_I_right': 'sagittal',
                           'SLF_II_left': 'sagittal',
                           'SLF_II_right': 'sagittal',
                           'SLF_III_left': 'sagittal',
                           'SLF_III_right': 'sagittal',
                           'STR_left': 'sagittal',
                           'STR_right': 'sagittal',
                           'UF_left': 'sagittal',
                           'UF_right': 'sagittal',
                           'CC': 'sagittal',
                           'T_PREF_left': 'sagittal',
                           'T_PREF_right': 'sagittal',
                           'T_PREM_left': 'sagittal',
                           'T_PREM_right': 'sagittal',
                           'T_PREC_left': 'sagittal',
                           'T_PREC_right': 'sagittal',
                           'T_POSTC_left': 'sagittal',
                           'T_POSTC_right': 'sagittal',
                           'T_PAR_left': 'sagittal',
                           'T_PAR_right': 'sagittal',
                           'T_OCC_left': 'sagittal',
                           'T_OCC_right': 'sagittal',
                           'ST_FO_left': 'sagittal',
                           'ST_FO_right': 'sagittal',
                           'ST_PREF_left': 'sagittal',
                           'ST_PREF_right': 'sagittal',
                           'ST_PREM_left': 'sagittal',
                           'ST_PREM_right': 'sagittal',
                           'ST_PREC_left': 'sagittal',
                           'ST_PREC_right': 'sagittal',
                           'ST_POSTC_left': 'sagittal',
                           'ST_POSTC_right': 'sagittal',
                           'ST_PAR_left': 'sagittal',
                           'ST_PAR_right': 'sagittal',
                           'ST_OCC_left': 'sagittal',
                           'ST_OCC_right': 'sagittal'}

    return bundles_orientation[bundle]