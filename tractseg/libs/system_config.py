
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from os.path import join
from os.path import expanduser


def get_config_name(input_type, output_type, dropout_sampling=False, tract_definition="TractQuerier+"):
    if tract_definition == "TractQuerier+":
        if input_type == "peaks":
            if output_type == "tract_segmentation" and dropout_sampling:
                config = "TractSeg_PeakRot4"
            elif output_type == "tract_segmentation":
                config = "TractSeg_PeakRot4"
                # config = "TractSeg_T1_12g90g270g_125mm_DAugAll"
            elif output_type == "endings_segmentation":
                config = "EndingsSeg_PeakRot4"
            elif output_type == "TOM":
                config = "Peaks_AngL"
            elif output_type == "dm_regression":
                config = "DmReg"
        else:  # T1
            if output_type == "tract_segmentation":
                config = "TractSeg_T1_125mm_DAugAll"
            elif output_type == "endings_segmentation":
                config = "EndingsSeg_12g90g270g_125mm_DAugAll"
            elif output_type == "TOM":
                print("ERROR: For TOM no pretrained model available for T1")
                sys.exit()
            elif output_type == "dm_regression":
                print("ERROR: For dm_regression no pretrained model available for T1")
                sys.exit()
    else:  # "xtract"
        if input_type == "peaks":
            if output_type == "tract_segmentation" and dropout_sampling:
                config = "TractSeg_All_xtract_PeakRot4"
            elif output_type == "tract_segmentation":
                config = "TractSeg_All_xtract_PeakRot4"
            elif output_type == "endings_segmentation":
                print("ERROR: tract_definition xtract in combination with output_type endings_segmentation "
                      "not supported.")
                sys.exit()
            elif output_type == "TOM":
                print("ERROR: tract_definition xtract in combination with output_type TOM not supported.")
                sys.exit()
            elif output_type == "dm_regression":
                config = "DmReg_All_xtract_PeakRot4"
        else:  # T1
            print("ERROR: xtract in combination with input_type T1 not supported.")
            sys.exit()

    return config


def get_config_file():
    '''
    Read variables in ~/.tractseg
    '''
    path = join(expanduser("~"), ".tractseg", "config.txt")
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
        paths = {l.strip().split("=")[0]:l.strip().split("=")[1] for l in lines}
        return paths
    else:
        return {}


class SystemConfig:
    TRACT_SEG_HOME = os.path.join(os.path.expanduser('~'), '.tractseg')

    paths = get_config_file()

    if "working_dir" in paths:  # check if config file
        HOME = paths["working_dir"]
    else:  # fallback
        HOME = join(expanduser("~/TractSeg"))

    if "network_dir" in paths:
        NETWORK_DRIVE = paths["network_dir"]
    else:
        NETWORK_DRIVE = None

    if os.environ.get("TRACTSEG_WEIGHTS_DIR") is not None:
        WEIGHTS_DIR = os.environ.get("TRACTSEG_WEIGHTS_DIR")
    elif "weights_dir" in paths:
        WEIGHTS_DIR = paths["weights_dir"]
    else:
        WEIGHTS_DIR = TRACT_SEG_HOME

    if os.environ.get("TRACTSEG_DATA_DIR") is not None:  # check if environment variable
        DATA_PATH = os.environ.get("TRACTSEG_DATA_DIR")
    else:
        DATA_PATH = HOME

    if NETWORK_DRIVE is not None:
        EXP_PATH = join(NETWORK_DRIVE, "hcp_exp_nodes")
    else:
        EXP_PATH = join(HOME, "hcp_exp")

