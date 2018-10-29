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

import os
from os.path import join
from os.path import expanduser


def get_config_name(input_type, output_type, dropout_sampling=False):
    if input_type == "peaks":
        if output_type == "tract_segmentation" and dropout_sampling:
            config = "TractSeg_12g90g270g_125mm_DS_DAugAll_Dropout"
            # config = "TractSeg_12g90g270g_125mm_DAugAll_Dropout"
        elif output_type == "tract_segmentation":
            config = "TractSeg_12g90g270g_125mm_DS_DAugAll"
            # config = "TractSeg_T1_12g90g270g_125mm_DAugAll"
            # config = "TractSeg72_888"
        elif output_type == "endings_segmentation":
            config = "EndingsSeg_12g90g270g_125mm_DS_DAugAll"
        elif output_type == "TOM":
            config = "Peaks_12g90g270g_125mm_DS_DAugAll"
        elif output_type == "dm_regression":
            config = "DmReg_12g90g270g_125mm_DAugAll"
    elif input_type == "T1":
        if output_type == "tract_segmentation":
            config = "TractSeg_T1_125mm_DAugAll"
        elif output_type == "endings_segmentation":
            config = "EndingsSeg_12g90g270g_125mm_DAugAll"
        elif output_type == "TOM":
            config = "Peaks20_12g90g270g_125mm"
        elif output_type == "dm_regression":
            raise ValueError("For dm_regression no pretrained model available for T1")
    else:
        raise ValueError("input_type not recognized")
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

    if "working_dir" in paths:                                # check if config file
        HOME = paths["working_dir"]
    else:                                                       # fallback
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

    if os.environ.get("TRACTSEG_DATA_DIR") is not None:      # check if environment variable
        DATA_PATH = os.environ.get("TRACTSEG_DATA_DIR")
    else:
        DATA_PATH = HOME

    if NETWORK_DRIVE is not None:
        EXP_PATH = join(NETWORK_DRIVE, "hcp_exp_nodes")
    else:
        EXP_PATH = join(HOME, "hcp_exp")

