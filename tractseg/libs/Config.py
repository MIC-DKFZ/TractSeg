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

import sys, os, inspect
from os.path import join
from os.path import expanduser
import getpass
import socket

def get_config_file():
    '''
    Read variables in ~/.tractseg
    '''
    with open(join(expanduser("~"), ".tractseg", "config.txt")) as f:
        lines = f.readlines()
    paths = {l.strip().split("=")[0]:l.strip().split("=")[1] for l in lines}
    return paths

class Config:
    paths = get_config_file()

    if "working_dir" in paths:
        HOME = paths["working_dir"]
    else:
        HOME = join(expanduser("~/TractSeg"))

    if "network_dir" in paths:
        NETWORK_DRIVE = paths["network_dir"]
    else:
        NETWORK_DRIVE = None

    if NETWORK_DRIVE is not None:
        EXP_PATH = join(NETWORK_DRIVE, "hcp_exp_nodes")
    else:
        EXP_PATH = join(HOME, "hcp_exp")

    TRACT_SEG_HOME = os.path.join(os.path.expanduser('~'), '.tractseg')


