import sys, os, inspect
from os.path import join
from os.path import expanduser
import getpass
import socket

def get_config_file():
    '''
    Read variables in ~/.tractseg
    '''
    with open(join(expanduser("~"), ".tractseg")) as f:
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


