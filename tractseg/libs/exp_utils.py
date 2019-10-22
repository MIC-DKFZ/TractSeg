
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re
import ast
import sys
from os.path import join
from pprint import pprint

import numpy as np

from tractseg.libs.system_config import SystemConfig as C


def create_experiment_folder(experiment_name, multi_parent_path, train):
    """
    Create a new experiment folder. If it already exists, create new one with increasing number at the end.
    If not training model (only predicting): Use existing folder
    """
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
                tailing_numbers = re.findall('x([0-9]+)$', experiment_name)  # find tailing numbers that start with a x
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


def get_brain_mask_path(predict_img_output, brain_mask, input):
    if brain_mask:
        return brain_mask

    brain_mask_path = join(predict_img_output, "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        return brain_mask_path

    brain_mask_path = join(os.path.dirname(input), "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        print("Loading brain mask from: {}".format(brain_mask_path))
        return brain_mask_path

    return None


def add_background_class(data):
    """
    Calculate BG class (where no other class is 1) and add it at idx=0 to array.

    Args:
        data: 3D array with bundle masks (nr_bundles, x,y,z)

    Returns:
        (x,y,z,nr_bundles+1)
    """
    s = data[0].shape
    mask_ml = np.zeros((s[0], s[1], s[2], len(data) + 1))
    background = np.ones((s[0], s[1], s[2]))  # everything that contains no bundle

    for idx in range(len(data)):
        mask = data[idx]
        mask_ml[:, :, :, idx + 1] = mask
        background[mask == 1] = 0  # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml


def print_and_save(exp_path, text, only_log=False):
    if not only_log:
        print(text)
    try:
        with open(join(exp_path, "Log.txt"), "a") as f:  # a for append
            f.write(text)
            f.write("\n")
    except IOError:
        print("WARNING: Could not write to Log.txt file")


def print_verbose(verbose, text):
    if verbose:
        print(text)


def get_correct_labels_type(Config):
    if Config.LABELS_TYPE == "int":
        Config.LABELS_TYPE = np.int16
    elif Config.LABELS_TYPE == "float":
        Config.LABELS_TYPE = np.float32
    else:
        raise ValueError("Config.LABELS_TYPE not recognized")
    return Config


def get_manual_exp_name_peaks(manual_exp_name, part):
    """
    If want to use manual experiment name for peak regression, replace part nr by X:
    e.g. PeaksPartX_HR_DAug_fold2
    -> will find correct part then automatically
    """
    if "PeaksPartX" in manual_exp_name:
        manual_exp_name_parts = manual_exp_name.split("X")
        return manual_exp_name_parts[0] + part[-1] + manual_exp_name_parts[1]
    else:
        return manual_exp_name


def load_config_from_txt(path):

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config_str = open(path, "r").read()
    clean_str = ""
    for line in config_str.splitlines():
        if not line.startswith("Average Epoch time:"):
            clean_str += line
    config_dict = ast.literal_eval(clean_str)
    config_obj = Struct(**config_dict)
    return config_obj
