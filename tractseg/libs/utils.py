
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tractseg.libs.system_config import SystemConfig as C

try:
    from urllib.request import urlopen     # For Python 3.0 and later
except ImportError:
    from urllib2 import urlopen            # Fall back to Python 2's urllib2


def invert_x_and_y(affineMatrix):
    """
    Change sign of x and y transformation (rotation, scaling and transformation)

    IMPORTANT note: only done for diagonal elements (if we need rotation (not only scaling) we may also need
    to do it for non-diagonal elements) -> not done yet
    """
    newAffine = affineMatrix.copy()
    newAffine[0,0] = newAffine[0,0] * -1
    newAffine[1,1] = newAffine[1,1] * -1
    newAffine[0,3] = newAffine[0,3] * -1
    newAffine[1,3] = newAffine[1,3] * -1
    return newAffine


def normalize_mean0_std1(data):
    """
    Normalizes along all axis for mean=0 and stddev=1
    """
    out = np.array(data, dtype='float32', copy=True)

    mean = data.mean()  # mean over all axis / over flattened array
    out -= mean

    std = data.std()
    out /= std

    return out


def to_unit_length(vec):
    """
    Vector to unit length

    Args:
        vec: 3D vector (x, y, z)

    Returns:
            3D vector with len=1, but same direction as original vector
    """
    vec_length = np.sqrt(np.sum(np.square(vec)))
    return vec / vec_length  # divide elementwise


def to_unit_length_batch(vec):
    vec_length = np.sqrt(np.sum(np.square(vec), axis=1))
    return vec / vec_length[:, np.newaxis]  # divide elementwise (only along one axis)


def get_lr_decay(epoch_nr):
    """
    Calc what lr_decay is need to make lr be 1/10 of original lr after epoch_nr number of epochs
    """
    target_lr = 0.1  # should be reduced to 1/10 of original
    return target_lr ** (1 / float(epoch_nr))


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Last chunk can be smaller.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(l):
    return [item for sublist in l for item in sublist]


def mem_usage(print_usage=True):
    import psutil
    process = psutil.Process()
    gb = process.memory_info().rss / 1e9
    gb = round(gb, 3)
    if print_usage:
        print("PID {} using {} GB".format(os.getpid(), gb))
    return gb


def download_pretrained_weights(experiment_type, dropout_sampling=False,
                                part="Part1", tract_definition="TractQuerier+"):

    if experiment_type == "tract_segmentation" and tract_definition == "xtract":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_tract_segmentation_xtract_v0.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_tract_segmentation_xtract_v1.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3634539/files/best_weights_ep266.npz?download=1"

    elif experiment_type == "tract_segmentation" and tract_definition == "TractQuerier+":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_tract_segmentation_v2.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_tract_segmentation_v3.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3518348/files/best_weights_ep220.npz?download=1"

    elif experiment_type == "endings_segmentation":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_endings_segmentation_v3.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_endings_segmentation_v4.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3518331/files/best_weights_ep143.npz?download=1"

    elif experiment_type == "dm_regression" and tract_definition == "xtract":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_dm_regression_xtract_v0.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_dm_regression_xtract_v1.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3634549/files/best_weights_ep207.npz?download=1"

    elif experiment_type == "dm_regression" and tract_definition == "TractQuerier+":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_dm_regression_v1.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_dm_regression_v2.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3518346/files/best_weights_ep199.npz?download=1"

    elif experiment_type == "peak_regression" and part == "Part1":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part1_v1.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part1_v2.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3239216/files/best_weights_ep62.npz?download=1"

    elif experiment_type == "peak_regression" and part == "Part2":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part2_v1.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part2_v2.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3239220/files/best_weights_ep130.npz?download=1"

    elif experiment_type == "peak_regression" and part == "Part3":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part3_v1.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part3_v2.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3239221/files/best_weights_ep91.npz?download=1"

    elif experiment_type == "peak_regression" and part == "Part4":
        weights_path_old = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part4_v1.npz')
        weights_path = os.path.join(C.WEIGHTS_DIR, 'pretrained_weights_peak_regression_part4_v2.npz')
        WEIGHTS_URL = "https://zenodo.org/record/3239222/files/best_weights_ep148.npz?download=1"


    if os.path.exists(weights_path_old):
        os.remove(weights_path_old)

    if WEIGHTS_URL is not None and not os.path.exists(weights_path):
        print("Downloading pretrained weights (~140MB) ...")
        if not os.path.exists(C.WEIGHTS_DIR):
            os.makedirs(C.WEIGHTS_DIR)

        #This results in an SSL Error on CentOS
        # urllib.urlretrieve(WEIGHTS_URL, weights_path)

        data = urlopen(WEIGHTS_URL).read()
        with open(weights_path, "wb") as weight_file:
            weight_file.write(data)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
