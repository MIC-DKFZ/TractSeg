"""
Run this script to crop images + segmentations to brain area. Then save as nifti.
Reduces datasize and therefore IO by at least factor of 2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import data_utils
from tractseg.data.subjects import get_all_subjects
from tractseg.libs import exp_utils


#todo: adapt
dataset = "HCP_final"
DATASET_FOLDER = "HCP_for_training_COPY"  # source folder
DATASET_FOLDER_PREPROC = "HCP_preproc"  # target folder

# dataset = "HCP_all"
# DATASET_FOLDER = "data/HCP_all_training"
# DATASET_FOLDER_PREPROC = "HCP_preproc_all"

# dataset = "biobank_20k"
# DATASET_FOLDER = "data/biobank_aPTX_20k_training"
# DATASET_FOLDER_PREPROC = "biobank_preproc"


def create_preprocessed_files(subject):

    # if file already exists skip it
    check_for_existing_files = False

    # Estimate bounding box from this file and then apply it to all other files
    bb_file = "12g_125mm_peaks"
    # bb_file = "105g_2mm_bedpostx_peaks_scaled"
    # bb_file = "270g_125mm_peaks"

    # todo: adapt
    # filenames_data = ["12g_125mm_peaks", "90g_125mm_peaks", "270g_125mm_peaks",
    #                   "12g_125mm_bedpostx_peaks_scaled", "90g_125mm_bedpostx_peaks_scaled",
    #                   "270g_125mm_bedpostx_peaks_scaled"]
    # filenames_seg = ["bundle_masks_72", "bundle_masks_dm", "endpoints_72_ordered",
    #                  "bundle_peaks_Part1", "bundle_peaks_Part2", "bundle_peaks_Part3", "bundle_peaks_Part4",
    #                  "bundle_masks_autoPTX_dm", "bundle_masks_autoPTX_thr001"]

    # filenames_data = ["105g_2mm_bedpostx_peaks_scaled", "105g_2mm_peaks"]
    # filenames_seg = ["bundle_masks_xtract_dm", "bundle_masks_xtract_thr001"]

    # filenames_data = ["32g_125mm_peaks", "90g_125mm_peaks", "270g_125mm_peaks",
    #                   "32g_125mm_bedpostx_peaks_scaled", "270g_125mm_bedpostx_peaks_scaled"]
    # filenames_seg = ["bundle_masks_xtract_dm", "bundle_masks_xtract_thr001"]

    filenames_data = ["12g_125mm_raw32g", "270g_125mm_raw32g"]
    filenames_seg = []


    print("idx: {}".format(subjects.index(subject)))
    exp_utils.make_dir(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject))

    bb_file_path = join(C.NETWORK_DRIVE, DATASET_FOLDER, subject, bb_file + ".nii.gz")
    if not os.path.exists(bb_file_path):
        print("Missing file: {}-{}".format(subject, bb_file))
        raise IOError("File missing")

    # Get bounding box
    data = nib.load(bb_file_path).get_fdata()
    _, _, bbox, _ = data_utils.crop_to_nonzero(np.nan_to_num(data))

    for idx, filename in enumerate(filenames_data):
        path_src = join(C.NETWORK_DRIVE, DATASET_FOLDER, subject, filename + ".nii.gz")
        path_target = join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")
        if os.path.exists(path_target) and check_for_existing_files:
            print("Already done: {} - {}".format(subject, filename))
        elif os.path.exists(path_src):
            img = nib.load(path_src)
            data = img.get_fdata()
            affine = img.affine
            data = np.nan_to_num(data)

            # Add channel dimension if does not exist yet
            if len(data.shape) == 3:
                data = data[..., None]

            data, _, _, _ = data_utils.crop_to_nonzero(data, bbox=bbox)

            # np.save(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, affine), path_target)
        else:
            print("Missing file: {}-{}".format(subject, idx))
            raise IOError("File missing")

    for idx, filename in enumerate(filenames_seg):
        path_src = join(C.NETWORK_DRIVE, DATASET_FOLDER, subject, filename + ".nii.gz")
        path_target = join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")
        if os.path.exists(path_target) and check_for_existing_files:
            print("Already done: {} - {}".format(subject, filename))
        elif os.path.exists(path_src):
            img = nib.load(path_src)
            data = img.get_fdata()
            data, _, _, _ = data_utils.crop_to_nonzero(data, bbox=bbox)
            # np.save(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, img.affine), path_target)
        else:
            print("Missing seg file: {}-{}".format(subject, idx))
            raise IOError("File missing")


if __name__ == "__main__":
    print("Output folder: {}".format(DATASET_FOLDER_PREPROC))
    subjects = get_all_subjects(dataset=dataset)
    Parallel(n_jobs=12)(delayed(create_preprocessed_files)(subject) for subject in subjects)
    # for subject in subjects:
    #     create_preprocessed_files(subject)
