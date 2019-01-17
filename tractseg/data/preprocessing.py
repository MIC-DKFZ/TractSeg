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

"""
Run this script to crop images + segmentations to brain area. Then save as nifti.
Reduces datasize and therefore IO by at least two times.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import nibabel as nib
import numpy as np

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import dataset_utils
from tractseg.libs.subjects import get_all_subjects
from tractseg.libs import exp_utils


DATASET_FOLDER = "HCP"  # source folder
DATASET_FOLDER_PREPROC = "HCP_preproc"  # target folder

def create_preprocessed_files():

    subjects = get_all_subjects(dataset="HCP")
    filenames_data = ["12g_125mm_peaks", "90g_125mm_peaks", "270g_125mm_peaks"]
    filenames_seg = ["bundle_masks_72"]

    # filenames_data = ["12g_125mm_peaks"]
    # filenames_seg = ["bundle_peaks_Part1"]

    for subject in subjects:
        print(subject)
        exp_utils.make_dir(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject))

        for idx, filename in enumerate(filenames_data):
            img = nib.load(join(C.DATA_PATH, DATASET_FOLDER, subject, filename + ".nii.gz"))
            data = img.get_data()
            affine = img.get_affine()
            data = np.nan_to_num(data)

            if idx == 0:
                data, _, bbox, _ = dataset_utils.crop_to_nonzero(data)
            else:
                data, _, _, _ = dataset_utils.crop_to_nonzero(data, bbox=bbox)

            # np.save(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, affine), join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject,
                                                         filename + ".nii.gz"))

        for filename in filenames_seg:
            data = nib.load(join(C.DATA_PATH, DATASET_FOLDER, subject, filename + ".nii.gz")).get_data().astype(np.uint8)
            data, _, _, _ = dataset_utils.crop_to_nonzero(data, bbox=bbox)
            # np.save(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, affine), join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename +
                                                         ".nii.gz"))


if __name__ == "__main__":
    create_preprocessed_files()

