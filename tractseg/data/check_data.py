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

import os
from os.path import join
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs.subjects import get_all_subjects
from tractseg.libs import img_utils


dataset = "HCP_all"
DATASET_FOLDER_PREPROC = "HCP_preproc_all"  # target folder

def create_preprocessed_files(subject):

    filenames_data = ["270g_125mm_bedpostx_peaks_scaled", "32g_125mm_bedpostx_peaks_scaled"]
    filenames_seg = ["bundle_masks_autoPTX_dm"]

    print("idx: {}, subject: {}".format(subjects.index(subject), subject))

    for idx, filename in enumerate(filenames_data):
        # print(filename)
        path = join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")

        data = nib.load(path).get_data()
        _ = img_utils.peak_image_to_tensor_image(data)

    for filename in filenames_seg:
        # print(filename)
        data = nib.load(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")).get_data()


if __name__ == "__main__":
    print("Check folder: {}".format(DATASET_FOLDER_PREPROC))
    subjects = get_all_subjects(dataset=dataset)
    Parallel(n_jobs=12)(delayed(create_preprocessed_files)(subject) for subject in subjects)  # 37

