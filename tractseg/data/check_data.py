from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import nibabel as nib
from joblib import Parallel, delayed
import numpy as np

from tractseg.libs.system_config import SystemConfig as C
from tractseg.data.subjects import get_all_subjects
from tractseg.libs import img_utils


# dataset = "HCP_all"
# DATASET_FOLDER_PREPROC = "HCP_preproc_all"  # target folder

dataset = "biobank_20k"
DATASET_FOLDER_PREPROC = "biobank_preproc"


def create_preprocessed_files(subject):

    # filenames_data = ["270g_125mm_bedpostx_peaks_scaled", "32g_125mm_bedpostx_peaks_scaled"]
    # filenames_seg = ["bundle_masks_autoPTX_dm"]

    filenames_data = ["105g_2mm_bedpostx_peaks_scaled"]
    filenames_seg = ["bundle_masks_autoPTX_dm", "bundle_masks_autoPTX_thr001"]

    print("idx: {}, subject: {}".format(subjects.index(subject), subject))

    for idx, filename in enumerate(filenames_data):
        print(filename)
        path = join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")
        data = nib.load(path).get_fdata()
        # _ = img_utils.peak_image_to_tensor_image(data)

    for filename in filenames_seg:
        print(filename)
        data = nib.load(join(C.DATA_PATH, DATASET_FOLDER_PREPROC, subject, filename + ".nii.gz")).get_fdata()


if __name__ == "__main__":
    print("Check folder: {}".format(DATASET_FOLDER_PREPROC))
    subjects = get_all_subjects(dataset=dataset)

    # bad_ids = []
    # files = []
    # for id in bad_ids:
    #     subjects.remove(id)

    Parallel(n_jobs=1)(delayed(create_preprocessed_files)(subject) for subject in subjects)
