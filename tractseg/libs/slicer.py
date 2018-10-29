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

import sys
import warnings
from os.path import join
import time
import nibabel as nib
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
import scipy.ndimage

from tractseg.libs import img_utils
from tractseg.libs import exp_utils
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import metric_utils
from tractseg.libs import dataset_utils
from tractseg.libs.subjects import get_all_subjects
from tractseg.libs.data_managers import DataManagerTrainingNiftiImgs


np.random.seed(1337)
warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings



def create_files(Config, slice, train_subjects, validate_subjects, test_subjects):
    print("Creating slice files...")
    start_time = time.time()
    target_data_dir = join(Config.EXP_PATH, "data")
    _create_slices_file(Config, validate_subjects, join(target_data_dir, "validate_slices"), slice)
    _create_slices_file(Config, test_subjects, join(target_data_dir, "test_slices"), slice)
    _create_slices_file(Config, train_subjects, join(target_data_dir, "train_slices"), slice)
    print("took: {}s".format(time.time() - start_time))


def _create_slices_file(Config, subjects, filename, slice, shuffle=True):
    data_dir = join(C.HOME, Config.DATASET_FOLDER)

    dwi_slices = []
    mask_slices = []

    print("\n\nProcessing Data...")
    for s in subjects:
        print("processing dwi subject {}".format(s))

        dwi = nib.load(join(data_dir, s, Config.FEATURES_FILENAME + ".nii.gz"))
        dwi_data = dwi.get_data()
        dwi_data = np.nan_to_num(dwi_data)
        dwi_data = dataset_utils.scale_input_to_unet_shape(dwi_data, Config.DATASET, Config.RESOLUTION)

        #Use slices from all directions in one dataset
        for z in range(dwi_data.shape[0]):
            dwi_slices.append(dwi_data[z, :, :, :])
        for z in range(dwi_data.shape[1]):
            dwi_slices.append(dwi_data[:, z, :, :])
        for z in range(dwi_data.shape[2]):
            dwi_slices.append(dwi_data[:, :, z, :])

    dwi_slices = np.array(dwi_slices)
    random_idxs = None
    if shuffle:
        random_idxs = np.random.choice(len(dwi_slices), len(dwi_slices))
        dwi_slices = dwi_slices[random_idxs]

    np.save(filename + "_data.npy", dwi_slices)
    del dwi_slices  #free memory


    print("\n\nProcessing Segs...")
    for s in subjects:
        print("processing seg subject {}".format(s))

        mask_data = img_utils.create_multilabel_mask(Config, s, labels_type=Config.LABELS_TYPE)
        if Config.RESOLUTION == "2.5mm":
            mask_data = img_utils.resize_first_three_dims(mask_data, order=0, zoom=0.5)
        mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, Config.DATASET, Config.RESOLUTION)

        # Use slices from all directions in one dataset
        for z in range(dwi_data.shape[0]):
            mask_slices.append(mask_data[z, :, :, :])
        for z in range(dwi_data.shape[1]):
            mask_slices.append(mask_data[:, z, :, :])
        for z in range(dwi_data.shape[2]):
            mask_slices.append(mask_data[:, :, z, :])

    mask_slices = np.array(mask_slices)
    print("SEG TYPE: {}".format(mask_slices.dtype))
    if shuffle:
        mask_slices = mask_slices[random_idxs]

    np.save(filename + "_seg.npy", mask_slices)


def create_prob_files(Config, bundle, train_subjects, validate_subjects, test_subjects):
    print("Creating prob slice files...")
    target_data_dir = join(Config.EXP_PATH, "combined")
    _create_prob_slices_file(Config, validate_subjects, join(target_data_dir, "validate_slices"), bundle)
    _create_prob_slices_file(Config, validate_subjects, join(target_data_dir, "test_slices"), bundle)
    _create_prob_slices_file(Config, test_subjects, join(target_data_dir, "train_slices"), bundle)


def _create_prob_slices_file(Config, subjects, filename, bundle, shuffle=True):

    input_dir = Config.MULTI_PARENT_PATH

    combined_slices = []
    mask_slices = []

    for s in subjects:
        print("processing subject {}".format(s))

        probs_x = nib.load(join(input_dir, "UNet_x_" + str(Config.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
        probs_y = nib.load(join(input_dir, "UNet_y_" + str(Config.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
        probs_z = nib.load(join(input_dir, "UNet_z_" + str(Config.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
        # probs_x = DatasetUtils.scale_input_to_unet_shape(probs_x, Config.DATASET, Config.RESOLUTION)
        # probs_y = DatasetUtils.scale_input_to_unet_shape(probs_y, Config.DATASET, Config.RESOLUTION)
        # probs_z = DatasetUtils.scale_input_to_unet_shape(probs_z, Config.DATASET, Config.RESOLUTION)
        combined = np.stack((probs_x, probs_y, probs_z), axis=4)  # (73, 87, 73, 18, 3); not working alone: one dim too much for UNet -> reshape
        combined = np.reshape(combined, (combined.shape[0], combined.shape[1], combined.shape[2],
                                         combined.shape[3] * combined.shape[4]))    # (73, 87, 73, 3*18)

        mask_data = img_utils.create_multilabel_mask(Config, s, labels_type=Config.LABELS_TYPE)
        if Config.DATASET == "HCP_2mm":
            #use "HCP" because for mask we need downscaling
            mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, "HCP", Config.RESOLUTION)
        elif Config.DATASET == "HCP_2.5mm":
            # use "HCP" because for mask we need downscaling
            mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, "HCP", Config.RESOLUTION)
        else:
            # Mask has same resolution as probmaps -> we can use same resizing
            mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, Config.DATASET, Config.RESOLUTION)

        # Save as Img
        img = nib.Nifti1Image(combined, img_utils.get_dwi_affine(Config.DATASET, Config.RESOLUTION))
        nib.save(img, join(Config.EXP_PATH, "combined", s + "_combinded_probmap.nii.gz"))


        combined = dataset_utils.scale_input_to_unet_shape(combined, Config.DATASET, Config.RESOLUTION)
        assert (combined.shape[2] == mask_data.shape[2])

        #Save as Slices
        for z in range(combined.shape[2]):
            combined_slices.append(combined[:, :, z, :])
            mask_slices.append(mask_data[:, :, z, :])

    if shuffle:
        combined_slices, mask_slices = sk_shuffle(combined_slices, mask_slices, random_state=9)

    if Config.TRAIN:
        np.save(filename + "_data.npy", combined_slices)
        np.save(filename + "_seg.npy", mask_slices)


#OLD
def create_one_3D_file():
    '''
    Create one big file which contains all 3D Images (not slices).
    '''
    class Config:
        DATASET = "HCP"
        RESOLUTION = "1.25mm"
        FEATURES_FILENAME = "270g_125mm_peaks"
        LABELS_TYPE = np.int16
        DATASET_FOLDER = "HCP"

    data_all = []
    seg_all = []

    print("\n\nProcessing Data...")
    for s in get_all_subjects():
        print("processing data subject {}".format(s))
        data = nib.load(join(C.HOME, Config.DATASET_FOLDER, s, Config.FEATURES_FILENAME + ".nii.gz")).get_data()
        data = np.nan_to_num(data)
        data = dataset_utils.scale_input_to_unet_shape(data, Config.DATASET, Config.RESOLUTION)
    data_all.append(np.array(data))
    np.save("data.npy", data_all)
    del data_all  # free memory

    print("\n\nProcessing Segs...")
    for s in get_all_subjects():
        print("processing seg subject {}".format(s))
        seg = img_utils.create_multilabel_mask(Config, s, labels_type=Config.LABELS_TYPE)
        if Config.RESOLUTION == "2.5mm":
            seg = img_utils.resize_first_three_dims(seg, order=0, zoom=0.5)
        seg = dataset_utils.scale_input_to_unet_shape(seg, Config.DATASET, Config.RESOLUTION)
    seg_all.append(np.array(seg))
    print("SEG TYPE: {}".format(seg_all.dtype))
    np.save("seg.npy", seg_all)


def save_fusion_nifti_as_npy():

    #Can leave this always the same (for 270g and 32g)
    class Config:
        DATASET = "HCP"
        RESOLUTION = "1.25mm"
        FEATURES_FILENAME = "270g_125mm_peaks"
        LABELS_TYPE = np.int16
        LABELS_FILENAME = "bundle_masks"
        DATASET_FOLDER = "HCP"

    DIFFUSION_FOLDER = "32g_25mm"
    subjects = get_all_subjects()

    print("\n\nProcessing Data...")
    for s in subjects:
        print("processing data subject {}".format(s))
        start_time = time.time()
        data = nib.load(join(C.NETWORK_DRIVE, "HCP_fusion_" + DIFFUSION_FOLDER, s + "_probmap.nii.gz")).get_data()
        print("Done Loading")
        data = np.nan_to_num(data)
        data = dataset_utils.scale_input_to_unet_shape(data, Config.DATASET, Config.RESOLUTION)
        data = data[:-1, :, :-1, :]  # cut one pixel at the end, because in scale_input_to_world_shape we ouputted 146 -> one too much at the end
        exp_utils.make_dir(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s))
        np.save(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s, DIFFUSION_FOLDER + "_xyz.npy"), data)
        print("Took {}s".format(time.time() - start_time))

        print("processing seg subject {}".format(s))
        start_time = time.time()
        # seg = ImgUtils.create_multilabel_mask(Config, s, labels_type=Config.LABELS_TYPE)
        seg = nib.load(join(C.NETWORK_DRIVE, "HCP_for_training_COPY", s, Config.LABELS_FILENAME + ".nii.gz")).get_data()
        if Config.RESOLUTION == "2.5mm":
            seg = img_utils.resize_first_three_dims(seg, order=0, zoom=0.5)
        seg = dataset_utils.scale_input_to_unet_shape(seg, Config.DATASET, Config.RESOLUTION)
        np.save(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s, "bundle_masks.npy"), seg)
        print("Took {}s".format(time.time() - start_time))


def precompute_batches(custom_type=None):
    '''
    9000 slices per epoch -> 200 batches (batchsize=44) per epoch
    => 200-1000 batches needed
    '''
    class Config:
        NORMALIZE_DATA = True
        DATA_AUGMENTATION = False
        CV_FOLD = 0
        INPUT_DIM = (144, 144)
        BATCH_SIZE = 44
        DATASET_FOLDER = "HCP"
        TYPE = "single_direction"
        EXP_PATH = "~"
        LABELS_FILENAME = "bundle_peaks"
        FEATURES_FILENAME = "270g_125mm_peaks"
        DATASET = "HCP"
        RESOLUTION = "1.25mm"
        LABELS_TYPE = np.float32

    Config.TRAIN_SUBJECTS, Config.VALIDATE_SUBJECTS, Config.TEST_SUBJECTS = exp_utils.get_cv_fold(Config.CV_FOLD)

    num_batches_base = 5000
    num_batches = {
        "train": num_batches_base,
        "validate": int(num_batches_base / 3.),
        "test": int(num_batches_base / 3.),
    }

    if custom_type is None:
        types = ["train", "validate", "test"]
    else:
        types = [custom_type]

    for type in types:
        dataManager = DataManagerTrainingNiftiImgs(Config)
        batch_gen = dataManager.get_batches(batch_size=Config.BATCH_SIZE, type=type,
                                            subjects=getattr(Config, type.upper() + "_SUBJECTS"), num_batches=num_batches[type])

        for idx, batch in enumerate(batch_gen):
            print("Processing: {}".format(idx))

            DATASET_DIR = "HCP_batches/270g_125mm_bundle_peaks_XYZ"
            exp_utils.make_dir(join(C.HOME, DATASET_DIR, type))

            data = nib.Nifti1Image(batch["data"], img_utils.get_dwi_affine(Config.DATASET, Config.RESOLUTION))
            nib.save(data, join(C.HOME, DATASET_DIR, type, "batch_" + str(idx) + "_data.nii.gz"))

            seg = nib.Nifti1Image(batch["seg"], img_utils.get_dwi_affine(Config.DATASET, Config.RESOLUTION))
            nib.save(seg, join(C.HOME, DATASET_DIR, type, "batch_" + str(idx) + "_seg.nii.gz"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        type = args[0]
        precompute_batches(custom_type=type)
    else:
        precompute_batches()

