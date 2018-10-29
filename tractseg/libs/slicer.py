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

import sys
import warnings
from os.path import join
import nibabel as nib
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
import scipy.ndimage
from tractseg.libs import img_utils
from tractseg.libs import exp_utils
from tractseg.libs.Config import Config as C
from tractseg.libs import metric_utils
import time
from tractseg.libs import dataset_utils
from tractseg.libs.subjects import get_all_subjects
from tractseg.libs.data_managers import DataManagerTrainingNiftiImgs

np.random.seed(1337)

warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings

class Slicer:

    @staticmethod
    def create_files(HP, slice, train_subjects, validate_subjects, test_subjects):
        print("Creating slice files...")
        start_time = time.time()
        target_data_dir = join(HP.EXP_PATH, "data")
        Slicer._create_slices_file(HP, validate_subjects, join(target_data_dir, "validate_slices"), slice)
        Slicer._create_slices_file(HP, test_subjects, join(target_data_dir, "test_slices"), slice)
        Slicer._create_slices_file(HP, train_subjects, join(target_data_dir, "train_slices"), slice)
        print("took: {}s".format(time.time() - start_time))

    @staticmethod
    def _create_slices_file(HP, subjects, filename, slice, shuffle=True):
        data_dir = join(C.HOME, HP.DATASET_FOLDER)

        dwi_slices = []
        mask_slices = []

        print("\n\nProcessing Data...")
        for s in subjects:
            print("processing dwi subject {}".format(s))

            dwi = nib.load(join(data_dir, s, HP.FEATURES_FILENAME + ".nii.gz"))
            dwi_data = dwi.get_data()
            dwi_data = np.nan_to_num(dwi_data)
            dwi_data = dataset_utils.scale_input_to_unet_shape(dwi_data, HP.DATASET, HP.RESOLUTION)

            # if slice == "x":
            #     for z in range(dwi_data.shape[0]):
            #         dwi_slices.append(dwi_data[z, :, :, :])
            #
            # if slice == "y":
            #     for z in range(dwi_data.shape[1]):
            #         dwi_slices.append(dwi_data[:, z, :, :])
            #
            # if slice == "z":
            #     for z in range(dwi_data.shape[2]):
            #         dwi_slices.append(dwi_data[:, :, z, :])

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

            mask_data = img_utils.create_multilabel_mask(HP, s, labels_type=HP.LABELS_TYPE)
            if HP.RESOLUTION == "2.5mm":
                mask_data = img_utils.resize_first_three_dims(mask_data, order=0, zoom=0.5)
            mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, HP.DATASET, HP.RESOLUTION)

            # if slice == "x":
            #     for z in range(dwi_data.shape[0]):
            #         mask_slices.append(mask_data[z, :, :, :])
            #
            # if slice == "y":
            #     for z in range(dwi_data.shape[1]):
            #         mask_slices.append(mask_data[:, z, :, :])
            #
            # if slice == "z":
            #     for z in range(dwi_data.shape[2]):
            #         mask_slices.append(mask_data[:, :, z, :])

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



    @staticmethod
    def create_prob_files(HP, bundle, train_subjects, validate_subjects, test_subjects):
        print("Creating prob slice files...")
        target_data_dir = join(HP.EXP_PATH, "combined")
        Slicer._create_prob_slices_file(HP, validate_subjects, join(target_data_dir, "validate_slices"), bundle)
        Slicer._create_prob_slices_file(HP, validate_subjects, join(target_data_dir, "test_slices"), bundle)
        Slicer._create_prob_slices_file(HP, test_subjects, join(target_data_dir, "train_slices"), bundle)

    @staticmethod
    def _create_prob_slices_file(HP, subjects, filename, bundle, shuffle=True):

        mask_dir = join(C.HOME, HP.DATASET_FOLDER)

        input_dir = HP.MULTI_PARENT_PATH

        combined_slices = []
        mask_slices = []

        for s in subjects:
            print("processing subject {}".format(s))

            probs_x = nib.load(join(input_dir, "UNet_x_" + str(HP.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
            probs_y = nib.load(join(input_dir, "UNet_y_" + str(HP.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
            probs_z = nib.load(join(input_dir, "UNet_z_" + str(HP.CV_FOLD), "probmaps", s + "_probmap.nii.gz")).get_data()
            # probs_x = DatasetUtils.scale_input_to_unet_shape(probs_x, HP.DATASET, HP.RESOLUTION)
            # probs_y = DatasetUtils.scale_input_to_unet_shape(probs_y, HP.DATASET, HP.RESOLUTION)
            # probs_z = DatasetUtils.scale_input_to_unet_shape(probs_z, HP.DATASET, HP.RESOLUTION)
            combined = np.stack((probs_x, probs_y, probs_z), axis=4)  # (73, 87, 73, 18, 3)  #not working alone: one dim too much for UNet -> reshape
            combined = np.reshape(combined, (combined.shape[0], combined.shape[1], combined.shape[2],
                                             combined.shape[3] * combined.shape[4]))    # (73, 87, 73, 3*18)

            # print("combined shape after", combined.shape)

            mask_data = img_utils.create_multilabel_mask(HP, s, labels_type=HP.LABELS_TYPE)
            if HP.DATASET == "HCP_2mm":
                #use "HCP" because for mask we need downscaling
                mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, "HCP", HP.RESOLUTION)
            elif HP.DATASET == "HCP_2.5mm":
                # use "HCP" because for mask we need downscaling
                mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, "HCP", HP.RESOLUTION)
            else:
                # Mask has same resolution as probmaps -> we can use same resizing
                mask_data = dataset_utils.scale_input_to_unet_shape(mask_data, HP.DATASET, HP.RESOLUTION)

            # Save as Img
            img = nib.Nifti1Image(combined, img_utils.get_dwi_affine(HP.DATASET, HP.RESOLUTION))
            nib.save(img, join(HP.EXP_PATH, "combined", s + "_combinded_probmap.nii.gz"))


            combined = dataset_utils.scale_input_to_unet_shape(combined, HP.DATASET, HP.RESOLUTION)
            assert (combined.shape[2] == mask_data.shape[2])

            #Save as Slices
            for z in range(combined.shape[2]):
                combined_slices.append(combined[:, :, z, :])
                mask_slices.append(mask_data[:, :, z, :])

        if shuffle:
            combined_slices, mask_slices = sk_shuffle(combined_slices, mask_slices, random_state=9)

        if HP.TRAIN:
            np.save(filename + "_data.npy", combined_slices)
            np.save(filename + "_seg.npy", mask_slices)


    #OLD
    @staticmethod
    def create_one_3D_file():
        '''
        Create one big file which contains all 3D Images (not slices).
        '''

        class HP:
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
            data = nib.load(join(C.HOME, HP.DATASET_FOLDER, s, HP.FEATURES_FILENAME + ".nii.gz")).get_data()
            data = np.nan_to_num(data)
            data = dataset_utils.scale_input_to_unet_shape(data, HP.DATASET, HP.RESOLUTION)
        data_all.append(np.array(data))
        np.save("data.npy", data_all)
        del data_all  # free memory

        print("\n\nProcessing Segs...")
        for s in get_all_subjects():
            print("processing seg subject {}".format(s))
            seg = img_utils.create_multilabel_mask(HP, s, labels_type=HP.LABELS_TYPE)
            if HP.RESOLUTION == "2.5mm":
                seg = img_utils.resize_first_three_dims(seg, order=0, zoom=0.5)
            seg = dataset_utils.scale_input_to_unet_shape(seg, HP.DATASET, HP.RESOLUTION)
        seg_all.append(np.array(seg))
        print("SEG TYPE: {}".format(seg_all.dtype))
        np.save("seg.npy", seg_all)

    @staticmethod
    def save_fusion_nifti_as_npy():

        #Can leave this always the same (for 270g and 32g)
        class HP:
            DATASET = "HCP"
            RESOLUTION = "1.25mm"
            FEATURES_FILENAME = "270g_125mm_peaks"
            LABELS_TYPE = np.int16
            LABELS_FILENAME = "bundle_masks"
            DATASET_FOLDER = "HCP"

        #change this for 270g and 32g
        DIFFUSION_FOLDER = "32g_25mm"

        subjects = get_all_subjects()
        # fold0 = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
        # fold1 = ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044']
        # fold2 = ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766']
        # fold3 = ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056']
        # fold4 = ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152']
        # subjects = fold2 + fold3 + fold4

        # subjects = ['654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']

        print("\n\nProcessing Data...")
        for s in subjects:
            print("processing data subject {}".format(s))
            start_time = time.time()
            data = nib.load(join(C.NETWORK_DRIVE, "HCP_fusion_" + DIFFUSION_FOLDER, s + "_probmap.nii.gz")).get_data()
            print("Done Loading")
            data = np.nan_to_num(data)
            data = dataset_utils.scale_input_to_unet_shape(data, HP.DATASET, HP.RESOLUTION)
            data = data[:-1, :, :-1, :]  # cut one pixel at the end, because in scale_input_to_world_shape we ouputted 146 -> one too much at the end
            exp_utils.make_dir(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s))
            np.save(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s, DIFFUSION_FOLDER + "_xyz.npy"), data)
            print("Took {}s".format(time.time() - start_time))

            print("processing seg subject {}".format(s))
            start_time = time.time()
            # seg = ImgUtils.create_multilabel_mask(HP, s, labels_type=HP.LABELS_TYPE)
            seg = nib.load(join(C.NETWORK_DRIVE, "HCP_for_training_COPY", s, HP.LABELS_FILENAME + ".nii.gz")).get_data()
            if HP.RESOLUTION == "2.5mm":
                seg = img_utils.resize_first_three_dims(seg, order=0, zoom=0.5)
            seg = dataset_utils.scale_input_to_unet_shape(seg, HP.DATASET, HP.RESOLUTION)
            np.save(join(C.NETWORK_DRIVE, "HCP_fusion_npy_" + DIFFUSION_FOLDER, s, "bundle_masks.npy"), seg)
            print("Took {}s".format(time.time() - start_time))

    @staticmethod
    def precompute_batches(custom_type=None):
        '''
        9000 slices per epoch -> 200 batches (batchsize=44) per epoch
        => 200-1000 batches needed


        270g_125mm_bundle_peaks_Y: no DAug, no Norm, only Y
        All_sizes_DAug_XYZ: 12g, 90g, 270g, DAug (no rotation, no elastic deform), Norm, XYZ
        270g_125mm_bundle_peaks_XYZ: no DAug, Norm, XYZ
        '''

        class HP:
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

        HP.TRAIN_SUBJECTS, HP.VALIDATE_SUBJECTS, HP.TEST_SUBJECTS = exp_utils.get_cv_fold(HP.CV_FOLD)

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
            dataManager = DataManagerTrainingNiftiImgs(HP)
            batch_gen = dataManager.get_batches(batch_size=HP.BATCH_SIZE, type=type,
                                                subjects=getattr(HP, type.upper() + "_SUBJECTS"), num_batches=num_batches[type])

            for idx, batch in enumerate(batch_gen):
                print("Processing: {}".format(idx))

                # DATASET_DIR = "HCP_batches/270g_125mm_bundle_peaks_Y"
                # DATASET_DIR = "HCP_batches/All_sizes_DAug_XYZ"
                DATASET_DIR = "HCP_batches/270g_125mm_bundle_peaks_XYZ"
                exp_utils.make_dir(join(C.HOME, DATASET_DIR, type))

                data = nib.Nifti1Image(batch["data"], img_utils.get_dwi_affine(HP.DATASET, HP.RESOLUTION))
                nib.save(data, join(C.HOME, DATASET_DIR, type, "batch_" + str(idx) + "_data.nii.gz"))

                seg = nib.Nifti1Image(batch["seg"], img_utils.get_dwi_affine(HP.DATASET, HP.RESOLUTION))
                nib.save(seg, join(C.HOME, DATASET_DIR, type, "batch_" + str(idx) + "_seg.nii.gz"))


if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) > 0:
        type = args[0]
        Slicer.precompute_batches(custom_type=type)
    else:
        Slicer.precompute_batches()

