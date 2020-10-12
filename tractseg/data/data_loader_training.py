"""
Code to load data and to create batches of 2D slices from 3D images.

Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import random

import numpy as np
import nibabel as nib

from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.spatial_transforms import ZoomTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_2D_image_batched
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.spatial_transformations import augment_zoom

# from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform as ZeroMeanUnitVarianceTransform_Standalone

from tractseg.data.custom_transformations import ResampleTransformLegacy
from tractseg.data.custom_transformations import FlipVectorAxisTransform
from tractseg.data.spatial_transform_peaks import SpatialTransformPeaks
from tractseg.data.spatial_transform_custom import SpatialTransformCustom
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import data_utils
from tractseg.libs import peak_utils


def load_training_data(Config, subject):
    """
    Load data and labels for one subject from the training set. Cut and scale to make them have
    correct size.

    Args:
        Config: config class
        subject: subject id (string)

    Returns:
        data and labels as 3D array
    """
    def load(filepath):
        data = nib.load(filepath + ".nii.gz").get_fdata()
        # data = np.load(filepath + ".npy", mmap_mode="r")
        return data

    if Config.FEATURES_FILENAME == "12g90g270g":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))

    elif Config.FEATURES_FILENAME == "12g90g270gRaw32g":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_raw32g"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_raw32g"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_raw32g"))

    elif Config.FEATURES_FILENAME == "12g90g270g_BX":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_bedpostx_peaks_scaled"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_bedpostx_peaks_scaled"))

    elif Config.FEATURES_FILENAME == "12g90g270g_FA":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_FA"))
        elif rnd_choice < 0.66:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_FA"))
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_FA"))

    elif Config.FEATURES_FILENAME == "12g90g270g_CSD_BX":
        rnd_choice_1 = np.random.random()
        rnd_choice_2 = np.random.random()
        if rnd_choice_1 < 0.5:  # CSD
            if rnd_choice_2 < 0.33:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
            elif rnd_choice_2 < 0.66:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))
        else:  # BX
            if rnd_choice_2 < 0.33:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
            elif rnd_choice_2 < 0.66:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_bedpostx_peaks_scaled"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_bedpostx_peaks_scaled"))
            # Flip x axis to make BedpostX compatible with mrtrix CSD
            data[:, :, :, 0] *= -1
            data[:, :, :, 3] *= -1
            data[:, :, :, 6] *= -1

    elif Config.FEATURES_FILENAME == "32g90g270g_CSD_BX":
        rnd_choice_1 = np.random.random()
        rnd_choice_2 = np.random.random()
        if rnd_choice_1 < 0.5:  # CSD
            if rnd_choice_2 < 0.33:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
            elif rnd_choice_2 < 0.66:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "32g_125mm_peaks"))
        else:  # BX
            if rnd_choice_2 < 0.5:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))
            else:
                data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "32g_125mm_bedpostx_peaks_scaled"))
            # Flip x axis to make BedpostX compatible with mrtrix CSD
            data[:, :, :, 0] *= -1
            data[:, :, :, 3] *= -1
            data[:, :, :, 6] *= -1

    elif Config.FEATURES_FILENAME == "105g_CSD_BX":
        rnd_choice_1 = np.random.random()
        if rnd_choice_1 < 0.5:  # CSD
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "105g_2mm_peaks"))
        else:  # BX
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "105g_2mm_bedpostx_peaks_scaled"))
            # Flip x axis to make BedpostX compatible with mrtrix CSD
            data[:, :, :, 0] *= -1
            data[:, :, :, 3] *= -1
            data[:, :, :, 6] *= -1

    elif Config.FEATURES_FILENAME == "32g270g_BX":
        rnd_choice = np.random.random()
        path_32g = join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "32g_125mm_bedpostx_peaks_scaled")
        if rnd_choice < 0.5:
            data = load(path_32g)
            rnd_choice_2 = np.random.random()
            if rnd_choice_2 < 0.5:
                data[:, :, :, 6:9] = 0  # set third peak to 0
        else:
            data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_bedpostx_peaks_scaled"))

    elif Config.FEATURES_FILENAME == "T1_Peaks270g":
        peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        t1 = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1"))
        data = np.concatenate((peaks, t1), axis=3)

    elif Config.FEATURES_FILENAME == "T1_Peaks12g90g270g":
        rnd_choice = np.random.random()
        if rnd_choice < 0.33:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "270g_125mm_peaks"))
        elif rnd_choice < 0.66:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "90g_125mm_peaks"))
        else:
            peaks = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "12g_125mm_peaks"))
        t1 = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, "T1"))
        data = np.concatenate((peaks, t1), axis=3)

    else:
        data = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.FEATURES_FILENAME))

    if "|" in Config.LABELS_FILENAME:
        parts = Config.LABELS_FILENAME.split("|")
        seg = []  # [4, x, y, z, 54]
        for part in parts:
            seg.append(load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, part)))
        seg = np.array(seg).transpose(1, 2, 3, 4, 0)
        seg = seg.reshape(data.shape[:3] + (-1,))  # [x, y, z, 54*4]
    else:
        seg = load(join(C.DATA_PATH, Config.DATASET_FOLDER, subject, Config.LABELS_FILENAME))

    return data, seg


class BatchGenerator2D_Nifti_random(SlimDataLoaderBase):
    """
    Randomly selects subjects and slices and creates batch of 2D slices.

    Takes image IDs provided via self._data, randomly selects one ID,
    loads the nifti image and randomly samples 2D slices from it.

    Timing:
    About 2s per 54-batch 45 bundles 1.25mm.
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def _zoom_x_and_y(self, x, y, zoom_factor):
        # Very slow
        x_new = []
        y_new = []
        for b in range(x.shape[0]):
            x_tmp, y_tmp = augment_zoom(x[b], y[b], zoom_factor, order=3, order_seg=1, cval_seg=0)
            x_new.append(x_tmp)
            y_new.append(y_tmp)
        return np.array(x_new), np.array(y_new)

    def generate_train_batch(self):

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))

        data, seg = load_training_data(self.Config, subjects[subject_idx])

        # Convert peaks to tensors if tensor model
        if self.Config.NR_OF_GRADIENTS == 18*self.Config.NR_SLICES:
            data = peak_utils.peaks_to_tensors(data)

        slice_direction = data_utils.slice_dir_to_int(self.Config.TRAINING_SLICE_DIRECTION)
        if data.shape[slice_direction] <= self.batch_size:
            print("INFO: Batch size bigger than nr of slices. Therefore sampling with replacement.")
            slice_idxs = np.random.choice(data.shape[slice_direction], self.batch_size, True, None)
        else:
            slice_idxs = np.random.choice(data.shape[slice_direction], self.batch_size, False, None)

        if self.Config.NR_SLICES > 1:
            x, y = data_utils.sample_Xslices(data, seg, slice_idxs, slice_direction=slice_direction,
                                             labels_type=self.Config.LABELS_TYPE, slice_window=self.Config.NR_SLICES)
        else:
            x, y = data_utils.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction,
                                            labels_type=self.Config.LABELS_TYPE)


        # Can be replaced by crop
        # x = pad_nd_image(x, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # y = pad_nd_image(y, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # x = center_crop_2D_image_batched(x, self.Config.INPUT_DIM)
        # y = center_crop_2D_image_batched(y, self.Config.INPUT_DIM)

        # If want to convert e.g. 1.25mm (HCP) image to 2mm image (bb)
        # x, y = self._zoom_x_and_y(x, y, 0.67)  # very slow -> try spatial_transform, should be fast

        if self.Config.PAD_TO_SQUARE:
            #Crop and pad to input size
            x, y = crop(x, y, crop_size=self.Config.INPUT_DIM)  # does not work with img with batches and channels
        else:
            # Works -> results as good?
            # Will pad each axis to be multiple of 16. (Each sample can end up having different dimensions. Also x and y
            # can be different)
            # This is needed for Schizo dataset
            x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})
            y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})

        # Does not make it slower
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # possible optimization: sample slices from different patients and pad all to same size (size of biggest)

        data_dict = {"data": x,  # (batch_size, channels, x, y, [z])
                     "seg": y,
                     "slice_dir": slice_direction}  # (batch_size, channels, x, y, [z])
        return data_dict


class BatchGenerator2D_Npy_random(SlimDataLoaderBase):
    """
    Takes image ID provided via self._data, loads the Npy (numpy array) image and randomly samples 2D slices from it.
    Needed for fusion training.

    Timing:
    About 2s per 54-batch 45 bundles 1.25mm.
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None

    def generate_train_batch(self):

        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))

        if self.Config.TYPE == "combined":
            if np.random.random() < 0.5:
                data = np.load(join(C.DATA_PATH, "HCP_fusion_npy_270g_125mm",
                                    subjects[subject_idx], "270g_125mm_xyz.npy"), mmap_mode="r")
            else:
                data = np.load(join(C.DATA_PATH, "HCP_fusion_npy_32g_25mm",
                                    subjects[subject_idx], "32g_25mm_xyz.npy"), mmap_mode="r")
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx],
                               self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")
        else:
            data = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx],
                                self.Config.FEATURES_FILENAME + ".npy"), mmap_mode="r")
            seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, subjects[subject_idx],
                               self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")

        data = np.nan_to_num(data)
        seg = np.nan_to_num(seg)

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)
        slice_direction = data_utils.slice_dir_to_int(self.Config.TRAINING_SLICE_DIRECTION)
        x, y = data_utils.sample_slices(data, seg, slice_idxs,
                                        slice_direction=slice_direction,
                                        labels_type=self.Config.LABELS_TYPE)

        data_dict = {"data": x,  # (batch_size, channels, x, y, [z])
                     "seg": y}  # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderTraining:

    def __init__(self, Config):
        self.Config = Config

    def _augment_data(self, batch_generator, type=None):

        if self.Config.DATA_AUGMENTATION:
            num_processes = 15  # 15 is a bit faster than 8 on cluster
            # num_processes = multiprocessing.cpu_count()  # on cluster: gives all cores, not only assigned cores
        else:
            num_processes = 6

        tfs = []

        if self.Config.NORMALIZE_DATA:
            # todo: Use original transform as soon as bug fixed in batchgenerators
            # tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))
            tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        if self.Config.SPATIAL_TRANSFORM == "SpatialTransformPeaks":
            SpatialTransformUsed = SpatialTransformPeaks
        elif self.Config.SPATIAL_TRANSFORM == "SpatialTransformCustom":
            SpatialTransformUsed = SpatialTransformCustom
        else:
            SpatialTransformUsed = SpatialTransform

        if self.Config.DATA_AUGMENTATION:
            if type == "train":
                # patch_center_dist_from_border:
                #   if 144/2=72 -> always exactly centered; otherwise a bit off center
                #   (brain can get off image and will be cut then)
                if self.Config.DAUG_SCALE:

                    if self.Config.INPUT_RESCALING:
                        source_mm = 2  # for bb
                        target_mm = float(self.Config.RESOLUTION[:-2])
                        scale_factor = target_mm / source_mm
                        scale = (scale_factor, scale_factor)
                    else:
                        scale = (0.9, 1.5)

                    if self.Config.PAD_TO_SQUARE:
                        patch_size = self.Config.INPUT_DIM
                    else:
                        patch_size = None  # keeps dimensions of the data

                    # spatial transform automatically crops/pads to correct size
                    center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                    tfs.append(SpatialTransformUsed(patch_size,
                                                patch_center_dist_from_border=center_dist_from_border,
                                                do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM,
                                                alpha=self.Config.DAUG_ALPHA, sigma=self.Config.DAUG_SIGMA,
                                                do_rotation=self.Config.DAUG_ROTATE,
                                                angle_x=self.Config.DAUG_ROTATE_ANGLE,
                                                angle_y=self.Config.DAUG_ROTATE_ANGLE,
                                                angle_z=self.Config.DAUG_ROTATE_ANGLE,
                                                do_scale=True, scale=scale, border_mode_data='constant',
                                                border_cval_data=0,
                                                order_data=3,
                                                border_mode_seg='constant', border_cval_seg=0,
                                                order_seg=0, random_crop=True,
                                                p_el_per_sample=self.Config.P_SAMP,
                                                p_rot_per_sample=self.Config.P_SAMP,
                                                p_scale_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

                if self.Config.DAUG_RESAMPLE_LEGACY:
                    tfs.append(ResampleTransformLegacy(zoom_range=(0.5, 1)))

                if self.Config.DAUG_GAUSSIAN_BLUR:
                    tfs.append(GaussianBlurTransform(blur_sigma=self.Config.DAUG_BLUR_SIGMA,
                                                     different_sigma_per_channel=False,
                                                     p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=self.Config.DAUG_NOISE_VARIANCE,
                                                      p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_generator, Compose(tfs), num_processes=num_processes,
                                           num_cached_per_queue=1, seeds=None, pin_memory=True)
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


    def get_batch_generator(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        if self.Config.TYPE == "combined":
            batch_gen = BatchGenerator2D_Npy_random((data, seg), batch_size=batch_size)
        else:
            batch_gen = BatchGenerator2D_Nifti_random((data, seg), batch_size=batch_size)
            # batch_gen = SlicesBatchGeneratorRandomNiftiImg_5slices((data, seg), batch_size=batch_size)

        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)

        return batch_gen
