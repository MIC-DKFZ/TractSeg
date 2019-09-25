"""
Code to load data and to create batches of 2D slices from 3D images.

Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_3D_image_batched
from batchgenerators.augmentations.crop_and_pad_augmentations import crop

from tractseg.data.data_loader_training import load_training_data
from tractseg.data.custom_transformations import FlipVectorAxisTransform


class BatchGenerator3D_Nifti_random(SlimDataLoaderBase):
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

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idxs = np.random.choice(len(subjects), self.batch_size, False, None)

        x = []
        y = []
        for subject_idx in subject_idxs:
            data, seg = load_training_data(self.Config, subjects[subject_idx])  # (x, y, z, channels)
            data = data.transpose(3, 0, 1, 2)  # channels have to be first
            seg = seg.transpose(3, 0, 1, 2)

            # Crop here instead of cropping entire batch at once to make each element in batch have same dimensions
            data, seg = crop(data[None,...], seg[None,...], crop_size=self.Config.INPUT_DIM)
            data = data.squeeze(axis=0)
            seg = seg.squeeze(axis=0)

            x.append(data)
            y.append(seg)

        x = np.array(x)
        y = np.array(y)

        # Can be replaced by crop -> shorter
        # x = pad_nd_image(x, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # y = pad_nd_image(y, self.Config.INPUT_DIM, mode='constant', kwargs={'constant_values': 0})
        # x = center_crop_3D_image_batched(x, self.Config.INPUT_DIM)
        # y = center_crop_3D_image_batched(y, self.Config.INPUT_DIM)

        # Crop and pad to input size
        # x, y = crop(x, y, crop_size=self.Config.INPUT_DIM)

        # This is needed for Schizo dataset, but only works with DAug=True
        # x = pad_nd_image(x, shape_must_be_divisible_by=(8, 8), mode='constant', kwargs={'constant_values': 0})
        # y = pad_nd_image(y, shape_must_be_divisible_by=(8, 8), mode='constant', kwargs={'constant_values': 0})

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        data_dict = {"data": x,  # (batch_size, channels, x, y, [z])
                     "seg": y}  # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderTraining:

    def __init__(self, Config):
        self.Config = Config

    def _augment_data(self, batch_generator, type=None):

        if self.Config.DATA_AUGMENTATION:
            num_processes = 16
        else:
            num_processes = 6

        tfs = []

        if self.Config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        if self.Config.DATA_AUGMENTATION:
            if type == "train":
                # patch_center_dist_from_border:
                #   if 144/2=72 -> always exactly centered; otherwise a bit off center
                #   (brain can get off image and will be cut then)
                if self.Config.DAUG_SCALE:
                    center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                    tfs.append(SpatialTransform(self.Config.INPUT_DIM,
                                                patch_center_dist_from_border=center_dist_from_border,
                                                do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM,
                                                alpha=(90., 120.), sigma=(9., 11.),
                                                do_rotation=self.Config.DAUG_ROTATE,
                                                angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8), angle_z=(-0.8, 0.8),
                                                do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
                                                border_cval_data=0,
                                                order_data=3,
                                                border_mode_seg='constant', border_cval_seg=0,
                                                order_seg=0, random_crop=True, p_el_per_sample=0.2,
                                                p_rot_per_sample=0.2, p_scale_per_sample=0.2))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.2))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        # num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_generator, Compose(tfs), num_processes=num_processes,
                                           num_cached_per_queue=1, seeds=None, pin_memory=True)
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)


    def get_batch_generator(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        if self.Config.TYPE == "combined":
            raise NotImplementedError("Not implemented yet")
        else:
            batch_gen = BatchGenerator3D_Nifti_random((data, seg), batch_size=batch_size)

        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)

        return batch_gen

