#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

from os.path import expanduser, join, isfile
import numpy as np
import nibabel as nib
import time
import logging
from libs.Utils import Utils
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.imaffine import AffineMap
from libs.Config import Config as C

logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
logging.getLogger().setLevel(logging.INFO)

class RegistrationUtils:

    @staticmethod
    def get_elastic_transform(subject_fa, atlas_fa, subject_path=".."):
        '''
        :param subject_fa: the FA (nibabel img) of a static image of a subject       (static)
        :param atlas_fa:  the FA (nibabel img) of an atlas (Atlas will be warped onto subject)   (moving)

        :return: elastic transformation map
        '''

        if isfile(subject_path + "/FAReg_elastic_transform.pklz"):
            logging.debug("Load existing elastic transform...")
            return Utils.load_pkl_compressed(subject_path + "/FAReg_elastic_transform.pklz")

        static_img = subject_fa
        static = static_img.get_data()
        moving_img = atlas_fa
        moving = moving_img.get_data()

        # Optional (affine transformation of moving image to static coordinate system) -> needed if on very different ones!
        affine_map = AffineMap(np.eye(4),
                               static.shape, static_img.get_affine(),
                               moving.shape, moving_img.get_affine())
        moving = affine_map.transform(moving)

        start_time = time.time()
        metric = CCMetric(3)
        level_iters = [10, 10, 5]  # better
        # level_iters = [2, 2, 2] #fast -> not much
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(static, moving)
        # mapping = sdr.optimize(static, moving, Utils.invert_x_and_y(static_img.get_affine()), Utils.invert_x_and_y(moving_img.get_affine())) #not needed
        logging.debug("elastic transform took {0:.2f}s".format(time.time() - start_time))

        logging.debug("write elastic transform...")
        Utils.save_pkl_compressed(subject_path + "/FAReg_elastic_transform.pklz", mapping)
        return mapping

    @staticmethod
    def register_mask(mask_data, mask_affine, reference_img, elastic_transform=None, binary_img=True, use_inverse=False):
        '''
        Transform a mask (binary image) with the given elastic_transform

        :param mask_data:            data of the mask that should be transformed
        :param mask_affine:     affine of the mask that should be transformed
        :param reference_img:   a nibabel image to get shape and affine from for the Affine Transformation
        :param elastic_transform:
        :param binary_img:      is input a float image (eg T1) or a binary image (eg a mask)

        :return: transformed mask (a binary Image)
        '''

        logging.debug("mask original shape: {}".format(mask_data.shape))

        # Apply affine for mask image (to t1 space)
        affine_map_inv = AffineMap(np.eye(4),
                                   reference_img.get_data().shape, Utils.invert_x_and_y(reference_img.get_affine()),
                                   mask_data.shape, Utils.invert_x_and_y(mask_affine)
                                   )  # If I do not use invert_x_and_y for source and target, result is identical
        mask_data_reg = affine_map_inv.transform(mask_data)
        if binary_img:
            mask_data_reg = mask_data_reg > 0
        logging.debug("mask registered shape: {}".format(mask_data_reg.shape))

        if elastic_transform:

            # img = nib.Nifti1Image(mask_data_reg.astype(np.uint8), reference_img.get_affine())
            # nib.save(img, "ROI_registered_before.nii.gz")

            if use_inverse:
                mask_data_reg = elastic_transform.transform_inverse(mask_data_reg)
            else:
                mask_data_reg = elastic_transform.transform(mask_data_reg)

            if binary_img:
                mask_data_reg = mask_data_reg > 0

            # img = nib.Nifti1Image(mask_data_reg.astype(np.uint8), reference_img.get_affine())
            # nib.save(img, "ROI_registered_after.nii.gz")

        else:
            logging.warning("Elastic Transform deactivated; only using Affine Transform")

        if binary_img:
            mask_data_reg = mask_data_reg > 0
        return mask_data_reg

    @staticmethod
    def register_HCP_pipeline_image(img_moving, use_inverse, is_binary_img, subject_path=".."):
        subject_fa = nib.load(subject_path + "/FA.nii.gz")
        atlas_fa = nib.load(join(C.NETWORK_DRIVE, "HCP", "994273", "270g_125mm", "FA.nii.gz"))
        reference_img = nib.load(subject_path + "/nodif_brain_mask.nii.gz")
        elastic_transform = RegistrationUtils.get_elastic_transform(subject_fa, atlas_fa, subject_path=subject_path)
        # Transform from current subject to 994273 (= inverse)
        return RegistrationUtils.register_mask(img_moving.get_data(), img_moving.get_affine(), reference_img,
                                               elastic_transform=elastic_transform,
                                               binary_img=is_binary_img,
                                               use_inverse=use_inverse)
