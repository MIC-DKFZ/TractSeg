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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import nibabel as nib
import numpy as np

from tractseg.libs import exp_utils

class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    # def test_csd_peaks(self):
    #     img_ref = nib.load("tests/reference_files/peaks.nii.gz").get_data()
    #     img_new = nib.load("examples/tractseg_output/peaks.nii.gz").get_data()
    #     images_equal = np.allclose(img_ref, img_new, rtol=1, atol=1)    #somehow not working; order of channels randomly changing?
    #     self.assertTrue(images_equal, "CSD peaks not correct")

    # def test_tractseg_output_docker(self):
    #     bundles = exp_utils.get_bundle_names("All")[1:]
    #     for bundle in bundles:
    #         img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_data()
    #         img_new = nib.load("examples/docker_test/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
    #         images_equal = np.array_equal(img_ref, img_new)
    #         self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_tractseg_output(self):
        bundles = exp_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_tractseg_output_SR_PP_BST(self):
        bundles = exp_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations_SR_PP_BST/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/SR_PP_BST/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            # Processing on travis slightly different from local environment -> have to allow for small margin
            # images_equal = np.array_equal(img_ref, img_new)
            nr_differing_voxels = np.abs(img_ref - img_new).sum()
            if nr_differing_voxels < 5:
                images_equal = True
            else:
                images_equal = False
            self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_endingsseg_output(self):
        bundles = exp_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/endings_segmentations/" + bundle + "_b.nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/endings_segmentations/" + bundle + "_b.nii.gz").get_data()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Bundle endings are not correct (bundle: " + bundle + "_b)")

            img_ref = nib.load("tests/reference_files/endings_segmentations/" + bundle + "_e.nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/endings_segmentations/" + bundle + "_e.nii.gz").get_data()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Bundle endings are not correct (bundle: " + bundle + "_e)")

    def test_peakreg_output(self):
        bundles = exp_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/TOM/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/TOM/" + bundle + ".nii.gz").get_data()
            # images_equal = np.allclose(img_ref, img_new, rtol=1e-5, atol=1e-5)  # because of floats small tolerance margin needed  #too low
            images_equal = np.allclose(img_ref, img_new, rtol=1e-4, atol=1e-4)  # because of floats small tolerance margin needed
            self.assertTrue(images_equal, "TOMs are not correct (bundle: " + bundle + ")")

    # def test_FA(self):
    #     img_ref = nib.load("tests/reference_files/FA.nii.gz").get_data()
    #     img_new = nib.load("examples/FA.nii.gz").get_data()
    #     images_equal = np.allclose(img_ref, img_new, rtol=1e-6, atol=1e-6)
    #     self.assertTrue(images_equal, "FA not correct")

    def test_tractometry_toy_example(self):
        ref = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7, 0.35])
        new = np.loadtxt("tractometry_toy_example/Tractometry.csv", delimiter=";", skiprows=1).transpose()

        print("NEW")
        print(new)

        arrays_equal = np.array_equal(ref, new)
        self.assertTrue(arrays_equal, "Tractometry toy example not correct")

    def test_tractometry(self):
        ref = np.loadtxt("tests/reference_files/Tractometry_2k.csv", delimiter=";", skiprows=1).transpose()
        new = np.loadtxt("examples/Tractometry.csv", delimiter=";", skiprows=1).transpose()
        # Because tracking is a stochastic process the results are not always the same. Check if they are the same
        #   within a certain margin.
        # mean_difference = abs((ref-new).mean())
        # arrays_equal = mean_difference < 0.003
        # arrays_equal = np.allclose(ref, new, rtol=0, atol=2e-1)  #allow error of around 0.2      #1k fibers
        arrays_equal = np.allclose(ref, new, rtol=3e-2, atol=3e-2)  #allow error of around 0.03      #2k fibers
        # arrays_equal = np.allclose(ref, new, rtol=9e-3, atol=9e-3)    #allow error of around 0.009     #10k fibers
        self.assertTrue(arrays_equal, "Tractometry not correct")



if __name__ == '__main__':
    unittest.main()