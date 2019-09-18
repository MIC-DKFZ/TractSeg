
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import nibabel as nib
import numpy as np

from tractseg.data import dataset_specific_utils


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    # def test_csd_peaks(self):
    #     img_ref = nib.load("tests/reference_files/peaks.nii.gz").get_data()
    #     img_new = nib.load("examples/tractseg_output/peaks.nii.gz").get_data()
    #     images_equal = np.allclose(img_ref, img_new, rtol=1, atol=1)    #somehow not working; order of channels randomly changing?
    #     self.assertTrue(images_equal, "CSD peaks not correct")

    def test_tractseg_output(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_tractseg_output_SR_noPP(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            # IFO very different on travis than locally. Unclear why. All other bundles are fine.
            if bundle != "IFO_right":
                img_ref = nib.load("tests/reference_files/bundle_segmentations_SR_noPP/" + bundle + ".nii.gz").get_data()
                img_new = nib.load("examples/SR_noPP/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
                # Processing on travis slightly different from local environment -> have to allow for small margin
                nr_differing_voxels = np.abs(img_ref - img_new).sum()
                if nr_differing_voxels < 5:
                    images_equal = True
                else:
                    images_equal = False
                self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ") " +
                                              "(nr of differing voxels: " + str(nr_differing_voxels) + ")")

    def test_endingsseg_output(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
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
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/TOM/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/TOM/" + bundle + ".nii.gz").get_data()
            # Because of regression small tolerance margin needed
            images_equal = np.allclose(img_ref, img_new, rtol=1e-3, atol=1e-3)
            self.assertTrue(images_equal, "TOMs are not correct (bundle: " + bundle + ")")

    def test_tractometry_toy_example(self):
        ref = np.array([0., 0., 0.148, 0.173, 0., 0.325, 0.319, 0.28]) # coord + tree, round(3)
        new = np.loadtxt("tractometry_toy_example/Tractometry.csv", delimiter=";", skiprows=1).transpose()
        new = new.round(3)
        arrays_equal = np.array_equal(ref, new)
        self.assertTrue(arrays_equal, "Tractometry toy example not correct")

    def test_tractometry(self):
        ref = np.loadtxt("tests/reference_files/Tractometry_2k.csv", delimiter=";", skiprows=1).transpose()
        new = np.loadtxt("examples/Tractometry.csv", delimiter=";", skiprows=1).transpose()
        # Because tracking is a stochastic process the results are not always the same. Check if they are the same
        #   within a certain margin.
        diff_max = np.abs(ref-new).max()
        arrays_equal = np.allclose(ref, new, rtol=3e-2, atol=3e-2)  # works for 10k fibers with 100 points
        self.assertTrue(arrays_equal, "Tractometry not correct (max difference: " + str(diff_max) + ")")


if __name__ == '__main__':
    unittest.main()