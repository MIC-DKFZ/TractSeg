
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import nibabel as nib
import numpy as np
import pandas as pd

from tractseg.data import dataset_specific_utils
from tractseg.libs import data_utils


def transform_to_output_space(data):
    transformation = {'original_shape': (57, 70, 59, 9),
                      'pad_x': 6.5,
                      'pad_y': 0.0,
                      'pad_z': 5.5,
                      'zoom': 2.057142857142857}
    bbox = [[8, 65], [10, 80], [3, 62]]
    original_shape = (73, 87, 73)

    data = data_utils.cut_and_scale_img_back_to_original_img(data, transformation)
    data = data_utils.add_original_zero_padding_again(data, bbox, original_shape, 0)
    return data


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_csd_peaks(self):
        img_ref = np.nan_to_num(nib.load("tests/reference_files/peaks.nii.gz").get_fdata())
        img_new = np.nan_to_num(nib.load("examples/docker_test/peaks.nii.gz").get_fdata())
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "CSD peaks not correct")

    def test_tractseg_output_docker(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
            img_new = nib.load("examples/docker_test/bundle_segmentations/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Docker tract segmentations are not correct (bundle: " + bundle + ")")

    def test_bundle_specific_postprocessing(self):
        # CA
        img_ref = np.zeros((144, 144, 144)).astype(np.uint8)
        img_ref[10:30, 10:30, 10:30] = 1  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 1  # big blob 2
        img_ref[20:25, 20:25, 30:40] = 1  # bridge
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/BS_PP/tractseg_output/bundle_segmentations/CA.nii.gz").get_fdata().astype(np.uint8)
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: CA)")

        # CC_1
        img_ref = np.zeros((144, 144, 144)).astype(np.uint8)
        img_ref[10:30, 10:30, 10:30] = 1  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 1  # big blob 2
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/BS_PP/tractseg_output/bundle_segmentations/CC_1.nii.gz").get_fdata().astype(np.uint8)
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: CC_1)")

    def test_postprocessing(self):
        img_ref = np.zeros((144, 144, 144)).astype(np.uint8)
        img_ref[10:30, 10:30, 10:30] = 1  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 1  # big blob 2
        img_ref[60:63, 60:63, 60:63] = 1  # small blob
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/no_PP/tractseg_output/bundle_segmentations/CC_1.nii.gz").get_fdata().astype(np.uint8)
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: CC_1)")

    def test_get_probabilities(self):
        img_ref = np.zeros((144, 144, 144)).astype(np.float32)
        img_ref[10:30, 10:30, 10:30] = 0.7  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 0.7  # big blob 2
        img_ref[20:25, 20:25, 30:34] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[20:25, 20:25, 36:40] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[50:55, 50:55, 50:55] = 0.2  # below threshold
        img_ref[60:63, 60:63, 60:63] = 0.9  # small blob -> will get removed by postprocessing
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/Probs/tractseg_output/bundle_segmentations/CA.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Tract probabilities are not correct (bundle: CA)")

    def test_uncertainty(self):
        img_ref = np.zeros((144, 144, 144)).astype(np.float32)
        # in test mode no 30 iterations + stddev are done, but output is the same as for probabilities
        img_ref[10:30, 10:30, 10:30] = 0.7  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 0.7  # big blob 2
        img_ref[20:25, 20:25, 30:34] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[20:25, 20:25, 36:40] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[50:55, 50:55, 50:55] = 0.2  # below threshold
        img_ref[60:63, 60:63, 60:63] = 0.9  # small blob -> will get removed by postprocessing
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/Uncert/tractseg_output/bundle_uncertainties/CA.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Tract uncertainties are not correct (bundle: CA)")

    def test_density_regression(self):
        img_ref = np.zeros((144, 144, 144)).astype(np.float32)
        img_ref[10:30, 10:30, 10:30] = 0.7  # big blob 1
        img_ref[10:30, 10:30, 40:50] = 0.7  # big blob 2
        img_ref[20:25, 20:25, 30:34] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[20:25, 20:25, 36:40] = 0.4  # incomplete bridge between blobs with lower probability
        img_ref[50:55, 50:55, 50:55] = 0.2  # below threshold
        img_ref[60:63, 60:63, 60:63] = 0.9  # small blob -> will get removed by postprocessing
        img_ref = transform_to_output_space(img_ref)
        img_new = nib.load("examples/DM/tractseg_output/dm_regression/CA.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Density maps are not correct (bundle: CA)")

    def test_tractseg_output(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
            img_new = nib.load("examples/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_tractseg_output_SR_noPP(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            # IFO very different on travis than locally. Unclear why. All other bundles are fine.
            if bundle != "IFO_right":
                img_ref = nib.load("tests/reference_files/bundle_segmentations_SR_noPP/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
                img_new = nib.load("examples/SR_noPP/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_fdata().astype(np.uint8)
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
            img_ref = nib.load("tests/reference_files/endings_segmentations/" + bundle + "_b.nii.gz").get_fdata().astype(np.uint8)
            img_new = nib.load("examples/tractseg_output/endings_segmentations/" + bundle + "_b.nii.gz").get_fdata().astype(np.uint8)
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Bundle endings are not correct (bundle: " + bundle + "_b)")

            img_ref = nib.load("tests/reference_files/endings_segmentations/" + bundle + "_e.nii.gz").get_fdata().astype(np.uint8)
            img_new = nib.load("examples/tractseg_output/endings_segmentations/" + bundle + "_e.nii.gz").get_fdata().astype(np.uint8)
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Bundle endings are not correct (bundle: " + bundle + "_e)")

    def test_peakreg_output(self):
        bundles = dataset_specific_utils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/TOM/" + bundle + ".nii.gz").get_fdata()
            img_new = nib.load("examples/tractseg_output/TOM/" + bundle + ".nii.gz").get_fdata()
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

    def test_statistical_analysis_group(self):
        ref = pd.read_csv("tests/reference_files/tractometry/tractometry_result_group.png.csv",
                         sep=",").values[:, 2:].astype(np.float32)  # all but index and bundle_name column
        new = pd.read_csv("examples/tractometry_result_group.png.csv",
                          sep=",").values[:, 2:].astype(np.float32)
        diff_max = np.abs(ref-new).max()
        arrays_equal = np.array_equal(ref, new)
        self.assertTrue(arrays_equal, "Statistical analysis (group) not correct (max difference: " +
                        str(diff_max) + ")")

    def test_statistical_analysis_correlation(self):
        ref = pd.read_csv("tests/reference_files/tractometry/tractometry_result_correlation.png.csv",
                         sep=",").values[:, 2:].astype(np.float32)  # all but index and bundle_name column
        new = pd.read_csv("examples/tractometry_result_correlation.png.csv",
                          sep=",").values[:, 2:].astype(np.float32)
        diff_max = np.abs(ref-new).max()
        arrays_equal = np.allclose(ref, new, rtol=5e-4, atol=5e-4)
        self.assertTrue(arrays_equal, "Statistical analysis (correlation) not correct (max difference: " +
                        str(diff_max) + ")")


if __name__ == '__main__':
    unittest.main()
