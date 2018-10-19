import unittest
import nibabel as nib
import numpy as np
from tractseg.libs.ExpUtils import ExpUtils

class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_csd_peaks(self):
        img_ref = nib.load("tests/reference_files/peaks.nii.gz").get_data()
        img_new = nib.load("examples/tractseg_output/peaks.nii.gz").get_data()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "CSD peaks not correct")

    def test_tractseg_output(self):
        bundles = ExpUtils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/bundle_segmentations/" + bundle + ".nii.gz").get_data()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, "Tract segmentations are not correct (bundle: " + bundle + ")")

    def test_endingsseg_output(self):
        bundles = ExpUtils.get_bundle_names("All")[1:]
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
        bundles = ExpUtils.get_bundle_names("All")[1:]
        for bundle in bundles:
            img_ref = nib.load("tests/reference_files/TOM/" + bundle + ".nii.gz").get_data()
            img_new = nib.load("examples/tractseg_output/TOM/" + bundle + ".nii.gz").get_data()
            images_equal = np.allclose(img_ref, img_new, rtol=1e-5, atol=1e-5)  # because of floats small tolerance margin needed
            self.assertTrue(images_equal, "TOMs are not correct (bundle: " + bundle + ")")

    def test_FA(self):
        img_ref = nib.load("tests/reference_files/FA.nii.gz").get_data()
        img_new = nib.load("examples/FA.nii.gz").get_data()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "FA not correct")

    def test_tractometry(self):
        #todo
        pass

if __name__ == '__main__':
    unittest.main()