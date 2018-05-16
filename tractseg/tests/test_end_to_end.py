import unittest
import nibabel as nib
import numpy as np

class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    # def test_tractseg_output(self):
    #     img_ref = nib.load("examples/Tests/bundle_segmentations.nii.gz").get_data()
    #     img_new = nib.load("examples/tractseg_output/bundle_segmentations.nii.gz").get_data()
    #     images_equal = np.array_equal(img_ref, img_new)
    #     self.assertTrue(images_equal, "Tract segmentations are not correct")

    def test_peakreg_output(self):
        img_ref = nib.load("examples/Tests/bundle_TOMs.nii.gz").get_data()
        img_new = nib.load("examples/tractseg_output/bundle_TOMs.nii.gz").get_data()
        images_equal = np.allclose(img_ref, img_new, rtol=1e-5, atol=1e-5)  # because of floats small tolerance margin needed
        self.assertTrue(images_equal, "TOMs are not correct")
