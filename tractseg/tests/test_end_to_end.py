import unittest
import nibabel as nib
import numpy as np

class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_images_equal(self):
        img_ref = nib.load("examples/example_output.nii.gz").get_data()
        img_new = nib.load("tractseg_output/bundle_segmentations.nii.gz").get_data()

        images_equal = np.array_equal(img_ref, img_new)

        self.assertTrue(images_equal, "Images are not equal")
