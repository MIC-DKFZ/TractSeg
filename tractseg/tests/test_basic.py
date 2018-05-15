import unittest
from tractseg.libs.ExpUtils import ExpUtils
import nibabel as nib
import numpy as np

class test_basic(unittest.TestCase):

    def setUp(self):
        pass

    def test_set_two(self):
        self.assertEqual((3*4), 12)

    def test_set_one(self):
        bundles = ExpUtils.get_bundle_names("CST_right")
        self.assertListEqual(bundles, ["BG", "CST_right"], "Error in list of bundle names")

    def test_images_equal(self):
        img_ref = nib.load("examples/example_output.nii.gz").get_data()
        img_new = nib.load("bundle_segmentations.nii.gz").get_data()

        images_equal = np.array_equal(img_ref, img_new)

        self.assertTrue(images_equal, "Images are not equal")

# if __name__ == '__main__':
#     unittest.main()