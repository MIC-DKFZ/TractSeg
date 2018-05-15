import unittest
from tractseg.libs.ExpUtils import ExpUtils

class test_basic(unittest.TestCase):

    def setUp(self):
        pass

    def test_set_two(self):
        self.assertEqual((3*4), 12)

    def test_set_one(self):
        bundles = ExpUtils.get_bundle_names("CST_right")
        self.assertListEqual(bundles, ["BG", "CST_right"], "Error in list of bundle names")


# if __name__ == '__main__':
#     unittest.main()