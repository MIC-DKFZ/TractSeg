import unittest
from tractseg.libs import exp_utils

class test_functions(unittest.TestCase):

    def setUp(self):
        pass

    def test_bundle_names(self):
        bundles = exp_utils.get_bundle_names("CST_right")
        self.assertListEqual(bundles, ["BG", "CST_right"], "Error in list of bundle names")

if __name__ == '__main__':
    unittest.main()