import unittest
from tractseg.libs.ExpUtils import ExpUtils

class test_functions(unittest.TestCase):

    def setUp(self):
        pass

    def test_bundle_names(self):
        bundles = ExpUtils.get_bundle_names("CST_right")
        self.assertListEqual(bundles, ["BG", "CST_right"], "Error in list of bundle names")
