
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tractseg.data import dataset_specific_utils


class test_functions(unittest.TestCase):

    def setUp(self):
        pass

    def test_bundle_names(self):
        bundles = dataset_specific_utils.get_bundle_names("CST_right")
        self.assertListEqual(bundles, ["BG", "CST_right"], "Error in list of bundle names")

if __name__ == '__main__':
    unittest.main()