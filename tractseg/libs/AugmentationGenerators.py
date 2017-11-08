# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)
import numpy as np

def reorder_seg_generator(generator):
    '''
    Yields reordered seg (needed for DataAugmentation: x&y have to be last 2 dims and nr_classes must be before, for DataAugmentation to work)
    -> here we move it back to (bs, x, y, nr_classes) for easy calculating of f1
    '''
    for data_dict in generator:
        assert "seg" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        y = data_dict["seg"]  # (bs, nr_of_classes, x, y)
        data_dict["seg"] = y.transpose(0, 2, 3, 1)  # (bs, x, y, nr_of_classes)
        yield data_dict
