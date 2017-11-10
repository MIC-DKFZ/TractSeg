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

from warnings import warn
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class ReorderSegTransform(AbstractTransform):
    """
    Yields reordered seg (needed for DataAugmentation: x&y have to be last 2 dims and nr_classes must be before, for DataAugmentation to work)
    -> here we move it back to (bs, x, y, nr_classes) for easy calculating of f1
    """
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")

        if seg is None:
            warn("You used ReorderSegTransform but there is no 'seg' key in your data_dict, returning data_dict unmodified", Warning)
        else:
            seg = data_dict["seg"]  # (bs, nr_of_classes, x, y)
            data_dict["seg"] = seg.transpose(0, 2, 3, 1)  # (bs, x, y, nr_of_classes)
        return data_dict



