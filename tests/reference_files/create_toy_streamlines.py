#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join

import nibabel as nib
import numpy as np

from tractseg.libs import fiber_utils
from tractseg.libs import exp_utils


def main():

    args = sys.argv[1:]
    out_dir = args[0]

    exp_utils.make_dir(join(out_dir, "endings_segmentations"))
    exp_utils.make_dir(join(out_dir, "TOM_trackings"))

    affine = np.eye(4)

    data = [
            [[0.9, 0.5, 0.9],
             [0.5, 0.9, 0.5],
             [0.9, 0.5, 0.9]],

            [[0.5, 0.9, 0.5],
             [0.9, 0.5, 0.9],
             [0.5, 0.9, 0.5]],

            [[0.9, 0.5, 0.9],
             [0.5, 0.9, 0.5],
             [0.9, 0.5, 0.9]],
            ]
    data = np.array(data)
    data[0, 0, 0] = 0.1
    data[2, 2, 2] = 0.3
    data[0, 2, 2] = 0.4
    img = nib.Nifti1Image(data, affine)
    nib.save(img, join(out_dir, "toy_FA.nii.gz"))

    mask = np.zeros((3, 3, 3))
    mask[0, 0, 0] = 1
    img = nib.Nifti1Image(mask, affine)
    nib.save(img, join(out_dir, "endings_segmentations", "toy_b.nii.gz"))

    sl1 = np.array([[0., 0., 0.], [2., 2., 2.]])
    sl2 = np.array([[0., 2., 2.], [0., 0., 0.]])
    streamlines = [sl1, sl2]

    # Have to substract 0.5 to move from convention "0mm is in voxel corner" to convention "0mm is in voxel center"
    # We have to do this because nifti uses convention "0mm is in voxel center" (streamlines are in world space,
    # but edge of first voxel of nifti is not at 0,0,0 but at -0.5,-0.5,-0.5). If we do not apply this results
    # will be displayed incorrectly in image viewers (e.g. MITK) and dipy functions (e.g. near_roi) will give wrong results.
    streamlines = fiber_utils.add_to_each_streamline(streamlines, -0.5)

    fiber_utils.save_streamlines_as_trk_legacy(join(out_dir, "TOM_trackings", "toy.trk"),
                                               streamlines, affine, data.shape)
    # fiber_utils.save_streamlines(join(out_dir, "TOM_trackings", "toy.trk"),
    #                             streamlines, affine, data.shape)

    # tractometry_result = [np.mean([data[0, 0, 0], data[0, 0, 0]]),
    #                       np.mean([data[2, 2, 2], data[0, 2, 2]])]
    # print("asserted result: {}".format(tractometry_result))  # [0.1, 0.35]
    # 10 points: [0.1  0.1  0.1  0.1  0.1  0.7  0.7  0.7  0.7  0.35]


if __name__ == '__main__':
    main()

