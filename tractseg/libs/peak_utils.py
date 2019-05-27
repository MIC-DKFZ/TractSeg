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

from os.path import join
from os.path import dirname
from os.path import exists

import numpy as np
import nibabel as nib



def peak_image_to_binary_mask(img, len_thr=0.1):
    '''

    :param img: [x,y,z,nr_bundles*3]
    :param len_thr:
    :return:
    '''
    img = np.nan_to_num(img)    # can contains nan because directly called on original peaks sometimes
    peaks = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 3.), 3))
    peaks_len = np.linalg.norm(peaks, axis=-1)
    return peaks_len > len_thr


def remove_small_peaks(img, len_thr=0.1):
    peaks = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 3.), 3))

    peaks_len = np.linalg.norm(peaks, axis=-1)
    mask = peaks_len > len_thr

    peaks[~mask] = 0
    return np.reshape(peaks, (img.shape[0], img.shape[1], img.shape[2], img.shape[3]))


def remove_small_peaks_bundle_specific(img, bundles, len_thr=0.1):

    bundles_thresholds = {
        "CA": 0.1,
    }

    peaks = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 3.), 3))
    peaks_len = np.linalg.norm(peaks, axis=-1)

    for idx, bundle in enumerate(bundles):
        if bundle in bundles_thresholds:
            thr = bundles_thresholds[bundle]
        else:
            thr = len_thr
        mask = peaks_len[:,:,:,idx] > thr
        peaks[:,:,:,idx][~mask] = 0

    return np.reshape(peaks, (img.shape[0], img.shape[1], img.shape[2], img.shape[3]))


def normalize_peak_to_unit_length(peaks):
    """
    :param peaks: [x,y,z,3]  (only 1 peak allowed)
    :return: [x,y,z,3]
    """
    return peaks / (np.linalg.norm(peaks, axis=-1) + 1e-20)[..., None]



def peak_image_to_binary_mask_path(path_in, path_out, peak_length_threshold=0.1):
    """
    Create binary mask from a peak image.

    Args:
        path_in: Path of peak image
        path_out: Path of binary output image
        peak_length_threshold:

    Returns:

    """
    peak_img = nib.load(path_in)
    peak_data = peak_img.get_data()
    peak_mask = peak_image_to_binary_mask(peak_data, len_thr=peak_length_threshold)
    peak_mask_img = nib.Nifti1Image(peak_mask.astype(np.uint8), peak_img.get_affine())
    nib.save(peak_mask_img, path_out)


def peak_image_to_tensor_image(peaks):
    """
    Convert peak image to tensor image

    Args:
        peaks: shape: [x,y,z,nr_peaks*3]

    Returns:
        tensor with shape: [x,y,z, nr_peaks*6]
    """

    def peak_to_tensor(peak):
        tensor = np.zeros(peak.shape[:3] + (6,), dtype=np.float32)
        tensor[..., 0] = peak[..., 0] * peak[..., 0]
        tensor[..., 1] = peak[..., 0] * peak[..., 1]
        tensor[..., 2] = peak[..., 0] * peak[..., 2]
        tensor[..., 3] = peak[..., 1] * peak[..., 1]
        tensor[..., 4] = peak[..., 1] * peak[..., 2]
        tensor[..., 5] = peak[..., 2] * peak[..., 2]
        return tensor

    nr_peaks = int(peaks.shape[3] / 3)
    tensor = np.zeros(peaks.shape[:3] + (nr_peaks * 6,), dtype=np.float32)
    for idx in range(nr_peaks):
        tensor[..., idx*6:(idx*6)+6] = peak_to_tensor(peaks[..., idx*3:(idx*3)+3])
    return tensor

def peak_image_to_tensor_image_nifti(peaks_img):
    """
    Same as peak_image_to_tensor_image() but takes nifti img as input and outputs a nifti img
    """
    tensors = peak_image_to_tensor_image(peaks_img.get_data())
    return nib.Nifti1Image(tensors, peaks_img.get_affine())

def load_bedpostX_dyads(path_dyads1, scale=True):
    """
    Load bedpostX dyads (following the default naming convention)

    Args:
        path_dyads1: path to dyads1.nii.gz

    Returns:
        peaks with shape: [x,y,z,9]
    """
    dyads1_img = nib.load(path_dyads1)
    dyads1 = dyads1_img.get_data()
    dyads2 = nib.load(join(dirname(path_dyads1), "dyads2_thr0.05.nii.gz")).get_data()
    dyads3_path = join(dirname(path_dyads1), "dyads3_thr0.05.nii.gz")
    if exists(dyads3_path):
        dyads3 = nib.load(dyads3_path).get_data()
    else:
        dyads3 = np.zeros(dyads2.shape, dtype=dyads2.dtype)

    if scale:
        dyads1 *= nib.load(join(dirname(path_dyads1), "mean_f1samples.nii.gz")).get_data()[...,None]
        dyads2 *= nib.load(join(dirname(path_dyads1), "mean_f2samples.nii.gz")).get_data()[...,None]
        f3_path = join(dirname(path_dyads1), "mean_f3samples.nii.gz")
        if exists(f3_path):
            dyads3 *= nib.load(f3_path).get_data()[...,None]
        else:
            dyads3 *= np.zeros(dyads2.shape[:3])[...,None]

    dyads = np.concatenate((dyads1, dyads2, dyads3), axis=3)

    # Flip x axis to make BedpostX compatible with mrtrix CSD
    dyads[:, :, :, 0] *= -1
    dyads[:, :, :, 3] *= -1
    dyads[:, :, :, 6] *= -1

    dyads_img = nib.Nifti1Image(dyads, dyads1_img.get_affine())
    return dyads_img
