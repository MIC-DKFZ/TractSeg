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
from pkg_resources import resource_filename
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from sklearn.externals import joblib

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import exp_utils


def pad_3d_image(image, pad_size, pad_value=None):
    '''
    :param pad_size: must be a np array with 3 entries, one for each dimension of the image

    IMPORTANT: numbers in pad_size should be even numbers; they are always divided by 2 and then rounded to floor !
        -> pad 3 -> would result in padding_left: 1 and padding_right: 1  => 1 gets lost
    '''
    image_shape = image.shape
    new_shape = np.array(list(image_shape)) + pad_size
    if pad_value is None:
        pad_value = image[0, 0, 0]
    new_image = np.ones(new_shape) * pad_value
    new_image[int(pad_size[0]/2.) : int(pad_size[0]/2.)+image_shape[0],
              int(pad_size[1]/2.) : int(pad_size[1]/2.)+image_shape[1],
              int(pad_size[2]/2.) : int(pad_size[2]/2.)+image_shape[2]] = image
    return new_image


def pad_4d_image(image, pad_size, pad_value=None):
    '''
    :param pad_size: must be a np array with 4 entries, one for each dimension of the image

    IMPORTANT: numbers in pad_size should be even numbers; they are always divided by 2 and then rounded to floor !
        -> pad 3 -> would result in padding_left: 1 and padding_right: 1  => 1 gets lost
    '''
    image_shape = image.shape
    new_shape = np.array(list(image_shape)) + pad_size
    if pad_value is None:
        pad_value = image[0, 0, 0, 0]
    new_image = np.ones(new_shape) * pad_value
    new_image[int(pad_size[0]/2.) : int(pad_size[0]/2.) + image_shape[0],
              int(pad_size[1]/2.) : int(pad_size[1]/2.) + image_shape[1],
              int(pad_size[2]/2.) : int(pad_size[2]/2.) + image_shape[2],
              int(pad_size[3]/2.) : int(pad_size[3]/2.) + image_shape[3]] = image
    return new_image


def pad_4d_image_left(image, pad_size, new_shape, pad_value=None):
    '''
    This can be used if we want to pad by uneven numbers; you specify padding on the right side of
    each dimension (left is then automatically filled up).

    :param pad_size: must be a np array with 4 entries, one for each dimension of the image
    '''
    image_shape = image.shape
    # new_shape = np.array(list(image_shape)) + pad_size
    new_shape = np.array(new_shape)
    if pad_value is None:
        pad_value = image[0, 0, 0, 0]
    new_image = np.ones(new_shape).astype(image.dtype) * pad_value
    new_image[int(pad_size[0]) : int(pad_size[0]) + image_shape[0],
              int(pad_size[1]) : int(pad_size[1]) + image_shape[1],
              int(pad_size[2]) : int(pad_size[2]) + image_shape[2],
              int(pad_size[3]) : int(pad_size[3]) + image_shape[3]] = image
    return new_image


def get_dwi_affine(dataset, resolution):
    #Info: Minus bei x und y invers gegenüber dem finalen Ergebnis (wie in MITK sehe), weil Dipy x und y mit -1 noch
    #       multipliziert

    if dataset == "HCP" and resolution == "1.25mm":
        # Size (145,174,145)
        return np.array([[-1.25, 0.,  0.,   90.],
                         [0., 1.25,   0.,  -126.],
                         [0.,    0., 1.25, -72.],
                         [0.,    0.,  0.,   1.]])

    elif dataset == "HCP_32g" and resolution == "1.25mm":
        # Size (145,174,145)
        return np.array([[-1.25, 0.,  0.,   90.],
                         [0., 1.25,   0.,  -126.],
                         [0.,    0., 1.25, -72.],
                         [0.,    0.,  0.,   1.]])

    elif (dataset == "HCP_32g" or dataset == "HCP_2mm") and resolution == "2mm":
        # Size (90,108,90)
        return np.array([[-2., 0.,  0.,   90.],
                         [0.,  2.,  0.,  -126.],
                         [0.,  0.,  2.,  -72.],
                         [0.,  0.,  0.,   1.]])

    elif (dataset == "HCP" or dataset == "HCP_32g" or dataset == "HCP_2.5mm") and resolution == "2.5mm":
        # Size (73,87,73)
        return np.array([[-2.5, 0.,  0.,   90.],
                         [0.,  2.5,  0.,  -126.],
                         [0.,  0.,  2.5,  -72.],
                         [0.,  0.,  0.,    1.]])

    else:
        raise ValueError("No Affine defined for this dataset and resolution !!")


def remove_small_blobs(img, threshold=1, debug=True):
    '''
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.

    :param img: 3D Image
    :param threshold:
    :return:
    '''
    # Also considers diagonal elements for determining if a element belongs to a blob
    # mask, number_of_blobs = ndimage.label(img, structure=np.ones((3, 3, 3)))
    mask, number_of_blobs = ndimage.label(img)
    if debug:
        print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    #If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1:
        return img

    # Find largest blob, to make sure we do not remove everything
    #   Largest blob is actually the second largest, because largest is the background
    second_largest_blob_value = np.sort(counts)[-2]
    second_largest_blob_idx = np.where(counts==second_largest_blob_value)
    if debug:
        print(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]  # somehow returns tupple with 1 value -> remove tupple, only keep value

    for idx in remove_idx:
        if idx != second_largest_blob_idx:  # make sure to keep at least one blob
            mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        mask_after, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def postprocess_segmentations(data, blob_thr=50, hole_closing=2):
    '''
    Postprocessing of segmentations. Fill holes and remove small blobs.

    :param data: 4D ndarray
    :param blob_thr:
    :param hole_closing:
    :return:
    '''
    nr_classes = data.shape[3]
    data_new = []
    for idx in range(nr_classes):
        data_single = data[:,:,:,idx]

        #Fill holes
        if hole_closing is not None:
            size = hole_closing  # Working as expected (size 2-3 good value)
            data_single = ndimage.binary_closing(data_single,
                                                 structure=np.ones((size, size, size))).astype(data_single.dtype)

        # Remove small blobs
        if blob_thr is not None:
            data_single = remove_small_blobs(data_single, threshold=blob_thr, debug=False)

        data_new.append(data_single)
    data_new = np.array(data_new).transpose(1, 2, 3, 0)
    return data_new


def resize_first_three_dims(img, order=0, zoom=0.62):
    """
    Runtime 35ms
    """
    img_sm = []
    for grad in range(img.shape[3]):
        #order: The order of the spline interpolation
        #  order=0 -> nearest interpolation; order=1 -> linear or bilinear interpolation?
        img_sm.append(ndimage.zoom(img[:, :, :, grad], zoom, order=order))
    img_sm = np.array(img_sm)
    return img_sm.transpose(1, 2, 3, 0)  # grads channel was in front -> put to back

def resize_first_three_dims_NUMPY(img, order=0, zoom=0.62):
    """
    Version of resize_first_three_dims using direct numpy indexing for storing results to test
    if it is faster. But it is not.
    Runtime 47ms
    """
    img_sm = None
    for grad in range(img.shape[3]):
        #order: The order of the spline interpolation
        #  order=0 -> nearest interpolation; order=1 -> linear or bilinear interpolation?
        grad_sm = ndimage.zoom(img[:, :, :, grad], zoom, order=order)
        if grad == 0:
            img_sm = np.zeros((grad_sm.shape[0], grad_sm.shape[1], grad_sm.shape[2], img.shape[3]))
        img_sm[:, :, :, grad] = grad_sm
    return img_sm

def create_multilabel_mask(Config, subject, labels_type=np.int16, dataset_folder="HCP", labels_folder="bundle_masks"):
    '''
    One-hot encoding of all bundles in one big image
    :param subject:
    :return: image of shape (x, y, z, nr_of_bundles + 1)
    '''
    bundles = exp_utils.get_bundle_names(Config.CLASSES)

    #Masks sind immer HCP_highRes (später erst downsample)
    mask_ml = np.zeros((145, 174, 145, len(bundles)))
    background = np.ones((145, 174, 145))   # everything that contains no bundle

    # first bundle is background -> already considered by setting np.ones in the beginning
    for idx, bundle in enumerate(bundles[1:]):
        mask = nib.load(join(C.HOME, dataset_folder, subject, labels_folder, bundle + ".nii.gz"))
        mask_data = mask.get_data()     # dtype: uint8
        mask_ml[:, :, :, idx+1] = mask_data
        background[mask_data == 1] = 0    # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml.astype(labels_type)


def save_multilabel_img_as_multiple_files(Config, img, affine, path, name="bundle_segmentations"):
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
        exp_utils.make_dir(join(path, name))
        nib.save(img_seg, join(path, name, bundle + ".nii.gz"))


def save_multilabel_img_as_multiple_files_peaks(Config, img, affine, path, name="TOM"):
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        data = img[:, :, :, (idx*3):(idx*3)+3]

        if Config.FLIP_OUTPUT_PEAKS:
            data[:, :, :, 2] *= -1  # flip z Axis for correct view in MITK
            filename = bundle + "_f.nii.gz"
        else:
            filename = bundle + ".nii.gz"

        img_seg = nib.Nifti1Image(data, affine)
        exp_utils.make_dir(join(path, name))
        nib.save(img_seg, join(path, name, filename))


def save_multilabel_img_as_multiple_files_endings(Config, img, affine, path):
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
        exp_utils.make_dir(join(path, "endings_segmentations"))
        nib.save(img_seg, join(path, "endings_segmentations", bundle + ".nii.gz"))


def save_multilabel_img_as_multiple_files_endings_OLD(Config, img, affine, path, multilabel=True):
    '''
    multilabel True:    save as 1 and 2 without fourth dimension
    multilabel False:   save with beginnings and endings combined
    '''
    # bundles = exp_utils.get_bundle_names("20")[1:]
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        data = img[:, :, :, (idx * 2):(idx * 2) + 2] > 0

        multilabel_img = np.zeros(data.shape[:3])

        if multilabel:
            multilabel_img[data[:, :, :, 0]] = 1
            multilabel_img[data[:, :, :, 1]] = 2
        else:
            multilabel_img[data[:, :, :, 0]] = 1
            multilabel_img[data[:, :, :, 1]] = 1

        img_seg = nib.Nifti1Image(multilabel_img, affine)
        exp_utils.make_dir(join(path, "endings"))
        nib.save(img_seg, join(path, "endings", bundle + ".nii.gz"))


def simple_brain_mask(data):
    '''
    Simple brain mask (for peak image). Does not matter if has holes
    because for cropping anyways only take min and max.

    :param data: peak image [x,y,z,9]
    '''
    data_max = data.max(axis=3)
    mask = data_max > 0.01
    mask = binary_dilation(mask, iterations=1)
    return mask.astype(np.uint8)


def probs_to_binary_bundle_specific(seg, bundles):
    '''

    :param seg: [x,y,z,bundles]
    :param bundles:
    :return:
    '''
    assert len(bundles) == seg.shape[3], "dimensions seg and bundles do not match"

    bundles_thresholds = {
        "CA": 0.3,
        "CST_left": 0.4,
        "CST_right": 0.4,
        "FX_left": 0.4,
        "FX_right": 0.4,
    }

    # Thresholds optimised from Phantom
    # bundles_thresholds = {
    #     "CA": 0.3,
    #     "CC_1": 0.2,
    #     "CC_3": 0.3,
    #     "CC_7": 0.3,
    #     "CG_left": 0.3,
    #     "CG_right": 0.3,
    #     "CST_left": 0.4,
    #     "CST_right": 0.4,
    #     "MLF_left": 0.4,
    #     "MLF_right": 0.4,
    #     "FPT_left": 0.4,
    #     "FPT_right": 0.4,
    #     "FX_left": 0.2,
    #     "FX_right": 0.2,
    #     "ICP_left": 0.2,
    #     "ICP_right": 0.2,
    #     "ILF_left": 0.3,
    #     "ILF_right": 0.3,
    #     "MCP": 0.2,
    #     "POPT_left": 0.4,
    #     "POPT_right": 0.4,
    #     "SCP_left": 0.2,
    #     "SCP_right": 0.2,
    #     "SLF_I_left": 0.3,
    #     "SLF_I_right": 0.3,
    #     "SLF_III_left": 0.4,
    #     "SLF_III_right": 0.4,
    #     "UF_left": 0.2,
    #     "UF_right": 0.2,
    #     "T_PREF_left": 0.4,
    #     "T_PREF_right": 0.4,
    #     "T_PREM_left": 0.2,
    #     "T_PREM_right": 0.2,
    #     "T_PREC_left": 0.4,
    #     "T_PREC_right": 0.4,
    #     "T_POSTC_left": 0.3,
    #     "T_POSTC_right": 0.3,
    #     "ST_FO_left": 0.3,
    #     "ST_FO_right": 0.3,
    #     "ST_PREF_left": 0.4,
    #     "ST_PREF_right": 0.4,
    #     "ST_PREM_left": 0.3,
    #     "ST_PREM_right": 0.3,
    #     "ST_PREC_left": 0.4,
    #     "ST_PREC_right": 0.4,
    #     "ST_POSTC_left": 0.3,
    #     "ST_POSTC_right": 0.3,
    # }

    segs_binary = []
    for idx, bundle in enumerate(bundles):
        if bundle in bundles_thresholds:
            thr = bundles_thresholds[bundle]
        else:
            thr = 0.5
        segs_binary.append(seg[:,:,:,idx] > thr)

    return np.array(segs_binary).transpose(1, 2, 3, 0).astype(np.uint8)


def dilate_binary_mask(file_in, file_out, dilation=2):
    img = nib.load(file_in)
    data = img.get_data()

    for i in range(dilation):
        data = binary_dilation(data)

    data = data > 0.5

    img_out = nib.Nifti1Image(data.astype(np.uint8), img.affine)
    nib.save(img_out, file_out)


def peaks2fixel(peaks_file_in, fixel_dir_out):
    """
    Transform TOM peak file to mrtrix fixels format. That can then be transformed to spherical harmonics using
    fixel2sh.

    Args:
        peaks_file_in: (x,y,z,3)   (only 1 peak allowed per voxel)
        fixel_dir_out:

    Returns:
        Void
    """
    exp_utils.make_dir(fixel_dir_out)

    peaks_img = nib.load(peaks_file_in)
    peaks = peaks_img.get_data()
    s = peaks.shape

    directions = []
    index = np.zeros(list(s[:3]) + [2])
    amplitudes = []

    idx_ctr = 0
    for x in range(s[0]):
        for y in range(s[1]):
            for z in range(s[2]):
                peak = peaks[x, y, z]
                peak_len = np.linalg.norm(peak)
                if peak_len > 0:
                    peak_normalized = peak / (peak_len + 1e-20)
                    directions.append(peak_normalized)
                    amplitudes.append(peak_len)
                    index[x, y, z] = [1, idx_ctr]
                    idx_ctr += 1

    nib.save(nib.Nifti2Image(np.array(directions), np.eye(4)), join(fixel_dir_out, "directions.nii.gz"))
    nib.save(nib.Nifti2Image(index, peaks_img.affine), join(fixel_dir_out, "index.nii.gz"))
    nib.save(nib.Nifti2Image(np.array(amplitudes), np.eye(4)), join(fixel_dir_out, "amplitudes.nii.gz"))


def flip_peaks(data, axis="x"):
    if axis == "x":
        # flip x Axis  (9 channel image)  (3 peaks)
        data[:, :, :, 0] *= -1
        data[:, :, :, 3] *= -1
        data[:, :, :, 6] *= -1
    elif axis == "y":
        data[:, :, :, 1] *= -1
        data[:, :, :, 4] *= -1
        data[:, :, :, 7] *= -1
    elif axis == "z":
        data[:, :, :, 2] *= -1
        data[:, :, :, 5] *= -1
        data[:, :, :, 8] *= -1
    return data


def enforce_shape(data, target_shape=(91, 109, 91, 9)):
    '''
    Cut and pad image to have same shape as target_shape (adapts first 3 dimensions, all further dimensions
    have to be the same in data and target_shape).

    :param data:
    :param target_shape:
    :return:
    '''
    ss = data.shape  # source shape
    ts = target_shape  # target shape

    data_new = np.zeros(ts)

    # cut if too much
    if ss[0] > ts[0]:
        data = data[ss[0] - ts[0]:, :, :]
    if ss[1] > ts[1]:
        data = data[:, ss[1] - ts[1]:, :]
    if ss[2] > ts[2]:
        data = data[:, :, ss[2] - ts[2]:]

    # pad with zero if too small
    data_new[:data.shape[0], :data.shape[1], :data.shape[2]] = data
    return data_new


def change_spacing_4D(img_in, new_spacing=1.25):
    """
    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)

    :param img_in:
    :param new_spacing:
    :return:
    """
    from dipy.align.imaffine import AffineMap

    data = img_in.get_data()
    old_shape = data.shape
    img_spacing = abs(img_in.affine[0, 0])

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)
    new_affine[0, 0] = new_spacing if img_in.affine[0, 0] > 0 else -new_spacing
    new_affine[1, 1] = new_spacing if img_in.affine[1, 1] > 0 else -new_spacing
    new_affine[2, 2] = new_spacing if img_in.affine[2, 2] > 0 else -new_spacing

    new_shape = np.floor(np.array(img_in.get_data().shape) * (img_spacing / new_spacing))
    new_shape = new_shape[:3]  # drop last dim

    new_data = []
    for i in range(data.shape[3]):
        affine_map = AffineMap(np.eye(4),
                               new_shape, new_affine,
                               old_shape, img_in.affine
                               )
        #Generally nearest a bit better results than linear interpolation
        # res = affine_map.transform(data[:,:,:,i], interp="linear")
        res = affine_map.transform(data[:, :, :, i], interp="nearest")
        new_data.append(res)

    new_data = np.array(new_data).transpose(1, 2, 3, 0)
    img_new = nib.Nifti1Image(new_data, new_affine)

    return img_new


def flip_peaks_to_correct_orientation_if_needed(peaks_input, do_flip=False):
    '''
    We use a pretrained random forest classifier to detect if the orientation of the peak is the same
    orientation as the peaks used for training TractSeg. Otherwise detect along which axis they
    have to be flipped to have the right orientation and return the flipped peaks.

    :param peaks_input: nifti peak img
    :param do_flip: also return flipped data or only return if flip needed
    :return: 4D numpy array (flipped peaks), boolean if flip was done
    '''
    peaks = change_spacing_4D(peaks_input, new_spacing=2.).get_data()
    #shape the classifier has been trained with
    peaks = enforce_shape(peaks, target_shape=(91, 109, 91, 9))

    peaks_x = peaks[int(peaks.shape[0] / 2.), :, :, :]
    peaks_y = peaks[:, int(peaks.shape[1] / 2.), :, :]
    peaks_z = peaks[:, :, int(peaks.shape[2] / 2.), :]
    X = [list(peaks_x.flatten()) + list(peaks_y.flatten()) + list(peaks_z.flatten())]
    X = np.nan_to_num(X)

    random_forest_path = resource_filename('resources', 'random_forest_peak_orientation_detection.pkl')
    clf = joblib.load(random_forest_path)
    predicted_label = clf.predict(X)[0]
    # labels:
    #  ok: 0, x:1, y:2, z:3
    peaks_input_data = peaks_input.get_data()
    if do_flip:
        if predicted_label == 0:
            return peaks_input_data, None
        elif predicted_label == 1:
            return flip_peaks(peaks_input_data, axis="x"), "x"
        elif predicted_label == 2:
            return flip_peaks(peaks_input_data, axis="y"), "y"
        elif predicted_label == 3:
            return flip_peaks(peaks_input_data, axis="z"), "z"
    else:
        if predicted_label == 0:
            return peaks_input_data, None
        elif predicted_label == 1:
            return peaks_input_data, "x"
        elif predicted_label == 2:
            return peaks_input_data, "y"
        elif predicted_label == 3:
            return peaks_input_data, "z"


def get_image_spacing(img_path):
    img = nib.load(img_path)
    affine = img.affine
    return str(abs(round(affine[0, 0], 2)))


def scale_to_range(data, range=(0, 1)):
    return (range[1] - range[0]) * (data - data.min()) / (data.max() - data.min()) + range[0]
