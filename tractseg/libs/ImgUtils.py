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

import numpy as np
import nibabel as nib
from scipy import ndimage
from os.path import join
from tractseg.libs.Config import Config as C
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.Utils import Utils
from scipy.ndimage.morphology import binary_dilation

class ImgUtils:
    
    def __init__(self):
        return None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        elif dataset == "Phantom" and resolution == "1.25mm":
            # Size (145,174,145)
            return np.array([[-1.24138, 0.,   0.,   90.],
                             [0., 1.24138,    0.,   -126.],
                             [0.,    0., 1.24138,   -72.],
                             [0.,    0.,      0.,    1.]])

        elif dataset == "Phantom" and resolution == "2mm":
            # Size (145,174,145)
            return np.array([[-2., 0.,   0.,   90.],
                             [0., 2.,    0.,   -126.],
                             [0.,  0.,    2.,   -72.],
                             [0.,  0.,      0.,    1.]])

        elif dataset == "TRACED" and resolution == "2.5mm":
            #Results in mask that fit to Training data (but not to original Challenge image)
            # -> flip x and multiply affine x with *-1 => then fits to challenge
            # Size (78,93,75)
            return np.array([[-2.5, 0.,  0.,   -94.],
                             [0.,  2.5, 0.,  -134.],
                             [0.,  0.,  2.5,  -72.],
                             [0.,  0.,  0.,   1.]])

        else:
            raise ValueError("No Affine defined for this dataset and resolution !!")

    @staticmethod
    def get_dwi_spacing(dataset, resolution):
        if dataset == "HCP" and resolution == "1.25mm":
            return (145,174,145)

        elif dataset == "HCP_32g" and resolution == "1.25mm":
            return(145,174,145)

        elif (dataset == "HCP_32g" or dataset == "HCP_2mm") and resolution == "2mm":
            return (90,108,90)

        elif (dataset == "HCP_32g" or dataset == "HCP" or dataset == "HCP_2.5mm") and resolution == "2.5mm":
            return (73,87,73)

        elif dataset == "Phantom" and resolution == "1.25mm":
            return (145,174,145)

        elif dataset == "Phantom" and resolution == "2mm":
            return (90,108,90)

        elif dataset == "TRACED" and resolution == "2.5mm":
            return (78,93,75)

        else:
            raise ValueError("No Affine defined for this dataset and resolution !!")

    @staticmethod
    def remove_small_blobs(img, threshold=1, debug=True):
        '''
        Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
        This can be used for postprocessing.

        :param img: 3D Image
        :param threshold:
        :return:
        '''
        # mask, number_of_blobs = ndimage.label(img, structure=np.ones((3, 3, 3)))  #Also considers diagonal elements for determining if a element belongs to a blob
        mask, number_of_blobs = ndimage.label(img)
        print('Number of blobs before: ' + str(number_of_blobs))
        counts = np.bincount(mask.flatten())  # number of pixels in each blob
        if debug:
            print(counts)

        remove = counts <= threshold
        remove_idx = np.nonzero(remove)[0]  # somehow returns tupple with 1 value -> remove tupple, only keep value

        for idx in remove_idx:
            mask[mask == idx] = 0  # set blobs we remove to 0
        mask[mask > 0] = 1  # set everything else to 1

        if debug:
            mask_after, number_of_blobs_after = ndimage.label(mask)
            print('Number of blobs after: ' + str(number_of_blobs_after))

        return mask

    @staticmethod
    def resize_first_three_dims(img, order=0, zoom=0.62):
        img_sm = []
        for grad in range(img.shape[3]):
            #order: The order of the spline interpolation
            img_sm.append(ndimage.zoom(img[:, :, :, grad], zoom, order=order))  # order=0 -> nearest interpolation; order=1 -> linear or bilinear interpolation?
        img_sm = np.array(img_sm)
        return img_sm.transpose(1, 2, 3, 0)  # grads channel was in front -> put to back

    @staticmethod
    def create_multilabel_mask(HP, subject, labels_type=np.int16, dataset_folder="HCP", labels_folder="bundle_masks"):
        '''
        One-hot encoding of all bundles in one big image
        :param subject:
        :return: image of shape (x, y, z, nr_of_bundles + 1)
        '''
        bundles = ExpUtils.get_bundle_names(HP.CLASSES)

        #Masks sind immer HCP_highRes (später erst downsample)
        mask_ml = np.zeros((145, 174, 145, len(bundles)))
        background = np.ones((145, 174, 145))   # everything that contains no bundle

        for idx, bundle in enumerate(bundles[1:]):   #first bundle is background -> already considered by setting np.ones in the beginning
            mask = nib.load(join(C.HOME, dataset_folder, subject, labels_folder, bundle + ".nii.gz"))
            mask_data = mask.get_data()     # dtype: uint8
            mask_ml[:, :, :, idx+1] = mask_data
            background[mask_data == 1] = 0    # remove this bundle from background

        mask_ml[:, :, :, 0] = background
        return mask_ml.astype(labels_type)

    @staticmethod
    def save_multilabel_img_as_multiple_files(HP, img, affine, path, name="tract_segmentations"):
        bundles = ExpUtils.get_bundle_names(HP.CLASSES)[1:]
        for idx, bundle in enumerate(bundles):
            img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
            ExpUtils.make_dir(join(path, name))
            nib.save(img_seg, join(path, name, bundle + ".nii.gz"))

    @staticmethod
    def save_multilabel_img_as_multiple_files_peaks(HP, img, affine, path):
        bundles = ExpUtils.get_bundle_names(HP.CLASSES)[1:]
        for idx, bundle in enumerate(bundles):
            data = img[:, :, :, (idx*3):(idx*3)+3]

            if HP.FLIP_OUTPUT_PEAKS:
                data[:, :, :, 2] *= -1  # flip z Axis for correct view in MITK
                filename = bundle + "_f.nii.gz"
            else:
                filename = bundle + ".nii.gz"

            img_seg = nib.Nifti1Image(data, affine)
            ExpUtils.make_dir(join(path, "TOM"))
            nib.save(img_seg, join(path, "TOM", filename))

    @staticmethod
    def save_multilabel_img_as_multiple_files_endings(HP, img, affine, path):
        bundles = ExpUtils.get_bundle_names(HP.CLASSES)[1:]
        for idx, bundle in enumerate(bundles):
            img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
            ExpUtils.make_dir(join(path, "endings_segmentations"))
            nib.save(img_seg, join(path, "endings_segmentations", bundle + ".nii.gz"))

    @staticmethod
    def save_multilabel_img_as_multiple_files_endings_OLD(HP, img, affine, path, multilabel=True):
        '''
        multilabel True:    save as 1 and 2 without fourth dimension
        multilabel False:   save with beginnings and endings combined
        '''
        # bundles = ExpUtils.get_bundle_names("20")[1:]
        bundles = ExpUtils.get_bundle_names(HP.CLASSES)[1:]
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
            ExpUtils.make_dir(join(path, "endings"))
            nib.save(img_seg, join(path, "endings", bundle + ".nii.gz"))

    @staticmethod
    def peak_image_to_binary_mask(img, len_thr=0.1):
        '''

        :param img: [x,y,z,nr_bundles*3]
        :param len_thr:
        :return:
        '''
        peaks = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 3.), 3))
        peaks_len = np.linalg.norm(peaks, axis=-1)
        return peaks_len > len_thr

    @staticmethod
    def remove_small_peaks(img, len_thr=0.1):
        peaks = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], int(img.shape[3] / 3.), 3))

        peaks_len = np.linalg.norm(peaks, axis=-1)
        mask = peaks_len > len_thr

        peaks[~mask] = 0
        return np.reshape(peaks, (img.shape[0], img.shape[1], img.shape[2], img.shape[3]))

    @staticmethod
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
