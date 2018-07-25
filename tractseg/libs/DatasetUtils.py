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
from scipy import ndimage
from tractseg.libs.ImgUtils import ImgUtils

class DatasetUtils():

    @staticmethod
    def scale_input_to_unet_shape(img4d, dataset, resolution="1.25mm"):
        '''
        Scale input image to right isotropic resolution and pad/cut image to make it square to fit UNet input shape

        :param img4d: (x, y, z, userdefined)  (userdefined could be gradients or classes)
        :param resolution: "1.25mm" / "2mm" / "2.5mm"     results in UNet input shape of (144,144,144) or (80,80,80)
        :return: img with dim 1mm: (144,144,144,none) or 2mm: (80,80,80,none) or 2.5mm: (80,80,80,none)
                    (note: 2.5mm padded with more zeros to reach 80,80,80)
        '''

        if resolution == "1.25mm":
            if dataset == "HCP":  # (145,174,145)
                # no resize needed
                return img4d[1:, 15:159, 1:]  # (144,144,144)
            elif dataset == "HCP_32g":  # (73,87,73)
                # return img4d[1:, 15:159, 1:]  # (144,144,144) #OLD when HCP_32g was still 125mm
                img4d = ImgUtils.resize_first_three_dims(img4d, zoom=2)  # (146,174,146,none)
                img4d = img4d[:-1,:,:-1]  #remove one voxel that came from upsampling   #(145,174,145)
                return img4d[1:, 15:159, 1:]  # (144,144,144)
            elif dataset == "TRACED":  # (78,93,75)
                raise ValueError("resolution '1.25mm' not supported for dataset 'TRACED'")
            elif dataset == "Schizo":  # (91,109,91)
                img4d = ImgUtils.resize_first_three_dims(img4d, zoom=1.60)  # (146,174,146)
                return img4d[1:145, 15:159, 1:145]                                # (144,144,144)

        elif resolution == "2mm":
            if dataset == "HCP":  # (145,174,145)
                img4d = ImgUtils.resize_first_three_dims(img4d, zoom=0.62)  # (90,108,90)
                return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
            elif dataset == "HCP_32g":  # (145,174,145)
                img4d = ImgUtils.resize_first_three_dims(img4d, zoom=0.62)  # (90,108,90)
                return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
            elif dataset == "HCP_2mm":  # (90,108,90)
                # no resize needed
                return img4d[5:85, 14:94, 5:85, :]  # (80,80,80)
            elif dataset == "TRACED":  # (78,93,75)
                raise ValueError("resolution '2mm' not supported for dataset 'TRACED'")

        elif resolution == "2.5mm":
            if dataset == "HCP":  # (145,174,145)
                img4d = ImgUtils.resize_first_three_dims(img4d, zoom=0.5)  # (73,87,73,none)
                bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
                bg = bg + img4d[0,0,0,:] #make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
                bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
                return bg  # (80,80,80)
            elif dataset == "HCP_2.5mm":  # (73,87,73,none)
                #no resize needed
                bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
                bg = bg + img4d[0,0,0,:] #make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
                bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
                return bg  # (80,80,80)
            elif dataset == "HCP_32g":  # (73,87,73,none)
                bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
                bg = bg + img4d[0, 0, 0, :]  # make bg have same value as bg from original img  (this adds last dim of img4d to last dim of bg)
                bg[4:77, :, 4:77] = img4d[:, 4:84, :, :]
                return bg  # (80,80,80)
            elif dataset == "TRACED":  # (78,93,75)
                # no resize needed
                bg = np.zeros((80, 80, 80, img4d.shape[3])).astype(img4d.dtype)
                bg = bg + img4d[0, 0, 0, :]  # make bg have same value as bg from original img
                bg[1:79, :, 3:78, :] = img4d[:, 7:87, :, :]
                return bg  # (80,80,80)


    @staticmethod
    def scale_input_to_world_shape(img4d, dataset, resolution="1.25mm"):
        '''
        Scale input image to original resolution and pad/cut image to make it original size

        :param img4d: (x, y, z, userdefined)  (userdefined could be gradients or classes)
        :param resolution: "1.25mm" / "2mm" / "2.5mm"
        :return: img with original size
        '''

        if resolution == "1.25mm":
            if dataset == "HCP":  # (144,144,144)
                # no resize needed
                return ImgUtils.pad_4d_image_left(img4d, np.array([1,15,1,0]), [146,174,146,img4d.shape[3]], pad_value=0)  # (146, 174, 146, none)
            elif dataset == "HCP_32g":  # (144,144,144)
                # no resize needed
                return ImgUtils.pad_4d_image_left(img4d, np.array([1,15,1,0]), [146,174,146,img4d.shape[3]], pad_value=0)  # (146, 174, 146, none)
            elif dataset == "TRACED":  # (78,93,75)
                raise ValueError("resolution '1.25mm' not supported for dataset 'TRACED'")
            elif dataset == "Schizo":  # (144,144,144)
                img4d = ImgUtils.pad_4d_image_left(img4d, np.array([1,15,1,0]), [145,174,145,img4d.shape[3]], pad_value=0)  # (145, 174, 145, none)
                return ImgUtils.resize_first_three_dims(img4d, zoom=0.62)  # (91,109,91)

        elif resolution == "2mm":
            if dataset == "HCP":  # (80,80,80)
                return ImgUtils.pad_4d_image_left(img4d, np.array([5,14,5,0]), [90,108,90,img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
            elif dataset == "HCP_32g":  # (80,80,80)
                return ImgUtils.pad_4d_image_left(img4d, np.array([5,14,5,0]), [90,108,90,img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
            elif dataset == "HCP_2mm":  # (80,80,80)
                return ImgUtils.pad_4d_image_left(img4d, np.array([5,14,5,0]), [90,108,90,img4d.shape[3]], pad_value=0)  # (90, 108, 90, none)
            elif dataset == "TRACED":  # (78,93,75)
                raise ValueError("resolution '2mm' not supported for dataset 'TRACED'")

        elif resolution == "2.5mm":
            if dataset == "HCP":  # (80,80,80)
                img4d = ImgUtils.pad_4d_image_left(img4d, np.array([0,4,0,0]), [80,87,80,img4d.shape[3]], pad_value=0) # (80,87,80,none)
                return img4d[4:77,:,4:77, :] # (73, 87, 73, none)
            elif dataset == "HCP_2.5mm":  # (80,80,80)
                img4d = ImgUtils.pad_4d_image_left(img4d, np.array([0,4,0,0]), [80,87,80,img4d.shape[3]], pad_value=0)  # (80,87,80,none)
                return img4d[4:77,:,4:77,:]  # (73, 87, 73, none)
            elif dataset == "HCP_32g":  # ((80,80,80)
                img4d = ImgUtils.pad_4d_image_left(img4d, np.array([0, 4, 0, 0]), [80, 87, 80, img4d.shape[3]], pad_value=0)  # (80,87,80,none)
                return img4d[4:77, :, 4:77, :]  # (73, 87, 73, none)
            elif dataset == "TRACED":  # (80,80,80)
                img4d = ImgUtils.pad_4d_image_left(img4d, np.array([0,7,0,0]), [80,93,80,img4d.shape[3]],pad_value=0)  # (80,93,80,none)
                return img4d[1:79, :, 3:78, :]  # (78,93,75,none)

    @staticmethod
    def pad_and_scale_img_to_square_img(data, target_size=144):
        '''
        Expects 3D or 4D image as input.

        Does
        1. Pad image with 0 to make it square
            (if uneven padding -> adds one more px "behind" img; but resulting img shape will be correct)
        2. Scale image to UNet size (144, 144, 144)
        '''
        nr_dims = len(data.shape)
        assert (nr_dims >= 3 and nr_dims <= 4)

        shape = data.shape
        biggest_dim = max(shape)

        # Pad to make square
        if nr_dims == 4:
            new_img = np.zeros((biggest_dim, biggest_dim, biggest_dim, shape[3])).astype(data.dtype)
        else:
            new_img = np.zeros((biggest_dim, biggest_dim, biggest_dim)).astype(data.dtype)
        pad1 = (biggest_dim - shape[0]) / 2.
        pad2 = (biggest_dim - shape[1]) / 2.
        pad3 = (biggest_dim - shape[2]) / 2.
        new_img[int(pad1):int(pad1) + shape[0],
                int(pad2):int(pad2) + shape[1],
                int(pad3):int(pad3) + shape[2]] = data

        # Scale to right size
        zoom = float(target_size) / biggest_dim
        if nr_dims == 4:
            #use order=0, otherwise image values of a DWI will be quite different after downsampling and upsampling
            new_img = ImgUtils.resize_first_three_dims(new_img, order=0, zoom=zoom)
        else:
            new_img = ndimage.zoom(new_img, zoom, order=0)

        transformation = {
            "original_shape": shape,
            "pad_x": pad1,
            "pad_y": pad2,
            "pad_z": pad3,
            "zoom": zoom
        }

        return new_img, transformation

    @staticmethod
    def cut_and_scale_img_back_to_original_img(data, t):
        '''
        Undo the transformations done with pad_and_scale_img_to_square_img

        data: 3D or 4D image
        t: transformation dict
        '''
        nr_dims = len(data.shape)
        assert (nr_dims >= 3 and nr_dims <= 4)

        # Back to old size
        # use order=0, otherwise image values of a DWI will be quite different after downsampling and upsampling
        if nr_dims == 3:
            new_data = ndimage.zoom(data, (1. / t["zoom"]), order=0)
        elif nr_dims == 4:
            new_data = ImgUtils.resize_first_three_dims(data, order=0, zoom=(1. / t["zoom"]))

        x_residual = 0
        y_residual = 0
        z_residual = 0

        # check if has 0.5 residual -> we have to cut 1 pixel more at the end
        if t["pad_x"] - int(t["pad_x"]) == 0.5:
            x_residual = 1
        if t["pad_y"] - int(t["pad_y"]) == 0.5:
            y_residual = 1
        if t["pad_z"] - int(t["pad_z"]) == 0.5:
            z_residual = 1

        # Cut padding
        shape = new_data.shape
        new_data = new_data[int(t["pad_x"]): shape[0] - int(t["pad_x"]) - x_residual,
                            int(t["pad_y"]): shape[1] - int(t["pad_y"]) - y_residual,
                            int(t["pad_z"]): shape[2] - int(t["pad_z"]) - z_residual]
        return new_data

    @staticmethod
    def get_bbox_from_mask(mask, outside_value=0):
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    @staticmethod
    def crop_to_bbox(image, bbox):
        assert len(image.shape) == 3, "only supports 3d images"
        return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

    @staticmethod
    def crop_to_nonzero(data, seg=None):
        original_shape = data.shape
        bbox = DatasetUtils.get_bbox_from_mask(data, 0)

        cropped_data = []
        for c in range(data.shape[3]):
            cropped = DatasetUtils.crop_to_bbox(data[:,:,:,c], bbox)
            cropped_data.append(cropped)
        data = np.array(cropped_data).transpose(1,2,3,0)

        return data, seg, bbox, original_shape

    @staticmethod
    def add_original_zero_padding_again(data, bbox, original_shape, nr_of_classes):
        data_new = np.zeros((original_shape[0], original_shape[1], original_shape[2], nr_of_classes)).astype(data.dtype)
        data_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = data
        return data_new
        # new_img = nib.Nifti1Image(data, data_img.get_affine())
        # nib.save(new_img, "/mnt/jakob/E130-Personal/Wasserthal/tmp/TEST.nii.gz")





