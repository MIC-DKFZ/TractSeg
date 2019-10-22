
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import ndimage
import random

from tractseg.libs import img_utils


def pad_and_scale_img_to_square_img(data, target_size=144, nr_cpus=-1):
    """
    Expects 3D or 4D image as input.

    Does
    1. Pad image with 0 to make it square
        (if uneven padding -> adds one more px "behind" img; but resulting img shape will be correct)
    2. Scale image to target size
    """
    nr_dims = len(data.shape)
    assert (nr_dims >= 3 and nr_dims <= 4), "image has to be 3D or 4D"

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
        #use order=0, otherwise does not work for peak images (results would be wrong)
        new_img = img_utils.resize_first_three_dims(new_img, order=0, zoom=zoom, nr_cpus=nr_cpus)
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


def cut_and_scale_img_back_to_original_img(data, t, nr_cpus=-1):
    """
    Undo the transformations done with pad_and_scale_img_to_square_img

    Args:
        data: 3D or 4D image
        t: transformation dict
        nr_cpus: nr of cpus to use

    Returns:
        3D or 4D image
    """
    nr_dims = len(data.shape)
    assert (nr_dims >= 3 and nr_dims <= 4), "image has to be 3D or 4D"

    # Back to old size
    # use order=0, otherwise image values of a DWI will be quite different after downsampling and upsampling
    if nr_dims == 3:
        new_data = ndimage.zoom(data, (1. / t["zoom"]), order=0)
    elif nr_dims == 4:
        new_data = img_utils.resize_first_three_dims(data, order=0, zoom=(1. / t["zoom"]), nr_cpus=nr_cpus)

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


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def crop_to_nonzero(data, seg=None, bbox=None):
    original_shape = data.shape
    if bbox is None:
        bbox = get_bbox_from_mask(data, 0)

    cropped_data = []
    for c in range(data.shape[3]):
        cropped = crop_to_bbox(data[:,:,:,c], bbox)
        cropped_data.append(cropped)
    data = np.array(cropped_data).transpose(1,2,3,0)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[3]):
            cropped = crop_to_bbox(seg[:,:,:,c], bbox)
            cropped_seg.append(cropped)
        seg = np.array(cropped_seg).transpose(1, 2, 3, 0)

    return data, seg, bbox, original_shape


def add_original_zero_padding_again(data, bbox, original_shape, nr_of_classes):
    if nr_of_classes > 0:
        data_new = np.zeros(original_shape[:3] + (nr_of_classes,)).astype(data.dtype)
    else:
        data_new = np.zeros(original_shape[:3]).astype(data.dtype)
    data_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = data
    return data_new


def slice_dir_to_int(slice_dir):
    """
    Convert slice direction identifier to int.

    Args:
        slice_dir: x|y|z|xyz  (string)

    Returns:
        0|1|2 (int)
    """
    if slice_dir == "xyz":
        slice_direction_int = int(round(random.uniform(0, 2)))
    elif slice_dir == "x":
        slice_direction_int = 0
    elif slice_dir == "y":
        slice_direction_int = 1
    elif slice_dir == "z":
        slice_direction_int = 2
    else:
        raise ValueError("Invalid value for 'training_slice_direction'.")
    return slice_direction_int


def sample_slices(data, seg, slice_idxs, slice_direction=0, labels_type=np.int16):
    if slice_direction == 0:
        x = data[slice_idxs, :, :].astype(np.float32)  # (bs, y, z, channels)
        y = seg[slice_idxs, :, :].astype(labels_type)
        # depth-channel has to be before width and height for Unet (but after batches)
        x = np.array(x).transpose(0, 3, 1, 2)
        # nr_classes channel has to be before with and height for DataAugmentation (bs, channels, x, y)
        y = np.array(y).transpose(0, 3, 1, 2)
    elif slice_direction == 1:
        x = data[:, slice_idxs, :].astype(np.float32)  # (x, bs, z, channels)
        y = seg[:, slice_idxs, :].astype(labels_type)
        x = np.array(x).transpose(1, 3, 0, 2)
        y = np.array(y).transpose(1, 3, 0, 2)
    elif slice_direction == 2:
        x = data[:, :, slice_idxs].astype(np.float32)  # (x, y, bs, channels)
        y = seg[:, :, slice_idxs].astype(labels_type)
        x = np.array(x).transpose(2, 3, 0, 1)
        y = np.array(y).transpose(2, 3, 0, 1)
    return x, y


def sample_Xslices(data, seg, slice_idxs, slice_direction=0, labels_type=np.int16, slice_window=5):
    """
    Sample slices but add slices_window/2 above and below.
    """
    sw = slice_window  # slice_window (only odd numbers allowed)
    assert sw % 2 == 1, "Slice_window has to be an odd number"
    pad = int((sw - 1) / 2)

    if slice_direction == 0:
        y = seg[slice_idxs, :, :].astype(labels_type)
        y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
    elif slice_direction == 1:
        y = seg[:, slice_idxs, :].astype(labels_type)
        y = np.array(y).transpose(1, 3, 0, 2)
    elif slice_direction == 2:
        y = seg[:, :, slice_idxs].astype(labels_type)
        y = np.array(y).transpose(2, 3, 0, 1)

    data_pad = np.zeros((data.shape[0] + sw - 1, data.shape[1] + sw - 1, data.shape[2] + sw - 1, data.shape[3])).astype(
        data.dtype)
    data_pad[pad:-pad, pad:-pad, pad:-pad, :] = data  # padded with two slices of zeros on all sides
    batch = []
    for s_idx in slice_idxs:
        if slice_direction == 0:
            # (s_idx+2)-2:(s_idx+2)+3 = s_idx:s_idx+5
            x = data_pad[s_idx:s_idx + sw:, pad:-pad, pad:-pad, :].astype(np.float32)  # (5, y, z, channels)
            x = np.array(x).transpose(0, 3, 1, 2)  # channels dim has to be before width and height for Unet (but after batches)
            x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
            batch.append(x)
        elif slice_direction == 1:
            x = data_pad[pad:-pad, s_idx:s_idx + sw, pad:-pad, :].astype(np.float32)  # (5, y, z, channels)
            x = np.array(x).transpose(1, 3, 0, 2)
            x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
            batch.append(x)
        elif slice_direction == 2:
            x = data_pad[pad:-pad, pad:-pad, s_idx:s_idx + sw, :].astype(np.float32)  # (5, y, z, channels)
            x = np.array(x).transpose(2, 3, 0, 1)
            x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
            batch.append(x)

    return np.array(batch), y  # (bs, channels, x, y)
