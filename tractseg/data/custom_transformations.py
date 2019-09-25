
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def augment_linear_downsampling_scipy(data, zoom_range=(0.5, 1)):
    """
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)
    """
    import random
    import scipy.ndimage
    import numpy as np

    zoom_range = list(zoom_range)
    zoom_range[1] += + 1e-6
    if zoom_range[0] >= zoom_range[1]:
        raise ValueError("First value of zoom_range must be smaller than second value.")

    dim = len(data.shape[2:])  # remove batch_size and nr_of_channels dimension
    for sample_idx in range(data.shape[0]):

        zoom = round(random.uniform(zoom_range[0], zoom_range[1]), 2)

        for channel_idx in range(data.shape[1]):
            img = data[sample_idx, channel_idx]
            img_down = scipy.ndimage.zoom(img, zoom, order=1)
            zoom_reverse = round(1. / zoom, 2)
            img_up = scipy.ndimage.zoom(img_down, zoom_reverse, order=0)

            if dim == 3:
                # cut if dimension got too long
                img_up = img_up[:img.shape[0], :img.shape[1], :img.shape[2]]

                # pad with 0 if dimension too small
                img_padded = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                img_padded[:img_up.shape[0], :img_up.shape[1], :img_up.shape[2]] = img_up

                data[sample_idx, channel_idx] = img_padded

            elif dim == 2:
                # cut if dimension got too long
                img_up = img_up[:img.shape[0], :img.shape[1]]

                # pad with 0 if dimension too small
                img_padded = np.zeros((img.shape[0], img.shape[1]))
                img_padded[:img_up.shape[0], :img_up.shape[1]] = img_up

                data[sample_idx, channel_idx] = img_padded
            else:
                raise ValueError("Invalid dimension size")

    return data


class ResampleTransformLegacy(AbstractTransform):
    """
    This is no longer part of batchgenerators, so we have an implementation here.
    CPU always 100% when using this, but batch_time on cluster not longer (1s)

    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor
      (like isotropic=True from linear_downsampling_generator_nilearn)

    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    """

    def __init__(self, zoom_range=(0.5, 1)):
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict['data'] = augment_linear_downsampling_scipy(data_dict['data'], zoom_range=self.zoom_range)
        return data_dict


def flip_vector_axis(data):
    data = np.copy(data)
    if (len(data.shape) != 4) and (len(data.shape) != 5) or data.shape[1] != 9:
        raise Exception("Invalid dimension for data. Data should be either [BATCH_SIZE, 9, x, y] or [BATCH_SIZE, 9, x, y, z]")
    axis = np.random.choice(["x", "y", "z"])   #chose axes to flip
    BATCH_SIZE = data.shape[0]
    for id in np.arange(BATCH_SIZE):
        if np.random.uniform() < 0.5:
            if axis == "x":
                data[id, 0] *= -1
                data[id, 3] *= -1
                data[id, 6] *= -1
            elif axis == "y":
                data[id, 1] *= -1
                data[id, 4] *= -1
                data[id, 7] *= -1
            elif axis == "z":
                data[id, 2] *= -1
                data[id, 5] *= -1
                data[id, 8] *= -1

    return data


class FlipVectorAxisTransform(AbstractTransform):
    """
    Expects as input an image with 3 3D-vectors at each voxels, encoded as a nine-channel image. Will randomly
    flip sign of one dimension of all 3 vectors (x, y or z).
    """
    def __init__(self, axes=(2, 3, 4), data_key="data"):
        self.data_key = data_key
        self.axes = axes

    def __call__(self, **data_dict):
        data_dict[self.data_key] = flip_vector_axis(data=data_dict[self.data_key])
        return data_dict
