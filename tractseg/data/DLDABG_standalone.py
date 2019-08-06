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

# from future import standard_library
# standard_library.install_aliases()
from builtins import object
import abc
from warnings import warn

"""
Copy part of code from https://github.com/MIC-DKFZ/batchgenerators needed for inference so we do not
need this dependency during inference. This way we can become windows compatible.
"""

class SingleThreadedAugmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.data_loader)
        item = self.transform(**item)
        return item


def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-7):
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                mean = data[b, c].mean()
                std = data[b, c].std() + epsilon
                data[b, c] = (data[b, c] - mean) / std
        else:
            mean = data[b].mean()
            std = data[b].std() + epsilon
            data[b] = (data[b] - mean) / std
    return data


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class Compose(AbstractTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class ZeroMeanUnitVarianceTransform(AbstractTransform):
    """ Zero mean unit variance transform

    Args:
        per_channel (bool): determines whether mean and std are computed for and applied to each color channel
        separately

        epsilon (float): prevent nan if std is zero, keep at 1e-7
    """

    def __init__(self, per_channel=True, epsilon=1e-7, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.epsilon = epsilon
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict[self.data_key] = zero_mean_unit_variance_normalization(data_dict[self.data_key], self.per_channel,
                                                                         self.epsilon)
        return data_dict


class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None, pin_memory=False):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param keys: specify keys to be converted to tensors. If None then all keys will be converted
        (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
        :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
        and long supported (use string)
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to
        self.pin_memory = pin_memory

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            else:
                raise ValueError('Unknown value for cast_to: %s' % self.cast_to)
        return tensor

    def __call__(self, **data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(val))
                    if self.pin_memory:
                        data_dict[key] = data_dict[key].pin_memory()
        else:
            for key in self.keys:
                data_dict[key] = self.cast(torch.from_numpy(data_dict[key]))
                if self.pin_memory:
                    data_dict[key] = data_dict[key].pin_memory()

        return data_dict


class ResampleTransformLegacy(AbstractTransform):
    '''
    This is no longer part of batchgenerators, so we have an implementation here.
    CPU always 100% when using this, but batch_time on cluster not longer (1s)

    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)
    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    '''

    def __init__(self, zoom_range=(0.5, 1)):
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict['data'] = augment_linear_downsampling_scipy(data_dict['data'], zoom_range=self.zoom_range)
        return data_dict


def augment_linear_downsampling_scipy(data, zoom_range=(0.5, 1)):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)
    '''
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
