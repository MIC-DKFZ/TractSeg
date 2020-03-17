"""
Copy part of code from https://github.com/MIC-DKFZ/batchgenerators needed for inference so we do not
need this dependency during inference. This way we can become windows compatible.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import object
import abc

import numpy as np
import torch


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


def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-8):
    data_normalized = np.zeros(data.shape, dtype=data.dtype)
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                mean = data[b, c].mean()
                std = data[b, c].std() + epsilon
                data_normalized[b, c] = (data[b, c] - mean) / std
        else:
            mean = data[b].mean()
            std = data[b].std() + epsilon
            data_normalized[b] = (data[b] - mean) / std
    return data_normalized


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
    """
    Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    :param keys: specify keys to be converted to tensors. If None then all keys will be converted
    (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
    :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
    and long supported (use string)
    """
    def __init__(self, keys=None, cast_to=None, pin_memory=False):
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
