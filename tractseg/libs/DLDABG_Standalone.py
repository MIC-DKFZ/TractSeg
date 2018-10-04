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

# from future import standard_library
# standard_library.install_aliases()
from builtins import object
import abc
from warnings import warn

'''
Copy part of code from https://github.com/MIC-DKFZ/batchgenerators needed for inference so we do not
need this dependency during inference. This way we can become windows compatible.
'''

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