
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from builtins import object
import numpy as np

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import exp_utils
from tractseg.libs import img_utils
from tractseg.libs import data_utils
from tractseg.libs import peak_utils
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform as ZeroMeanUnitVarianceTransform_Standalone
from tractseg.data.DLDABG_standalone import SingleThreadedAugmenter
from tractseg.data.DLDABG_standalone import Compose
from tractseg.data.DLDABG_standalone import NumpyToTensor

np.random.seed(1337)  # for reproducibility


class BatchGenerator2D_data_ordered_standalone(object):
    '''
    Creates batch of 2D slices from one subject.

    Does not depend on DKFZ/BatchGenerators package. Therefore good for inference on windows
    where DKFZ/Batchgenerators do not work (because of MultiThreading problems)
    '''
    def __init__(self, data, batch_size):
        # super(self.__class__, self).__init__(*args, **kwargs)
        self.Config = None
        self.batch_size = batch_size
        self.global_idx = 0
        self._data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def generate_train_batch(self):
        data = self._data[0]
        seg = self._data[1]

        if self.Config.SLICE_DIRECTION == "x":
            end = data.shape[0]
        elif self.Config.SLICE_DIRECTION == "y":
            end = data.shape[1]
        elif self.Config.SLICE_DIRECTION == "z":
            end = data.shape[2]

        # Stop iterating if we reached end of data
        if self.global_idx >= end:
            # print("Stopped because end of file")
            self.global_idx = 0
            raise StopIteration

        new_global_idx = self.global_idx + self.batch_size

        # If we reach end, make last batch smaller, so it fits exactly into rest
        if new_global_idx >= end:
            new_global_idx = end  # not end-1, because this goes into range, and there automatically -1

        slice_idxs = list(range(self.global_idx, new_global_idx))
        slice_direction = data_utils.slice_dir_to_int(self.Config.SLICE_DIRECTION)

        if self.Config.NR_SLICES > 1:
            x, y = data_utils.sample_Xslices(data, seg, slice_idxs, slice_direction=slice_direction,
                                             labels_type=self.Config.LABELS_TYPE, slice_window=self.Config.NR_SLICES)
        else:
            x, y = data_utils.sample_slices(data, seg, slice_idxs,
                                            slice_direction=slice_direction,
                                            labels_type=self.Config.LABELS_TYPE)

        data_dict = {"data": x,     # (batch_size, channels, x, y, [z])
                     "seg": y}      # (batch_size, channels, x, y, [z])
        self.global_idx = new_global_idx
        return data_dict


class BatchGenerator3D_data_ordered_standalone(object):
    def __init__(self, data, batch_size=1):
        self.Config = None
        if batch_size != 1:
            raise ValueError("only batch_size=1 allowed")
        self.batch_size = batch_size
        self.global_idx = 0
        self._data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def generate_train_batch(self):
        data = self._data[0]    # (x, y, z, channels)
        seg = self._data[1]

        # Stop iterating if we reached end of data
        if self.global_idx >= 1:
            self.global_idx = 0
            raise StopIteration
        self.global_idx += self.batch_size

        x = data.transpose(3, 0, 1, 2)[np.newaxis,...]  # channels have to be first, add batch_size of 1
        y = seg.transpose(3, 0, 1, 2)[np.newaxis,...]

        data_dict = {"data": np.array(x),     # (batch_size, channels, x, y, [z])
                     "seg": np.array(y)}      # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderInference():
    """
    Data loader for only one subject and returning slices in ordered way.
    """

    def __init__(self, Config, data=None, subject=None):
        """
        Set either data or subject, not both.

        :param Config: Config class
        :param data: 4D numpy array with subject data
        :param subject: ID for a subject from the training data (string)
        """
        self.Config = Config
        self.data = data
        self.subject = subject

    def _augment_data(self, batch_generator, type=None):
        tfs = []  # transforms

        if self.Config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        # Not used, because those transformations are not easily invertible with batchgenerators framework:
        #  Mirroring would be the only easy test time DAug, but not trained with this DAug
        # if self.Config.TEST_TIME_DAUG:
            # from batchgenerators.transforms.spatial_transforms import SpatialTransform
            # center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
            # tfs.append(SpatialTransform(self.Config.INPUT_DIM,
            #                             patch_center_dist_from_border=center_dist_from_border,
            #                             do_elastic_deform=True, alpha=(90., 120.), sigma=(9., 11.),
            #                             do_rotation=True, angle_x=(-0.8, 0.8), angle_y=(-0.8, 0.8),
            #                             angle_z=(-0.8, 0.8),
            #                             do_scale=True, scale=(0.9, 1.5), border_mode_data='constant',
            #                             border_cval_data=0,
            #                             order_data=3,
            #                             border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True))
            # tfs.append(ResampleTransform(zoom_range=(0.5, 1)))
            # tfs.append(GaussianNoiseTransform(noise_variance=(0, 0.05)))
            # tfs.append(ContrastAugmentationTransform(contrast_range=(0.7, 1.3), preserve_range=True, per_channel=False))
            # tfs.append(BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), per_channel=False))

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        batch_gen = SingleThreadedAugmenter(batch_generator, Compose(tfs))
        return batch_gen

    def get_batch_generator(self, batch_size=1):

        if self.data is not None:
            exp_utils.print_verbose(self.Config, "Loading data from PREDICT_IMG input file")
            data = np.nan_to_num(self.data)
            # Use dummy mask in case we only want to predict on some data (where we do not have Ground Truth))
            seg = np.zeros((self.Config.INPUT_DIM[0], self.Config.INPUT_DIM[0],
                            self.Config.INPUT_DIM[0], self.Config.NR_OF_CLASSES)).astype(self.Config.LABELS_TYPE)
        elif self.subject is not None:
            if self.Config.TYPE == "combined":
                # Load from Npy file for Fusion
                data = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject,
                                    self.Config.FEATURES_FILENAME + ".npy"), mmap_mode="r")
                seg = np.load(join(C.DATA_PATH, self.Config.DATASET_FOLDER, self.subject,
                                   self.Config.LABELS_FILENAME + ".npy"), mmap_mode="r")
                data = np.nan_to_num(data)
                seg = np.nan_to_num(seg)
                data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            else:
                from tractseg.data.data_loader_training import load_training_data
                data, seg = load_training_data(self.Config, self.subject)

                # Convert peaks to tensors if tensor model
                if self.Config.NR_OF_GRADIENTS == 18 * self.Config.NR_SLICES:
                    data = peak_utils.peaks_to_tensors(data)

                data, transformation = data_utils.pad_and_scale_img_to_square_img(data,
                                                                                  target_size=self.Config.INPUT_DIM[0],
                                                                                  nr_cpus=1)
                seg, transformation = data_utils.pad_and_scale_img_to_square_img(seg,
                                                                                 target_size=self.Config.INPUT_DIM[0],
                                                                                 nr_cpus=1)
        else:
            raise ValueError("Neither 'data' nor 'subject' set.")

        if self.Config.DIM == "2D":
            batch_gen = BatchGenerator2D_data_ordered_standalone((data, seg), batch_size=batch_size)
        else:
            batch_gen = BatchGenerator3D_data_ordered_standalone((data, seg), batch_size=batch_size)
        batch_gen.Config = self.Config

        batch_gen = self._augment_data(batch_gen, type=type)
        return batch_gen

