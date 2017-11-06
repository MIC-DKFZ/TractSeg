#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

#from vtk.util.numpy_support import vtk_to_numpy
#import vtk

import time
import numpy as np
import nibabel as nib
from dipy.tracking import utils as tracking_utils
import cPickle
import bz2
import psutil

class Utils:
    
    def __init__(self):
        return None

    '''
    MASK STATISTICS:
    SMALL (DWI):
    shape (80, 80, 90)
    nr pixel 1: 463
    nr pixels not 1: 576000
    => 0.0008 %    (andere Zahl, als bei big, weil bei resampling rounding danach gemacht habe auf int)
          Epoch 0, current w_dec 1.0
          Epoch 0, current freq0_decayed 0.00160763888889
          Epoch 0, current freq1_decayed 1.99839236111

    BIG (T1):
    (176, 176, 180)
    3549
    5575680
    => 0.0006 %
        Epoch 0, current w_dec 1.0
        Epoch 0, current freq0_decayed 0.00127302858127
        Epoch 0, current freq1_decayed 1.99872697142
    '''
    @staticmethod
    def get_class_frequency(dataManager, batch_size):
        '''
        Expects labels (y) to be 0 or 1.

        :param dataManager:
        :return: tupel: frequency of class0 and frequency of class1
        '''
        print("Calculating class frequency...")
        start_time = time.time()

        # print("Setting frequencies manually!!!")
        # print("frequencies for 3DUNet HCP30 CA Training:")
        # freq0 = 0.0019133839134572177
        # freq1 = 1.9980866160865427
        # print("invFreqBg", freq0)
        # print("invFreqBun", freq1)
        # return freq0, freq1

        # print("Setting frequencies manually!!!")
        # print("frequencies for 3DUNet HCP30 CST_right Training:")
        # freq0 = 0.012968554687500002
        # freq1 = 1.9870314453125
        # print("invFreqBg", freq0)
        # print("invFreqBun", freq1)
        # return freq0, freq1

        # print("Setting frequencies manually!!!")
        # print("frequencies for UNet_ML HCP30 3 bundles Training:")
        # freq0 = 0.04285798611111112
        # freq1 = 1.9571420138888889
        # print("invFreqBg", freq0)
        # print("invFreqBun", freq1)
        # return freq0, freq1

        # print("Setting frequencies manually!!!")
        # print("frequencies for HCP30sm_17Bun_x Training:")
        # freq0 = 0.2262892578125
        # freq1 = 1.7737107421875
        # print("invFreqBg", freq0)
        # print("invFreqBun", freq1)
        # return freq0, freq1

        print("Setting frequencies manually!!!")
        print("frequencies for HCP100_45B_UNet_x_bs32 Training:")
        freq0 = 0.22181060749153378
        freq1 = 1.7781893925084662
        print("invFreqBg", freq0)
        print("invFreqBun", freq1)
        return freq0, freq1


        batch_generator = dataManager.get_batches(batch_size=batch_size, shuffle=False, type="train")
        count0 = 0
        count1 = 0
        for batch in batch_generator:
            # x, y = batch
            # x = np.array(x)
            y = batch["seg"]  # (bs, 1, x, y, nr_of_classes)

            y = np.array(y)
            y = np.reshape(y, (-1, y.shape[-1]))  # (bs*x*y, nr_of_classes)
            y = y[:, 0]  # only select background (calc frequencies on basis of background vs all-bundles) -> not sure if ideal

            #invert 0 und 1 (because here background is 1, but in the past background was 0)  -> make it like in the past
            y = y == 1    # to bool
            y = np.invert(y)
            y = y.astype(np.int16)   #back to int

            nr_elems = y.size
            non_zero = np.count_nonzero(y)

            count0 += (nr_elems - non_zero)
            count1 += non_zero

        invFreqBg = 1 / (count0 / float(count0 + count1))
        invFreqBun = 1 / (count1 / float(count0 + count1))

        mean_freq = sum([invFreqBg, invFreqBun]) / 2.0

        print("invFreqBg", invFreqBg / mean_freq)
        print("invFreqBun", invFreqBun / mean_freq)
        print("took: {}s".format(time.time() - start_time))

        return invFreqBg / mean_freq, invFreqBun / mean_freq

    '''
    Comes from dipy (dipy/io/vtk.py) (could not be imported from there)
    '''
    @staticmethod
    def xxx_load_polydata(file_name):
        """ Load a vtk polydata to a supported format file
        Supported file formats are OBJ, VTK, FIB, PLY, STL and XML
        Parameters
        ----------
        file_name : string
        Returns
        -------
        output : vtkPolyData
        """
        # get file extension (type) lower case
        file_extension = file_name.split(".")[-1].lower()

        if file_extension == "vtk":
            reader = vtk.vtkPolyDataReader()
        elif file_extension == "fib":
            reader = vtk.vtkPolyDataReader()
        elif file_extension == "ply":
            reader = vtk.vtkPLYReader()
        elif file_extension == "stl":
            reader = vtk.vtkSTLReader()
        elif file_extension == "xml":
            reader = vtk.vtkXMLPolyDataReader()
        elif file_extension == "obj":
            try:  # try to read as a normal obj
                reader = vtk.vtkOBJReader()
            except:  # than try load a MNI obj format
                reader = vtk.vtkMNIObjectReader()
        else:
            raise "polydata " + file_extension + " is not suported"

        reader.SetFileName(file_name)
        reader.Update()
        # print(file_name + " Mesh " + file_extension + " Loaded")
        
        #print(type(reader))
        
        return reader.GetOutput()

    '''
    Comes from dipy (dipy/viz/utils.py) (could not be imported from there / was not found)
    '''
    @staticmethod
    def xxx_get_polydata_lines(line_polydata):
        """ vtk polydata to a list of lines ndarrays
        Parameters
        ----------
        line_polydata : vtkPolyData
        Returns
        -------
        lines : list
            List of N curves represented as 2D ndarrays
        """
        lines_vertices = vtk_to_numpy(line_polydata.GetPoints().GetData())
        lines_idx = vtk_to_numpy(line_polydata.GetLines().GetData())

        lines = []
        current_idx = 0
        while current_idx < len(lines_idx):
            line_len = lines_idx[current_idx]

            next_idx = current_idx + line_len + 1
            line_range = lines_idx[current_idx + 1: next_idx]

            lines += [lines_vertices[line_range]]
            current_idx = next_idx
        return lines

    @staticmethod
    def invert_x_and_y(affineMatrix):
        '''
        Change sign of x and y transformation (rotation, scaling and transformation)

        IMPORTANT note: only done for diagonal elements (if we need rotation (not only scaling) we may also need
        to do it for non-diagonal elements) -> not done yet
        '''
        newAffine = affineMatrix.copy()
        newAffine[0,0] = newAffine[0,0] * -1
        newAffine[1,1] = newAffine[1,1] * -1
        newAffine[0,3] = newAffine[0,3] * -1
        newAffine[1,3] = newAffine[1,3] * -1
        return newAffine

    #For an up to date version see FiberUtils
    @staticmethod
    def save_streamlines_as_trk_manual_DEPRECATED(filename, streamlines, zooms, shape, affine):
        '''
        streamlines: list of 2D ndarrays   list(ndarray(N,3))
        '''

        # Make a trackvis header so we can save streamlines
        trackvis_header = nib.trackvis.empty_header()
        trackvis_header['voxel_size'] = zooms #zooms has to be only 3 elements -> dwi zooms as 4 -> take only first 3
        # Do not use this line:
        # trackvis_header['voxel_order'] = 'RAS'
        # If we use RAS (says if going from left to right or from right to left...) then we have to remove invert_x_and_y
        # here to make it align properly
        # => dipy uses other voxel_order than MITK
        trackvis_header['dim'] = shape

        # Move streamlines to "trackvis space"
        streamlines_world_space = list(tracking_utils.move_streamlines(streamlines, Utils.invert_x_and_y(affine)))

        streamlines_trk_format = [(sl, None, None) for sl in streamlines_world_space]
        nib.trackvis.write(filename, streamlines_trk_format, trackvis_header)

    # For an up to date version see FiberUtils
    @staticmethod
    def save_streamlines_as_trk_DEPRECATED(filename, streamlines, dwi):
        '''
        streamlines: list of 2D ndarrays   list(ndarray(N,3))
        '''
        Utils.save_streamlines_as_trk_manual_DEPRECATED(filename, streamlines, dwi.get_header().get_zooms()[:3], dwi.get_data().shape[:3], dwi.get_affine())
    
    @staticmethod
    def normalize_data(data, where_b0, min_signal=1., out=None):
        """
        Normalizes the data with respect to the mean b0 (mean of b0 along z Axis)

        method from: https://github.com/nipy/dipy/blob/d0bee8c811daf00c5f9c153168ccbc82fa3b5557/dipy/reconst/shm.py#L741
        
        Ergebnisse schauen mehr verändert aus, als wenn normalize_mean0_std0 mache => besser normalize_mean0_std0 verwenden
        """
        if out is None:
            out = np.array(data, dtype='float32', copy=True)
        else:
            if out.dtype.kind != 'f':
                raise ValueError("out must be floating point")
            out[:] = data

        #out.clip(min_signal, out=out)
        b0 = out[..., where_b0].mean(-1) #mean(-1) -> mean along the last axis (here: z)
        #print(b0.shape)
        #print(b0[..., None].shape)
        #print(out.shape)
        out /= b0[..., None, None] # original: out /= b0[..., None]  -> error dim mismatch
        return out

    @staticmethod
    def normalize_mean0_std0(data):
        '''
        Normalizes along all axis for mean=0 and stddev=1

        :param data: ndarray, 4D
        :return: ndarray, 4D
        '''
        out = np.array(data, dtype='float32', copy=True)

        #mean = 0
        # mean = data.mean((0,1,2,3)) #mean over axis 0,1,2,3
        mean = data.mean() #mean over all axis / over flattened array
        out -= mean
        
        #std = 1
        std = data.std()
        out /= std
        
        return out

    @staticmethod
    def to_unit_length(vec):
        '''
        :param vec: 3D vector ("point")
        :return: 3D vector with len=1, but same direction as original vector
        '''
        vec_length = np.sqrt(np.sum(np.square(vec)))
        return vec / vec_length  # divide elementwise

    @staticmethod
    def to_unit_length_batch(vec):
        '''
        :param vec: array of 3D vectors
        :return: array of 3D vectors with len=1, but same direction as original vector
        '''
        vec_length = np.sqrt(np.sum(np.square(vec), axis=1))
        return vec / vec_length[:, np.newaxis]  # divide elementwise (only along one axis)

    @staticmethod
    def get_lr_decay(epoch_nr):
        '''
        Calc what lr_decay is need to make lr be 1/10 of original lr after epoch_nr number of epochs
        :return: lr_decay
        '''
        target_lr = 0.1 #should be reduced to 1/10 of original
        return target_lr ** (1 / float(epoch_nr))

    @staticmethod
    def is_abort_sign(direction):
        # todo: Is abort sign working??
        if direction[0] == 0 and direction[1] == 0 and direction[2] == 0:
            return True
        else:
            return False

    @staticmethod
    def save_pkl_compressed(filename, myobj):
        """
        save object to file using pickle

        @param filename: name of destination file
        @type filename: str
        @param myobj: object to save (has to be pickleable)
        @type myobj: obj
        """
        try:
            f = bz2.BZ2File(filename, 'wb')
        except IOError, details:
            sys.stderr.write('File ' + filename + ' cannot be written\n')
            sys.stderr.write(details)
            return

        cPickle.dump(myobj, f, protocol=2)
        f.close()

    @staticmethod
    def load_pkl_compressed(filename):
        """
        Load from filename using pickle

        @param filename: name of file to load from
        @type filename: str
        """
        try:
            f = bz2.BZ2File(filename, 'rb')
        except IOError, details:
            sys.stderr.write('File ' + filename + ' cannot be read\n')
            sys.stderr.write(details)
            return

        myobj = cPickle.load(f)
        f.close()
        return myobj

    @staticmethod
    def chunks(l, n):
        '''
        Yield successive n-sized chunks from l.
        Last chunk can be smaller.
        '''
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    @staticmethod
    def mem_usage(print_usage=True):
        process = psutil.Process()
        gb = process.memory_info().rss / 1e9
        gb = round(gb, 3)
        if print_usage:
            print("PID {} using {} GB".format(os.getpid(), gb))
        return gb



        
