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

import os, sys
import numpy as np
import cPickle
import bz2
import urllib2
from tractseg.libs.Config import Config as C

class Utils:

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
        import psutil
        process = psutil.Process()
        gb = process.memory_info().rss / 1e9
        gb = round(gb, 3)
        if print_usage:
            print("PID {} using {} GB".format(os.getpid(), gb))
        return gb

    @staticmethod
    def download_pretrained_weights():
        weights_path = os.path.join(C.TRACT_SEG_HOME, 'pretrained_weights.npz')
        WEIGHTS_URL = "https://www.dropbox.com/s/94yiiebrxjj5pw6/unet_weights_Mir_ep278.npz?dl=1"

        if not os.path.exists(weights_path):
            print("Downloading pretrained weights (~140MB) ...")
            if not os.path.exists(C.TRACT_SEG_HOME):
                os.makedirs(C.TRACT_SEG_HOME)

            #This results in an SSL Error on CentOS
            # urllib.urlretrieve(WEIGHTS_URL, weights_path)

            data = urllib2.urlopen(WEIGHTS_URL).read()
            with open(weights_path, "wb") as weight_file:
                weight_file.write(data)

            print("Done")

