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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from os import getpid
import logging
import psutil
import numpy as np
import nibabel as nib
from dipy.tracking.streamline import compress_streamlines
from dipy.segment.metric import ResampleFeature

from tractseg.libs import utils

logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
logging.getLogger().setLevel(logging.INFO)


#Global variables needed for shared memory of parallel fiber compression
global _COMPRESSION_ERROR_THRESHOLD
_COMPRESSION_ERROR_THRESHOLD = None
global _FIBER_BATCHES
_FIBER_BATCHES = None

# Worker Functions for multithreaded compression
def compress_fibers_worker_shared_mem(idx):
    # Function that runs in parallel must be on top level (not in class/function) otherwise it can not be pickled and then error
    streamlines_chunk = _FIBER_BATCHES[idx]  # shared memory; by using indices each worker accesses only his part
    result = compress_streamlines(streamlines_chunk, tol_error=_COMPRESSION_ERROR_THRESHOLD)
    logging.debug('PID {}, DONE'.format(getpid()))
    return result


def compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=-1):
    if nr_cpus == -1:
        nr_processes = psutil.cpu_count()
    else:
        nr_processes = nr_cpus
    number_streamlines = len(streamlines)

    if nr_processes >= number_streamlines:
        nr_processes = number_streamlines - 1
        if nr_processes < 1:
            nr_processes = 1

    chunk_size = int(number_streamlines / nr_processes)

    if chunk_size < 1:
        # logging.warning("\nReturning early because chunk_size=0")
        return streamlines
    fiber_batches = list(utils.chunks(streamlines, chunk_size))

    global _COMPRESSION_ERROR_THRESHOLD
    global _FIBER_BATCHES
    _COMPRESSION_ERROR_THRESHOLD = error_threshold
    _FIBER_BATCHES = fiber_batches

    # logging.debug("Main program using: {} GB".format(round(Utils.mem_usage(print_usage=False), 3)))
    pool = multiprocessing.Pool(processes=nr_processes)

    #Do not pass data in (doubles amount of memory needed), but only idx of shared memory (needs only as much memory as single
    # thread version (only main thread needs memory, others almost 0).
    # Shared memory version also faster (around 20-30%?).
    # Needed otherwise memory problems when processing the raw tracking output (on disk >10GB and in memory >20GB)
    result = pool.map(compress_fibers_worker_shared_mem, range(0, len(fiber_batches)))

    pool.close()
    pool.join()

    streamlines_c = utils.flatten(result)
    return streamlines_c


def save_streamlines_as_trk(filename, streamlines, affine, shape):
    '''
    streamlines: list of 2D ndarrays   list(ndarray(N,3))
    affine: affine of reference img (e.g. brainmask)
    '''
    affine = np.abs(affine) #have to positive
    #offset not needed (already part of streamline coordinates?)
    affine[0, 3] = 0
    affine[1, 3] = 0
    affine[2, 3] = 0
    # Make a trackvis header so we can save streamlines
    trackvis_header = nib.trackvis.empty_header()
    trackvis_header['voxel_order'] = 'RAS'
    trackvis_header['dim'] = shape
    nib.trackvis.aff_to_hdr(affine, trackvis_header, pos_vox=False, set_order=False)
    streamlines_trk_format = [(sl, None, None) for sl in streamlines]
    nib.trackvis.write(filename, streamlines_trk_format, trackvis_header, points_space="rasmm")


def convert_tck_to_trk(filename_in, filename_out, reference_affine, reference_shape,
                       compress_err_thr=0.1, smooth=None, nr_cpus=-1):
    '''
    Convert tck file to trk file and compress

    :param filename_in:
    :param filename_out:
    :param compress_err_thr: compress fibers if setting error threshold here (default: 0.1mm)
    :param smooth: smooth streamlines (default: None)
                   10: slight smoothing,  100: very smooth from beginning to end
    :param nr_cpus:
    :return:
    '''
    from dipy.tracking.metrics import spline

    streamlines = nib.streamlines.load(filename_in).streamlines  # Load Fibers (Tck)

    if smooth is not None:
        streamlines_smooth = []
        for sl in streamlines:
            streamlines_smooth.append(spline(sl, s=smooth))
        streamlines = streamlines_smooth

    #Compressing also good to remove checkerboard artefacts from tracking on peaks
    if compress_err_thr is not None:
        streamlines = compress_streamlines(streamlines, compress_err_thr, nr_cpus=nr_cpus)
    save_streamlines_as_trk(filename_out, streamlines, reference_affine, reference_shape)


def resample_fibers(streamlines, nb_points=12):
    streamlines_new = []
    for sl in streamlines:
        feature = ResampleFeature(nb_points=nb_points)
        streamlines_new.append(feature.extract(sl))
    return streamlines_new

