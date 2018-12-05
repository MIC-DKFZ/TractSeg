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
from dipy.tracking.streamline import compress_streamlines as compress_streamlines_dipy
from dipy.segment.metric import ResampleFeature
from dipy.tracking.metrics import spline
from dipy.tracking import utils as utils_trk

from tractseg.libs import utils

logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
logging.getLogger().setLevel(logging.INFO)


# Global variables needed for shared memory of parallel fiber compression
global _COMPRESSION_ERROR_THRESHOLD
_COMPRESSION_ERROR_THRESHOLD = None
global _FIBER_BATCHES
_FIBER_BATCHES = None

def compress_fibers_worker_shared_mem(idx):
    """
    Worker Functions for multithreaded compression.

    Function that runs in parallel must be on top level (not in class/function) otherwise it can
    not be pickled and then error.
    """
    streamlines_chunk = _FIBER_BATCHES[idx]  # shared memory; by using indices each worker accesses only his part
    result = compress_streamlines_dipy(streamlines_chunk, tol_error=_COMPRESSION_ERROR_THRESHOLD)
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

    #Do not pass data in (doubles amount of memory needed), but only idx of shared memory
    #  (needs only as much memory as single thread version (only main thread needs memory, others almost 0).
    #  Shared memory version also faster (around 20-30%?).
    #  Needed otherwise memory problems when processing the raw tracking output (on disk >10GB and in memory >20GB)
    result = pool.map(compress_fibers_worker_shared_mem, range(0, len(fiber_batches)))

    pool.close()
    pool.join()

    streamlines_c = utils.flatten(result)
    return streamlines_c


def save_streamlines_as_trk_legacy(out_file, streamlines, affine, shape):
    """
    This function saves tracts in Trackvis '.trk' format.
    Uses the old nib.trackvis API (streamlines are saved in coordinate space. Affine is not applied.)

    Args:
        out_file: string with filepath of the output file
        streamlines: sequence of streamlines in RASmm coordinate (list of 2D numpy arrays)
        affine: 4d array with voxel to RASmm transformation
        shape: 1d array with dimensions of the brain volume, default [145, 174, 145]

    Returns:
        void
    """
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
    nib.trackvis.write(out_file, streamlines_trk_format, trackvis_header, points_space="rasmm")


def save_streamlines(out_file, streamlines, affine=None, shape=None, vox_sizes=None, vox_order='RAS'):
    """
    Saves streamlines either in .trk format or in .tck format. Depending on the ending of out_file.

    If using .trk: This function saves tracts in Trackvis '.trk' format.
    The default values for the parameters are the values for the HCP data.
    The HCP default affine is: array([[  -1.25,    0.  ,    0.  ,   90.  ],
                                      [   0.  ,    1.25,    0.  , -126.  ],
                                      [   0.  ,    0.  ,    1.25,  -72.  ],
                                      [   0.  ,    0.  ,    0.  ,    1.  ]],
                                     dtype=float32)
    Uses the new nib.streamlines API (streamlines are saved in voxel space and affine is applied to transform them to
    coordinate space).

    Args:
        out_file: string with filepath of the output file
        streamlines: sequence of streamlines in RASmm coordinate
        affine: 4d array with voxel to RASmm transformation
        shape: 1d array with dimensions of the brain volume, default [145, 174, 145]
        vox_sizes: 1d array with the voxels sizes, if None takes the absolute values of the diagonal of the affine
        vox_order: orientation convention, default to 'LAS'

    Returns:
        void
    """
    if affine is None:
        affine = np.array([[-1.25, 0., 0., 90.],
                           [0., 1.25, 0., -126.],
                           [0., 0., 1.25, -72.],
                           [0., 0., 0., 1.]],
                          dtype=np.float32)

    if shape is None:
        shape = np.array([145, 174, 145], dtype=np.int16)

    if vox_sizes is None:
        vox_sizes = np.array([abs(affine[0,0]), abs(affine[1,1]), abs(affine[2,2])], dtype=np.float32)

    # Create a new header with the correct affine and # of streamlines
    hdr = nib.streamlines.trk.TrkFile.create_empty_header()
    hdr['voxel_sizes'] = vox_sizes
    hdr['voxel_order'] = vox_order
    hdr['dimensions'] = shape
    hdr['voxel_to_rasmm'] = affine
    hdr['nb_streamlines'] = len(streamlines)

    nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), out_file, header=hdr)


def convert_tck_to_trk(filename_in, filename_out, reference_affine, reference_shape,
                       compress_err_thr=0.1, smooth=None, nr_cpus=-1, tracking_format="trk_legacy"):
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

    if tracking_format == "trk_legacy":
        save_streamlines_as_trk_legacy(filename_out, streamlines, reference_affine, reference_shape)
    else:
        save_streamlines(filename_out, streamlines, reference_affine, reference_shape)


def resample_fibers(streamlines, nb_points=12):
    streamlines_new = []
    for sl in streamlines:
        feature = ResampleFeature(nb_points=nb_points)
        streamlines_new.append(feature.extract(sl))
    return streamlines_new


def smooth_streamlines(streamlines, smoothing_factor=10):
    """

    Args:
        streamlines:
        smoothing_factor: 10: slight smoothing,  100: very smooth from beginning to end

    Returns:

    """
    streamlines_smooth = []
    for sl in streamlines:
        streamlines_smooth.append(spline(sl, s=smoothing_factor))
    return streamlines_smooth


def get_streamline_statistics(streamlines, subsample=False, raw=False):
    '''
    Returns (in mm)
    - mean streamline length (mm)
    - mean space between two following points (mm)
    - max space between two following points (mm)

    If raw: return list of fibers length and spaces

    :param streamlines:
    :return:
    '''
    if subsample:   #subsample for faster processing
        STEP_SIZE = 20
    else:
        STEP_SIZE = 1

    lengths = []
    spaces = [] #spaces between 2 points
    for j in range(0, len(streamlines), STEP_SIZE):
        sl = streamlines[j]
        length = 0
        for i in range(len(sl)):
            if i < (len(sl)-1):
                space = np.linalg.norm(sl[i+1] - sl[i])
                spaces.append(space)
                length += space
        lengths.append(length)

    if raw:
        # print("raw")
        return lengths, spaces
    else:
        # print("mean")
        return np.array(lengths).mean(), np.array(spaces).mean(), np.array(spaces).max()


def filter_streamlines_leaving_mask(streamlines, mask):
    '''
    Remove all streamlines that exit the mask
    '''
    # max_seq_len = abs(ref_affine[0, 0] / 4)
    max_seq_len = 0.1
    streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))

    new_str_idxs = []
    for i, streamline in enumerate(streamlines):
        new_str_idxs.append(i)
        for point in streamline:
            if mask[int(point[0]), int(point[1]), int(point[2])] == 0:
                new_str_idxs.pop()
                break
    return [streamlines[idx] for idx in new_str_idxs]


def get_best_original_peaks(peaks_pred, peaks_orig, peak_len_thr=0.1):
    """
    Find the peak from preaks_orig which is closest to the peak in peaks_pred.

    Args:
        peaks_pred: file containing 1 peak [x,y,z,3]
        peaks_orig: file containing 4 peaks [x,y,z,9]
        peak_len_thr: all peaks shorter than this threshold will be removed

    Returns:
        Image containing 1 peak [x,y,z,3]
    """

    def angle_last_dim(a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension

        without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)

        return: one dimension less then input
        '''
        return abs(np.einsum('...i,...i', a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-7))

    def get_most_aligned_peak(pred, orig):
        orig = np.array(orig)
        angle1 = angle_last_dim(pred, orig[0])
        angle2 = angle_last_dim(pred, orig[1])
        angle3 = angle_last_dim(pred, orig[2])
        argmax = np.argmax(np.stack([angle1, angle2, angle3], axis=-1), axis=-1)

        x, y, z = (orig.shape[1], orig.shape[2], orig.shape[3])
        return orig[tuple([argmax] + np.ogrid[:x, :y, :z])]
        # Other ways that would also work
        # return orig[argmax, np.arange(x)[:, None, None], np.arange(y)[:, None], np.arange(z)]
        # return np.take_along_axis(orig, argmax[None, ..., None], axis=0)[0]   # only supported in newest numpy version

    peaks_pred = np.nan_to_num(peaks_pred)
    peaks_orig = np.nan_to_num(peaks_orig)

    #Remove all peaks where predicted peaks are too short
    peaks_orig[np.linalg.norm(peaks_pred, axis=-1) < peak_len_thr] = 0

    best_orig = get_most_aligned_peak(peaks_pred,
                                      [peaks_orig[:, :, :, 0:3],
                                       peaks_orig[:, :, :, 3:6],
                                       peaks_orig[:, :, :, 6:9]])
    return best_orig


def add_to_each_streamline(streamlines, scalar):
    """
    Add scalar value to each coordinate of each streamline

    Args:
        streamlines:
        scalar:

    Returns:

    """
    sl_new = []
    for sl in streamlines:
        sl_new.append(np.array(sl) + scalar)
    return sl_new