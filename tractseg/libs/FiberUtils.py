#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import multiprocessing
from os import getpid
import logging
import psutil
from dipy.tracking.streamline import compress_streamlines
from tractseg.libs.Utils import Utils
from dipy.segment.metric import ResampleFeature

logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
logging.getLogger().setLevel(logging.INFO)

#Global variables needed for shared memory of parallel fiber compression
global COMPRESSION_ERROR_THRESHOLD
COMPRESSION_ERROR_THRESHOLD = None
global FIBER_BATCHES
FIBER_BATCHES = None

# Worker Functions for multithreaded compression
def compress_fibers_worker_shared_mem(idx):
    # Function that runs in parallel must be on top level (not in class/function) otherwise it can not be pickled and then error
    streamlines_chunk = FIBER_BATCHES[idx]  # shared memory; by using indices each worker accesses only his part
    result = compress_streamlines(streamlines_chunk, tol_error=COMPRESSION_ERROR_THRESHOLD)
    logging.debug('PID {}, DONE'.format(getpid()))
    return result


class FiberUtils:

    @staticmethod
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
        fiber_batches = list(Utils.chunks(streamlines, chunk_size))

        global COMPRESSION_ERROR_THRESHOLD
        global FIBER_BATCHES
        COMPRESSION_ERROR_THRESHOLD = error_threshold
        FIBER_BATCHES = fiber_batches

        # logging.debug("Main program using: {} GB".format(round(Utils.mem_usage(print_usage=False), 3)))
        pool = multiprocessing.Pool(processes=nr_processes)

        #Do not pass data in (doubles amount of memory needed), but only idx of shared memory (needs only as much memory as single
        # thread version (only main thread needs memory, others almost 0).
        # Shared memory version also faster (around 20-30%?).
        # Needed otherwise memory problems when processing the raw tracking output (on disk >10GB and in memory >20GB)
        result = pool.map(compress_fibers_worker_shared_mem, range(0, len(fiber_batches)))

        pool.close()
        pool.join()

        streamlines_c = Utils.flatten(result)
        return streamlines_c

    @staticmethod
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

    @staticmethod
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
            streamlines = FiberUtils.compress_streamlines(streamlines, compress_err_thr, nr_cpus=nr_cpus)
        FiberUtils.save_streamlines_as_trk(filename_out, streamlines, reference_affine, reference_shape)

    @staticmethod
    def resample_fibers(streamlines, nb_points=12):
        streamlines_new = []
        for sl in streamlines:
            feature = ResampleFeature(nb_points=nb_points)
            streamlines_new.append(feature.extract(sl))
        return streamlines_new

