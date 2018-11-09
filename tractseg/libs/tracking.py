#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import randint
import multiprocessing
import psutil

from dipy.tracking.utils import move_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils


global _PEAKS
_PEAKS = None

global _START_MASK
_START_MASK = None

global _END_MASK
_END_MASK = None



def process_seedpoint(seed_point):
    """

    Args:
        peaks:
        seed_point:
        spacing: Only one value. Assumes isotropic images.

    Returns:

    """

    # Has to be sub-method otherwise not working
    def process_one_way(peaks, streamline, max_nr_steps, step_size, probabilistic, next_step_displacement_std,
                        max_tract_len, peak_len_thr, reverse=False):
        last_dir = None
        for i in range(max_nr_steps):
            last_point = streamline[-1]
            dir_raw = peaks[int(last_point[0]), int(last_point[1]), int(last_point[2])]
            if reverse and i == 0:
                dir_raw = -dir_raw  # inverse first step

            # first normalize to length=1 then set to length of step_size
            dir_scaled = (dir_raw / (np.linalg.norm(dir_raw) + 1e-20)) * step_size
            dir_scaled = np.nan_to_num(dir_scaled)

            if i > 0:
                angle = np.dot(dir_scaled,
                               last_dir)  # not needed: (np.linalg.norm(dir_scaled) * np.linalg.norm(last_dir))
                if angle < 0:  # flip dir if not aligned with the direction of the streamline
                    dir_scaled = -dir_scaled

            if probabilistic:
                uncertainty = np.random.normal(0, next_step_displacement_std, 3)
                dir_scaled = dir_scaled + uncertainty

            next_point = streamline[-1] + dir_scaled
            last_dir = dir_scaled

            if np.linalg.norm(peaks[int(next_point[0]), int(next_point[1]), int(next_point[2])]) < peak_len_thr:
                break
            else:
                lengths, spaces = fiber_utils.get_streamline_statistics([streamline], raw=True)  # filter by length
                # check max length
                if lengths[0] < max_tract_len:
                    streamline.append(next_point)
                else:
                    break
        return streamline

    def streamline_ends_in_masks(sl, start_mask, end_mask):
        if (start_mask[int(sl[0][0]), int(sl[0][1]), int(sl[0][2])] == 1 and
                end_mask[int(sl[-1][0]), int(sl[-1][1]), int(sl[-1][2])] == 1):
            return True

        if (start_mask[int(sl[-1][0]), int(sl[-1][1]), int(sl[-1][2])] == 1 and
                end_mask[int(sl[0][0]), int(sl[0][1]), int(sl[0][2])] == 1):
            return True

        return False


    # Parameters
    probabilistic = True
    max_nr_steps = 1000
    min_tract_len = 20
    max_tract_len = 200
    step_size = 0.5  # 1 = voxel size (=spacing)
    peak_len_thr = 0.1

    # Displacements are relative to voxel size. If you have bigger voxel size displacement is higher. Depends on
    #   application if this is desired. Keep in mind.
    seedpoint_displacement_std = 0.15
    next_step_displacement_std = 0.2
    # If we want to set displacement in mm use this code:
    # seedpoint_displacement_std = 0.3    # mm
    # next_step_displacement_std = 0.4    # mm
    # seedpoint_displacement_std = seedpoint_displacement_std / spacing
    # next_step_displacement_std = next_step_displacement_std / spacing

    global _PEAKS
    peaks = _PEAKS
    global _START_MASK
    start_mask = _START_MASK
    global _END_MASK
    end_mask = _END_MASK

    streamline1 = []
    if probabilistic:
        random_seedpoint_displacement = np.random.normal(0, seedpoint_displacement_std, 3)
        seed_point = seed_point + random_seedpoint_displacement
    streamline1.append([seed_point[0], seed_point[1], seed_point[2]])  # add first point to streamline
    streamline2 = list(streamline1)  # deep copy

    streamline_part1 = process_one_way(peaks, streamline1, max_nr_steps, step_size, probabilistic,
                                   next_step_displacement_std, max_tract_len, peak_len_thr, reverse=False)

    # Roughly doubles execution time but also roughly doubles number of resulting streamlines
    #   Makes sense because many too short if seeding in middle of streamline
    streamline_part2 = process_one_way(peaks, streamline2, max_nr_steps, step_size, probabilistic,
                                   next_step_displacement_std, max_tract_len, peak_len_thr, reverse=True)
    # streamline_part2 = []

    if len(streamline_part2) > 0:
        # remove first element of part2 otherwise have seed_point 2 times
        streamline = list(reversed(streamline_part2[1:])) + streamline_part1
    else:
        streamline = streamline_part1

    # Check min and max length
    lengths, spaces = fiber_utils.get_streamline_statistics([streamline], raw=True)
    if lengths[0] < min_tract_len or lengths[0] > max_tract_len:
        return []

    # Filter by mask
    if start_mask is not None and end_mask is not None:
        if streamline_ends_in_masks(streamline, start_mask, end_mask):
            return streamline

    return []


def seed_generator(peaks, nr_seeds, seed_mask_shape, peak_threshold):
    ctr = 0
    while ctr < nr_seeds:
        x = randint(0, seed_mask_shape[0] - 1)
        y = randint(0, seed_mask_shape[1] - 1)
        z = randint(0, seed_mask_shape[2] - 1)
        if np.linalg.norm(peaks[x, y, z]) > peak_threshold:
            yield [x, y, z]
        ctr += 1


def track(peaks, seed_image, max_nr_fibers=2000, peak_threshold=0.01, smooth=None, start_mask=None,
          end_mask=None, dilation=1, verbose=True):
    """

    Args:
        peaks:
        seed_image:
        max_nr_fibers:
        peak_threshold:
        smooth:
        start_mask:
        end_mask:
        dilation:
        verbose:

    Returns:

    """
    peaks[:, :, :, 0] *= -1  # how to flip along x axis to work properly
    if dilation > 0:
        start_mask = binary_dilation(start_mask, iterations=dilation).astype(np.uint8)
        end_mask = binary_dilation(end_mask, iterations=dilation).astype(np.uint8)

    global _PEAKS
    _PEAKS = peaks
    global _START_MASK
    _START_MASK = start_mask
    global _END_MASK
    _END_MASK = end_mask

    # spacing = seed_image.header.get_zooms()[0]

    p_shape = seed_image.get_data().shape

    max_nr_seeds = 3000000  # after how many seeds to abort (to avoid endless runtime)
    if start_mask is not None:
        # use higher number, because we find less valid fibers -> faster processing
        seeds_per_batch = 100000  # how many seeds to process in each pool.map iteration
    else:
        seeds_per_batch = 20000  # how many seeds to process in each pool.map iteration

    nr_processes = psutil.cpu_count()

    streamlines = []
    fiber_ctr = 0
    seed_ctr = 0
    # Processing seeds in batches to we can stop after we reached desired nr of streamlines. Not ideal. Could be
    #   optimised if more familiar with multiprocessing.
    while fiber_ctr < max_nr_fibers:
        pool = multiprocessing.Pool(processes=nr_processes)
        streamlines_tmp = pool.map(process_seedpoint, seed_generator(peaks, seeds_per_batch, p_shape, peak_threshold))
        # streamlines_tmp = [process_seedpoint(seed) for seed in
        #                    seed_generator(peaks, seeds_per_batch, p_shape, peak_threshold)] # single threaded for debug
        pool.close()
        pool.join()

        streamlines_tmp = [sl for sl in streamlines_tmp if len(sl) > 0]  # filter empty
        streamlines += streamlines_tmp
        fiber_ctr = len(streamlines)
        if verbose:
            print("nr_fibs: {}".format(fiber_ctr))
        seed_ctr += seeds_per_batch
        if seed_ctr > max_nr_seeds:
            if verbose:
                print("Early stopping because max nr of seeds reached.")
            break

    if verbose:
        print("final nr streamlines: {}".format(len(streamlines)))

    streamlines = Streamlines(streamlines)  # Generate streamlines object
    # move streamlines to coordinate space
    streamlines = list(move_streamlines(streamlines, seed_image.get_affine()))

    if smooth:
        streamlines = fiber_utils.smooth_streamlines(streamlines, smoothing_factor=smooth)

    return streamlines
