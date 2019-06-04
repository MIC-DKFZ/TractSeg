#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import randint
import multiprocessing
from functools import partial

from dipy.tracking.utils import move_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils
from tractseg.libs import dataset_utils


global _PEAKS
_PEAKS = None

global _BUNDLE_MASK
_BUNDLE_MASK = None

global _START_MASK
_START_MASK = None

global _END_MASK
_END_MASK = None



def process_seedpoint(seed_point, spacing):
    """

    Args:
        seed_point:
        spacing: Only one value. Assumes isotropic images.

    Returns:

    """
    def get_at_idx(img, idx):
        return img[int(idx[0]), int(idx[1]), int(idx[2])]
        # return img[int(idx[0]+0.5), int(idx[1]+0.5), int(idx[2]+0.5)]

    # Has to be sub-method otherwise not working
    def process_one_way(peaks, streamline, max_nr_steps, step_size, probabilistic, next_step_displacement_std,
                        max_tract_len, peak_len_thr, bundle_mask=None, reverse=False):
        last_dir = None
        sl_len = 0
        for i in range(max_nr_steps):
            last_point = streamline[-1]
            dir_raw = get_at_idx(peaks, (last_point[0], last_point[1], last_point[2]))
            if reverse and i == 0:
                dir_raw = -dir_raw  # inverse first step

            dir_raw_len = np.linalg.norm(dir_raw)
            # first normalize to length=1 then set to length of step_size
            dir_scaled = (dir_raw / (dir_raw_len + 1e-20)) * step_size
            dir_scaled = np.nan_to_num(dir_scaled)

            if i > 0:
                angle = np.dot(dir_scaled,
                               last_dir)  # not needed: (np.linalg.norm(dir_scaled) * np.linalg.norm(last_dir))
                if angle < 0:  # flip dir if not aligned with the direction of the streamline
                    dir_scaled = -dir_scaled

            if probabilistic:
                uncertainty = np.random.normal(0, next_step_displacement_std, 3)
                dir_scaled = dir_scaled + uncertainty

                # If step_size too small and next_step_displacement_std too big: sometimes even goes back
                #  -> ends up in random places (better since normalizing peak length after random displacing,
                #  but still happens if step_size to small)
                # dir_scaled = (dir_scaled / (np.linalg.norm(dir_scaled) + 1e-20)) * step_size

            next_point = streamline[-1] + dir_scaled
            last_dir = dir_scaled

            # stop fiber if running out of bundle mask
            if bundle_mask is not None:
                if get_at_idx(bundle_mask, (next_point[0], next_point[1], next_point[2])) == 0:
                    break

            # This does not take too much runtime, because most of these cases already caught by previous bundle_mask
            #  check. Here it is mainly only the good fibers (a lot less than all fibers seeded).
            next_peak_len = np.linalg.norm(get_at_idx(peaks, (next_point[0], next_point[1], next_point[2])))
            if next_peak_len < peak_len_thr:
                break
            else:
                if sl_len < max_tract_len:
                    streamline.append(next_point)
                    sl_len += dir_raw_len
                else:
                    break
        return streamline, sl_len

    def streamline_ends_in_masks(sl, start_mask, end_mask):
        if (get_at_idx(start_mask, (sl[0][0], sl[0][1], sl[0][2])) == 1 and
                get_at_idx(end_mask, (sl[-1][0], sl[-1][1], sl[-1][2])) == 1):
            return True

        if (get_at_idx(start_mask, (sl[-1][0], sl[-1][1], sl[-1][2])) == 1 and
                get_at_idx(end_mask, (sl[0][0], sl[0][1], sl[0][2])) == 1):
            return True

        return False


    # Good setting
    #  1.  step_size=0.7 and next_step_displacement_std=0.2   (8s)
    #  2.  step_size=0.5 and next_step_displacement_std=0.15  (10s)
    #  3.  step_size=0.5 and next_step_displacement_std=0.11  (11s)  -> not complete enough
    #   -> results very similar, but 1. a bit more complete + faster

    # Parameters
    probabilistic = True
    max_nr_steps = 1000
    min_tract_len = 50      # mm
    max_tract_len = 200     # mm
    # If step_size too small and next_step_displacement_std too big: sometimes even goes back -> ends up in random
    # places (better since normalizing peak length after random displacing, but still happens if step_size to small)
    step_size = 0.7  # relative to voxel size (=spacing)
    peak_len_thr = 0.1

    # transform length to voxel space
    min_tract_len = int(min_tract_len / spacing)
    max_tract_len = int(max_tract_len / spacing)

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
    global _BUNDLE_MASK
    bundle_mask = _BUNDLE_MASK
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

    streamline_part1, length_1 = process_one_way(peaks, streamline1, max_nr_steps, step_size, probabilistic,
                                   next_step_displacement_std, max_tract_len, peak_len_thr, bundle_mask, reverse=False)

    # Roughly doubles execution time but also roughly doubles number of resulting streamlines
    #   Makes sense because many too short if seeding in middle of streamline
    streamline_part2, length_2 = process_one_way(peaks, streamline2, max_nr_steps, step_size, probabilistic,
                                   next_step_displacement_std, max_tract_len, peak_len_thr, bundle_mask, reverse=True)

    if len(streamline_part2) > 0:
        # remove first element of part2 otherwise have seed_point 2 times
        streamline = list(reversed(streamline_part2[1:])) + streamline_part1
    else:
        streamline = streamline_part1

    # Check min and max length
    length = length_1 + length_2
    if length < min_tract_len or length > max_tract_len:
        return []

    # Filter by start and end mask
    if start_mask is not None and end_mask is not None:
        if streamline_ends_in_masks(streamline, start_mask, end_mask):
            return streamline

    return []


def seed_generator(mask_coords, nr_seeds):
    """
    Randomly select #nr_seeds voxels from mask.

    Args:
        mask_coords:
        nr_seeds:

    Returns:

    """
    nr_voxels = mask_coords.shape[0]
    random_indices = np.random.choice(nr_voxels, nr_seeds, replace=True)
    res = np.take(mask_coords, random_indices, axis=0)
    return res


def track(peaks, seed_image, max_nr_fibers=2000, smooth=None, compress=0.1, bundle_mask=None,
          start_mask=None, end_mask=None, dilation=1, nr_cpus=-1, verbose=True):
    """
    Great speedup was archived by:
    - only seeding in bundle_mask instead of entire image (seeding took very long)
    - calculating fiber length on the fly instead of using extra function which has to iterate over entire fiber a
    second time

    Args:
        peaks:
        seed_image:
        max_nr_fibers:
        peak_threshold:
        smooth:
        compress:
        bundle_mask:
        start_mask:
        end_mask:
        dilation:
        nr_cpus:
        verbose:

    Returns:

    """
    import psutil

    peaks[:, :, :, 0] *= -1  # how to flip along x axis to work properly
    if dilation > 0:
        # Add +1 dilation for start and end mask to be more robust
        start_mask = binary_dilation(start_mask, iterations=dilation+1).astype(np.uint8)
        end_mask = binary_dilation(end_mask, iterations=dilation+1).astype(np.uint8)
        bundle_mask = binary_dilation(bundle_mask, iterations=dilation).astype(np.uint8)

    global _PEAKS
    _PEAKS = peaks
    global _BUNDLE_MASK
    _BUNDLE_MASK = bundle_mask
    global _START_MASK
    _START_MASK = start_mask
    global _END_MASK
    _END_MASK = end_mask

    # Get list of coordinates of each voxel in mask to seed from those
    mask_coords = np.array(np.where(bundle_mask == 1)).transpose()
    nr_voxels = mask_coords.shape[0]
    spacing = seed_image.header.get_zooms()[0]

    # max_nr_seeds = 250 * max_nr_fibers
    max_nr_seeds = 100 * max_nr_fibers  # after how many seeds to abort (to avoid endless runtime)
    # How many seeds to process in each pool.map iteration
    seeds_per_batch = 5000

    if nr_cpus == -1:
        nr_processes = psutil.cpu_count()
    else:
        nr_processes = nr_cpus

    streamlines = []
    fiber_ctr = 0
    seed_ctr = 0
    # Processing seeds in batches so we can stop after we reached desired nr of streamlines. Not ideal. Could be
    #   optimised by more multiprocessing fanciness.
    while fiber_ctr < max_nr_fibers:
        pool = multiprocessing.Pool(processes=nr_processes)
        streamlines_tmp = pool.map(partial(process_seedpoint, spacing=spacing),
                                   seed_generator(mask_coords, seeds_per_batch))
        # streamlines_tmp = [process_seedpoint(seed, spacing=spacing) for seed in
        #                    seed_generator(mask_coords, seeds_per_batch)] # single threaded for debug
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

    streamlines = streamlines[:max_nr_fibers]   # remove surplus of fibers (comes from multiprocessing)
    streamlines = Streamlines(streamlines)  # Generate streamlines object

    # Move from origin being at the edge of the voxel to the origin being at the center of the voxel. Otherwise
    # tractogram and mask do not perfectly align when viewing in MITK, but are slightly offset.
    # We can still see a few fibers a little bit outside of mask because of big step size (no resegmenting done).
    streamlines = fiber_utils.add_to_each_streamline(streamlines, -0.5)

    # move streamlines to coordinate space
    streamlines = list(move_streamlines(streamlines, output_space=seed_image.affine))

    if smooth:
        streamlines = fiber_utils.smooth_streamlines(streamlines, smoothing_factor=smooth)

    if compress:
        streamlines = fiber_utils.compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=nr_cpus)

    return streamlines
