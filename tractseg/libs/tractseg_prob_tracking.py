
import psutil
import numpy as np
import multiprocessing
from functools import partial

from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

import inaccel_track

global _PEAKS
_PEAKS = None

global _BUNDLE_MASK
_BUNDLE_MASK = None

global _START_MASK
_START_MASK = None

global _END_MASK
_END_MASK = None

global _TRACKING_UNCERTAINTIES
_TRACKING_UNCERTAINTIES = None

def seed_generator(mask_coords, nr_seeds):
    """
    Randomly select #nr_seeds voxels from mask.
    """
    nr_voxels = mask_coords.shape[0]
    random_indices = np.random.choice(nr_voxels, nr_seeds, replace=True)
    res = np.take(mask_coords, random_indices, axis=0)
    return res


def track(peaks, max_nr_fibers=2000, smooth=None, compress=0.1, bundle_mask=None,
          start_mask=None, end_mask=None, tracking_uncertainties=None, dilation=0,
          next_step_displacement_std=0.15, nr_cpus=-1, affine=None, spacing=None, verbose=True):
    """
    Generate streamlines.

    Great speedup was archived by:
    - only seeding in bundle_mask instead of entire image (seeding took very long)
    - calculating fiber length on the fly instead of using extra function which has to iterate over entire fiber a
    second time
    """
    # If orientation is not same as MNI we flip the image to make it the same. Therefore we now also have to flip
    # the affine (which is used to map the peaks from world space to voxel space) the same way
    flip_axes = img_utils.get_flip_axis_to_match_MNI_space(affine)
    affine_MNI_ori = img_utils.flip_affine(affine, flip_axes)

    # Have to flip along x axis to work properly  (== moving HCP peaks to voxel spacing)
    # This works if no rotation in affine.
    # Not needed anymore because now doing properly with apply_rotation_to_peaks.
    # peaks[:, :, :, 0] *= -1  

    # Move peaks from world space (model predicts TOMs in world space because training data are 
    # also in world space) to voxel space. This flips the peaks (e.g. for HCP space a x-flip is needed to
    # make peaks align with array properly) and applies rotation from affine.
    # (Enough to move rotation to voxel space. Length anyways being normalize to 1 and offset does 
    # not matter for peak orientation.)
    peaks = img_utils.apply_rotation_to_peaks(peaks, affine_MNI_ori)

    # Add +1 dilation for start and end mask to be more robust
    start_mask = binary_dilation(start_mask, iterations=dilation + 1).astype(np.uint8)
    end_mask = binary_dilation(end_mask, iterations=dilation + 1).astype(np.uint8)
    if dilation > 0:
        bundle_mask = binary_dilation(bundle_mask, iterations=dilation).astype(np.uint8)

    if tracking_uncertainties is not None:
        tracking_uncertainties = img_utils.scale_to_range(tracking_uncertainties, range=(0, 1))

    global _PEAKS
    _PEAKS = peaks
    global _BUNDLE_MASK
    _BUNDLE_MASK = bundle_mask
    global _START_MASK
    _START_MASK = start_mask
    global _END_MASK
    _END_MASK = end_mask
    global _TRACKING_UNCERTAINTIES
    _TRACKING_UNCERTAINTIES = tracking_uncertainties

    # Get list of coordinates of each voxel in mask to seed from those
    mask_coords = np.array(np.where(bundle_mask == 1)).transpose()

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
        seeds = seed_generator(mask_coords, seeds_per_batch)
        streamlines_tmp = inaccel_track.pool_process_seedpoint(seeds,spacing, peaks, bundle_mask, start_mask, end_mask)      
              
        streamlines_tmp = [sl for sl in streamlines_tmp if len(sl) > 0]  # filter empty ones
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

    # Move from convention "0mm is in voxel corner" to convention "0mm is in voxel center". Most toolkits use the
    # convention "0mm is in voxel center".
    # We have to add 0.5 before applying affine otherwise 0.5 is not half a voxel anymore. Then we would have to add
    # half of the spacing and consider the sign of the affine (not needed here).
    streamlines = fiber_utils.add_to_each_streamline(streamlines, -0.5)

    # move streamlines from voxel space to coordinate space
    streamlines = list(transform_streamlines(streamlines, affine_MNI_ori))

    # If the original image was not in MNI space we have to flip back to the original space
    # before saving the streamlines.
    # This is not needed anymore when using affine_MNI_ori in transform_streamlines because this already 
    # contains the flipping.
    # for axis in flip_axes:
        # streamlines = fiber_utils.invert_streamlines(streamlines, bundle_mask, affine, axis=axis)

    # Smoothing does not change overall results at all because is just little smoothing. Just removes small unevenness.
    if smooth:
        streamlines = fiber_utils.smooth_streamlines(streamlines, smoothing_factor=smooth)

    if compress:
        streamlines = fiber_utils.compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=nr_cpus)

    return streamlines
