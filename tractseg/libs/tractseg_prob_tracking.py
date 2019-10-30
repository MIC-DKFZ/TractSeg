
import psutil
import numpy as np
import multiprocessing
from functools import partial

from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

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


def process_seedpoint(seed_point, spacing, next_step_displacement_std):
    """
    Create one streamline from one seed point.

    Args:
        seed_point: 3d point
        spacing: Only one value. Assumes isotropic images.
        next_step_displacement_std: stddev for gaussian distribution
    Returns:
        (streamline, streamline_length)
    """
    def get_at_idx(img, idx):
        return img[int(idx[0]), int(idx[1]), int(idx[2])]

    # Has to be sub-method otherwise not working
    def process_one_way(peaks, streamline, max_nr_steps, step_size, probabilistic, next_step_displacement_std,
                        max_tract_len, peak_len_thr, bundle_mask, tracking_uncertainties, reverse=False):
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
                angle = np.dot(dir_scaled, last_dir)
                if angle < 0:  # flip dir if not aligned with the direction of the streamline
                    dir_scaled = -dir_scaled

            if probabilistic:
                if tracking_uncertainties is not None:
                    uncertainty = get_at_idx(tracking_uncertainties, (last_point[0], last_point[1], last_point[2]))
                    # If maximal uncertainty we use full next_step_displacement_std. If minimal uncertainty we do not
                    # use any displacement
                    displacement_std_scaled = next_step_displacement_std * uncertainty
                else:
                    displacement_std_scaled = next_step_displacement_std
                displacement = np.random.normal(0, displacement_std_scaled, 3)
                dir_scaled = dir_scaled + displacement

                # If step_size too small and next_step_displacement_std too big: sometimes even goes back
                #  -> ends up in random places (better when normalizing peak length after random displacing,
                #  but still happens if step_size to small)
                # dir_scaled = (dir_scaled / (np.linalg.norm(dir_scaled) + 1e-20)) * step_size

            next_point = streamline[-1] + dir_scaled
            last_dir = dir_scaled

            # stop fiber if running out of image or out of bundle mask
            if bundle_mask is not None:
                sh = bundle_mask.shape
                if int(next_point[0]) >= sh[0] or int(next_point[1]) >= sh[1] or int(next_point[2]) >= sh[2]:
                    break
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

    # Parameters
    probabilistic = True
    max_nr_steps = 1000
    min_tract_len = 50  # mm
    max_tract_len = 200  # mm
    peak_len_thr = 0.1
    # If step_size too small and next_step_displacement_std too big: sometimes even goes back -> ends up in random
    # places (better when normalizing peak length after random displacing, but still happens if step_size to small)
    step_size = 0.7  # relative to voxel size (=spacing)

    # transform length to voxel space
    min_tract_len = int(min_tract_len / spacing)
    max_tract_len = int(max_tract_len / spacing)

    # Displacements are relative to voxel size. If you have bigger voxel size displacement is higher. Depends on
    # application if this is desired. Keep in mind.
    seedpoint_displacement_std = 0.15

    # If we want to set displacement in mm use this code:
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
    global _TRACKING_UNCERTAINTIES
    tracking_uncertainties = _TRACKING_UNCERTAINTIES

    streamline1 = []
    if probabilistic:
        random_seedpoint_displacement = np.random.normal(0, seedpoint_displacement_std, 3)
        seed_point = seed_point + random_seedpoint_displacement
    streamline1.append([seed_point[0], seed_point[1], seed_point[2]])  # add first point to streamline
    streamline2 = list(streamline1)  # deep copy

    streamline_part1, length_1 = process_one_way(peaks, streamline1, max_nr_steps, step_size, probabilistic,
                                                 next_step_displacement_std, max_tract_len, peak_len_thr, bundle_mask,
                                                 tracking_uncertainties, reverse=False)

    # Roughly doubles execution time but also roughly doubles number of resulting streamlines
    # Makes sense because many too short if seeding in middle of streamline.
    streamline_part2, length_2 = process_one_way(peaks, streamline2, max_nr_steps, step_size, probabilistic,
                                                 next_step_displacement_std, max_tract_len, peak_len_thr, bundle_mask,
                                                 tracking_uncertainties, reverse=True)

    if len(streamline_part2) > 0:
        # remove first element of part2 otherwise we have seed_point 2 times
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

    peaks[:, :, :, 0] *= -1  # have to flip along x axis to work properly
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
        pool = multiprocessing.Pool(processes=nr_processes)
        streamlines_tmp = pool.map(partial(process_seedpoint, next_step_displacement_std=next_step_displacement_std,
                                           spacing=spacing),
                                   seed_generator(mask_coords, seeds_per_batch))
        # streamlines_tmp = [process_seedpoint(seed, spacing=spacing) for seed in
        #                    seed_generator(mask_coords, seeds_per_batch)] # single threaded for debugging
        pool.close()
        pool.join()

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

    # move streamlines to coordinate space
    #  This is doing: streamlines(coordinate_space) = affine * streamlines(voxel_space)
    streamlines = list(transform_streamlines(streamlines, affine))

    # If the original image was not in MNI space we have to flip back to the original space
    # before saving the streamlines
    flip_axes = img_utils.get_flip_axis_to_match_MNI_space(affine)
    for axis in flip_axes:
        streamlines = fiber_utils.invert_streamlines(streamlines, bundle_mask, affine, axis=axis)

    # Smoothing does not change overall results at all because is just little smoothing. Just removes small unevenness.
    if smooth:
        streamlines = fiber_utils.smooth_streamlines(streamlines, smoothing_factor=smooth)

    if compress:
        streamlines = fiber_utils.compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=nr_cpus)

    return streamlines
