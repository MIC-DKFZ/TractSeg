import psutil
import numpy as np
import multiprocessing
from functools import partial

from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

################ Cython code START ################

from libc.math cimport sqrt
from libc.stdlib cimport malloc

cimport numpy as cnp
import ctypes

cdef float fabs(double a) nogil:
    if a >= 0:
        return a
    return -a

cdef float norm(double a, double b, double c) nogil:
    cdef double result = 0, abs_img
    abs_img = fabs(a)
    result += abs_img*abs_img
    abs_img = fabs(b)
    result += abs_img*abs_img
    abs_img = fabs(c)
    result += abs_img*abs_img

    result = sqrt(result)
    return result

cdef int process_one_way(double* peaks, double* seed_point, double* random, double max_tract_len, unsigned char* bundle_mask, bint reverse, double* streamline, double* str_length, int MASK_SHAPE_0, int MASK_SHAPE_1, int MASK_SHAPE_2) nogil:
    cdef double angle
    cdef double last_dir[3]
    cdef double dir_raw[3]
    cdef double last_point[3]
    cdef double  dir_scaled[3]
    cdef double next_point[3]
    cdef int MAX_NR_STEPS = 1000
    cdef float PEAK_LEN_THR = 0.1
    cdef float STEP_SIZE = 0.7
    cdef int offset
    cdef int count
    cdef double sl_len = 0
    cdef double tmp_0
    cdef double tmp_1
    cdef double tmp_2
    cdef unsigned char value
    cdef int i
    cdef int j
    cdef double next_peak_len
    cdef double dir_raw_len
    #cdef double displacement[3]
    streamline[0] = seed_point[0]
    streamline[1] = seed_point[1]
    streamline[2] = seed_point[2]
    last_point[0] = seed_point[0]
    last_point[1] = seed_point[1]
    last_point[2] = seed_point[2]
    count = 1

    for i in range(MAX_NR_STEPS):
        offset = (<int>last_point[0])*3*MASK_SHAPE_2*MASK_SHAPE_1 + (<int>last_point[1])*3*MASK_SHAPE_2 + (<int>last_point[2])*3
        for j in range(3):
            dir_raw[j] = peaks[offset + j]
        if reverse and i == 0:
            for j in range(3):
                dir_raw[j] = -dir_raw[j]  # inverse first step

        dir_raw_len = norm(dir_raw[0], dir_raw[1], dir_raw[2])
        # first normalize to length=1 then set to length of step_size
        for j in range(3):
            dir_scaled[j] = (dir_raw[j] / (dir_raw_len + 1e-20)) * STEP_SIZE

        if i > 0:
            angle = 0.0
            for j in range(3):
                angle = angle + dir_scaled[j]*last_dir[j];

            if angle < 0:  # flip dir if not aligned with the direction of the streamline
                for j in range(3):
                    dir_scaled[j] = -dir_scaled[j]

        for j in range(3):
            dir_scaled[j] = dir_scaled[j] + random[((i+1)*3+j)%1000]

        for j in range(3):
            next_point[j] = last_point[j] + dir_scaled[j]
        for j in range(3):
            last_dir[j] = dir_scaled[j]

        # stop fiber if running out of image or out of bundle mask
        if <int>(next_point[0]) >= MASK_SHAPE_0 or <int>(next_point[1]) >= MASK_SHAPE_1 or <int>(next_point[2]) >= MASK_SHAPE_2:
            break
        offset = (<int>next_point[0])*MASK_SHAPE_2*MASK_SHAPE_1 + (<int>next_point[1])*MASK_SHAPE_2 + (<int>next_point[2])
        value = bundle_mask[offset]
        if value == 0:
            break

        # This does not take too much runtime, because most of these cases already caught by previous bundle_mask
        #  check. Here it is mainly only the good fibers (a lot less than all fibers seeded).

        offset = (<int>next_point[0])*3*MASK_SHAPE_2*MASK_SHAPE_1 + (<int>next_point[1])*3*MASK_SHAPE_2 + (<int>next_point[2])*3
        tmp_0 = peaks[offset + 0]
        tmp_1 = peaks[offset + 1]
        tmp_2 = peaks[offset + 2]
        next_peak_len =  norm(tmp_0, tmp_1, tmp_2)

        if next_peak_len < PEAK_LEN_THR:
            break
        else:
            if sl_len < max_tract_len:
                streamline[count*3 + 0] = next_point[0]
                streamline[count*3 + 1] = next_point[1]
                streamline[count*3 + 2] = next_point[2]
                last_point[0] = next_point[0]
                last_point[1] = next_point[1]
                last_point[2] = next_point[2]
                sl_len += dir_raw_len
                count = count + 1
            else:
                break

    str_length[0] = sl_len
    return count

cdef bint streamline_ends_in_masks(double* first,double* last,unsigned char* start_mask,unsigned char* end_mask, int MASK_SHAPE_1, int MASK_SHAPE_2) nogil:
    cdef int offset_first = <int>(first[0])*MASK_SHAPE_2*MASK_SHAPE_1 + <int>(first[1])*MASK_SHAPE_2 + <int>(first[2])
    cdef int offset_last = <int>(last[0])*MASK_SHAPE_2*MASK_SHAPE_1 + <int>(last[1])*MASK_SHAPE_2 + <int>(last[2])
    cdef unsigned char start_mask_f, start_mask_l, end_mask_f, end_mask_l
    start_mask_f = start_mask[offset_first]
    start_mask_l = start_mask[offset_last]
    end_mask_f = end_mask[offset_last]
    end_mask_l = end_mask[offset_first]
    if (start_mask_f == 1 and end_mask_f == 1):
        return True

    if (start_mask_l == 1 and end_mask_l == 1):
        return True

    return False

cdef int process_seedpoint(double* seed_point,double spacing,double* peaks,unsigned char* bundle_mask,unsigned char* start_mask,unsigned char* end_mask,double* random, double* streamline_c, int MASK_SHAPE_0, int MASK_SHAPE_1, int MASK_SHAPE_2) nogil:
    """
    Create one streamline from one seed point.

    Args:
        seed_point: 3d point
        spacing: Only one value. Assumes isotropic images.
        peaks: double array
        bundle_mask: unsigned char array
        start_mask: unsigned char array
        end_mask: unsigned char array
        random: double array with 1000 random normal values with mean = 0.0 and standard deviation = 0.15
        streamline_c: double array on which will be stored the points of the generated streamline
        MASK_SHAPE_0: 1rst dimension of mask
        MASK_SHAPE_1: 2nd dimension of mask
        MASK_SHAPE_2: 3rd dimension of mask
    Returns:
        The total number of 3d points of the generated streamline.
    """

    # Parameters
    cdef int min_tract_len = 50  # mm
    cdef int max_tract_len = 200  # mm
    cdef int total_count = 0
    cdef int cnt = 0
    cdef int i
    cdef double start_point[3]
    cdef double end_point[3]

    cdef double streamline_part1[3000]
    cdef double streamline_part2[3000]
    cdef double length_1[1]
    cdef double length_2[1]
    cdef int count_1, count_2
    cdef double point_0, point_1, point_2

    # transform length to voxel space
    min_tract_len = <int>(min_tract_len / spacing)
    max_tract_len = <int>(max_tract_len / spacing)

    # Displacements are relative to voxel size. If you have bigger voxel size displacement is higher. Depends on
    # application if this is desired. Keep in mind.

    for i in range(3):
        seed_point[i] = seed_point[i]  + random[i]

    count_1 = process_one_way(peaks, seed_point, random, max_tract_len, bundle_mask, False, streamline_part1, length_1, MASK_SHAPE_0, MASK_SHAPE_1, MASK_SHAPE_2)

    # Roughly doubles execution time but also roughly doubles number of resulting streamlines
    # Makes sense because many too short if seeding in middle of streamline.
    count_2 = process_one_way(peaks, seed_point, random, max_tract_len, bundle_mask, True, streamline_part2, length_2, MASK_SHAPE_0, MASK_SHAPE_1, MASK_SHAPE_2)

    # Check min and max length
    length = length_1[0] + length_2[0]
    if length < min_tract_len or length > max_tract_len:
        return 0

    if count_2 > 1:
        for i in range(count_2 - 1, 0, -1):
            # remove first element of part2 otherwise we have seed_point 2 times
            streamline_c[cnt] = streamline_part2[i*3 + 0]
            streamline_c[cnt + 1] = streamline_part2[i*3 + 1]
            streamline_c[cnt + 2] = streamline_part2[i*3 + 2]
            cnt = cnt + 3
        total_count = count_2 - 1

    for i in range(count_1):
        streamline_c[cnt] = streamline_part1[i*3 + 0]
        streamline_c[cnt + 1] = streamline_part1[i*3 + 1]
        streamline_c[cnt + 2] = streamline_part1[i*3 + 2]
        cnt = cnt + 3
    total_count = total_count + count_1

    # Filter by start and end mask
    for i in range(3):
        start_point[i] = streamline_c[i]
        end_point[i] = streamline_c[(total_count - 1)*3 + i]
    if streamline_ends_in_masks(start_point, end_point, start_mask, end_mask, MASK_SHAPE_1, MASK_SHAPE_2):
        return total_count

    return 0


def pool_process_seedpoint(np_seeds, spacing, np_peaks, np_bundle_mask, np_start_mask, np_end_mask):
    """
    Create one streamline for each seed point.

    Args:
        np_seeds: numpy array with 5000 3d-points
        spacing: Only one value. Assumes isotropic images.
        peaks: numpy array
        bundle_mask: numpy array
        start_mask: numpy array
        end_mask: numpy array
    Returns:
        List with all the generated streamlines
    """
    cdef Py_ssize_t k
    cdef Py_ssize_t i
    cdef int MASK_SHAPE_0
    cdef int MASK_SHAPE_1
    cdef int MASK_SHAPE_2

    # Initialization of the output list
    streamlines = []

    # Allocate memory for all the streamlines
    cdef double* streamline_c = <double*>malloc(5000*3000*sizeof(double))

    # Array on which will be stored the total number of 3d-points of each streamline
    cdef int total_count[5000]

    MASK_SHAPE_0 = <int> np_bundle_mask.shape[0]
    MASK_SHAPE_1 = <int> np_bundle_mask.shape[1]
    MASK_SHAPE_2 = <int> np_bundle_mask.shape[2]

    # Reshape each numpy array to 1d
    np_peaks = np_peaks.reshape(-1)
    np_bundle_mask = np_bundle_mask.reshape(-1)
    np_start_mask = np_start_mask.reshape(-1)
    np_end_mask = np_end_mask.reshape(-1)
    np_seeds = np_seeds.reshape(-1)

    # Generate random normal values with mean = 0.0 and standard deviation = 0.15
    # we generate 1000 random normal values for each seed point (in total 5000 seed points)
    rand = np.random.normal(0, 0.15, (5000*1000))

    # Get the c pointers of the numpy arrays
    cdef cnp.ndarray[double,mode="c"] buff_peaks = np_peaks
    cdef double* peaks = &buff_peaks[0]

    cdef cnp.ndarray[double,mode="c"] buff_seeds = np.array(np_seeds,dtype=np.float64)
    cdef double* seeds = &buff_seeds[0]

    cdef cnp.ndarray[unsigned char,mode="c"] buff_bundle_mask = np_bundle_mask
    cdef unsigned char* bundle_mask = &buff_bundle_mask[0]

    cdef cnp.ndarray[unsigned char,mode="c"] buff_start_mask = np_start_mask
    cdef unsigned char* start_mask = &buff_start_mask[0]

    cdef cnp.ndarray[unsigned char,mode="c"] buff_end_mask = np_end_mask
    cdef unsigned char* end_mask = &buff_end_mask[0]

    cdef cnp.ndarray[double,mode="c"] buff_rand = rand
    cdef double* random = &buff_rand[0]

    cdef float spacing_c = spacing

    for k in range(5000):
        total_count[k] = process_seedpoint(&seeds[k*3], spacing_c, peaks, bundle_mask, start_mask, end_mask, &random[k*1000], &streamline_c[k*3000], MASK_SHAPE_0, MASK_SHAPE_1, MASK_SHAPE_2)

    # Create the python list with all the generated streamlines
    for k in range(5000):
        if total_count[k] > 0:
            streamline = np.ndarray((total_count[k],3), dtype=np.float64)
            for i in range(total_count[k]):
                streamline[i][0] = streamline_c[k*3000 + i*3 + 0] - 0.5
                streamline[i][1] = streamline_c[k*3000 + i*3 + 1] - 0.5
                streamline[i][2] = streamline_c[k*3000 + i*3 + 2] - 0.5
            streamlines.append(streamline)

    return streamlines


################ Cython code END ################

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

    # Get list of coordinates of each voxel in mask to seed from those
    mask_coords = np.array(np.where(bundle_mask == 1)).transpose()

    max_nr_seeds = 100 * max_nr_fibers  # after how many seeds to abort (to avoid endless runtime)
    # How many seeds to process in each pool.map iteration
    seeds_per_batch = 5000

    streamlines = []
    fiber_ctr = 0
    seed_ctr = 0
    # Processing seeds in batches so we can stop after we reached desired nr of streamlines. Not ideal. Could be
    #   optimised by more multiprocessing fanciness.
    while fiber_ctr < max_nr_fibers:
        seeds = seed_generator(mask_coords, seeds_per_batch)
        streamlines_tmp = pool_process_seedpoint(seeds,spacing, peaks, bundle_mask, start_mask, end_mask)

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
    # This part is done at the generation of the python list at pool_process_seedpoint function
    #streamlines = fiber_utils.add_to_each_streamline(streamlines, -0.5)

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
