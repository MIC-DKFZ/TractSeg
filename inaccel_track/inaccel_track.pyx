from libc.math cimport fabs
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

import numpy as np

cdef float norm(float a, float b, float c):
    cdef float result = 0, abs_img
    abs_img = fabs(a)
    result += abs_img*abs_img
    abs_img = fabs(b)
    result += abs_img*abs_img
    abs_img = fabs(c)
    result += abs_img*abs_img

    result = sqrt(result)
    return result

#################################################################################
# Has to be sub-method otherwise not working
cdef int process_one_way(float* peaks, float* seed_point, float* random, float max_tract_len, unsigned int* bundle_mask, bint reverse, float* streamline, float* str_length):
    cdef float angle
    cdef float last_dir[3]
    cdef float dir_raw[3]
    cdef float last_point[3]
    cdef float  dir_scaled[3]
    cdef float next_point[3]
    cdef int MAX_NR_STEPS = 1000, MASK_SHAPE_0 = 73, MASK_SHAPE_1 = 87, MASK_SHAPE_2 = 73
    cdef float PEAK_LEN_THR = 0.1, STEP_SIZE = 0.7
    cdef int offset, count
    cdef float sl_len = 0
    cdef float tmp_0, tmp_1, tmp_2
    cdef unsigned int value
    cdef bint stop = False
    cdef int i = 0
    #cdef float displacement[3]
    streamline[0] = seed_point[0]
    streamline[1] = seed_point[1]
    streamline[2] = seed_point[2]
    last_point[0] = seed_point[0]
    last_point[1] = seed_point[1]
    last_point[2] = seed_point[2]
    count = 1

    #for i in range(MAX_NR_STEPS):
    while i < MAX_NR_STEPS and stop == False:
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
            dir_scaled[j] = dir_scaled[j] + random[((i+1)*3+j)%2000]

        for j in range(3):
            next_point[j] = last_point[j] + dir_scaled[j]
        for j in range(3):
            last_dir[j] = dir_scaled[j]

        # stop fiber if running out of image or out of bundle mask
        if int(next_point[0]) >= MASK_SHAPE_0 or int(next_point[1]) >= MASK_SHAPE_1 or int(next_point[2]) >= MASK_SHAPE_2:
            stop = True
            #break
        offset = (<int>next_point[0])*MASK_SHAPE_2*MASK_SHAPE_1 + (<int>next_point[1])*MASK_SHAPE_2 + (<int>next_point[2])
        value = bundle_mask[offset]
        if value == 0:
            stop = True
            #break

        # This does not take too much runtime, because most of these cases already caught by previous bundle_mask
        #  check. Here it is mainly only the good fibers (a lot less than all fibers seeded).

        offset = (<int>next_point[0])*3*MASK_SHAPE_2*MASK_SHAPE_1 + (<int>next_point[1])*3*MASK_SHAPE_2 + (<int>next_point[2])*3
        tmp_0 = peaks[offset + 0]
        tmp_1 = peaks[offset + 1]
        tmp_2 = peaks[offset + 2]
        next_peak_len = norm(tmp_0, tmp_1, tmp_2)

        if next_peak_len < PEAK_LEN_THR:
            stop = True
            #break
        elif stop == False:
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
                stop = True
                #break
        i = i + 1

    str_length[0] = sl_len
    return count

cdef bint streamline_ends_in_masks(float* first,float* last,unsigned int* start_mask,unsigned int* end_mask):
    cdef int MASK_SHAPE_1 = 87, MASK_SHAPE_2 = 73
    cdef int offset_first = int(first[0])*MASK_SHAPE_2*MASK_SHAPE_1 + int(first[1])*MASK_SHAPE_2 + int(first[2])
    cdef int offset_last = int(last[0])*MASK_SHAPE_2*MASK_SHAPE_1 + int(last[1])*MASK_SHAPE_2 + int(last[2])
    cdef unsigned int start_mask_f, start_mask_l, end_mask_f, end_mask_l
    start_mask_f = start_mask[offset_first]
    start_mask_l = start_mask[offset_last]
    end_mask_f = end_mask[offset_last]
    end_mask_l = end_mask[offset_first]
    if (start_mask_f == 1 and end_mask_f == 1):
        return True

    if (start_mask_l == 1 and end_mask_l == 1):
        return True

    return False

cdef int process_seedpoint(float* seed_point,float spacing,float* peaks,unsigned int* bundle_mask,unsigned int* start_mask,unsigned int* end_mask,float* random, float* streamline_c):
    """
    Create one streamline from one seed point.

    Args:
        seed_point: 3d point
        spacing: Only one value. Assumes isotropic images.
        next_step_displacement_std: stddev for gaussian distribution
    Returns:
        (streamline, streamline_length)
    """

    # Parameters
    cdef int min_tract_len = 50  # mm
    cdef int max_tract_len = 200  # mm
    cdef int total_count = 0
    cdef int cnt = 0
    cdef float start_point[3]
    cdef float end_point[3]

    cdef float streamline_part1[250]
    cdef float streamline_part2[250]
    cdef float length_1[1], length_2[1]
    cdef int count_1, count_2
    cdef float point_0, point_1, point_2

    # transform length to voxel space
    min_tract_len = <int>(min_tract_len / spacing)
    max_tract_len = <int>(max_tract_len / spacing)

    # Displacements are relative to voxel size. If you have bigger voxel size displacement is higher. Depends on
    # application if this is desired. Keep in mind.

    for i in range(3):
        seed_point[i] = seed_point[i]  + random[i]

    count_1 = process_one_way(peaks, seed_point, random, max_tract_len, bundle_mask, False, streamline_part1, length_1)

    # Roughly doubles execution time but also roughly doubles number of resulting streamlines
    # Makes sense because many too short if seeding in middle of streamline.
    count_2 = process_one_way(peaks, seed_point, random, max_tract_len, bundle_mask, True, streamline_part2, length_2)

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
    total_count = total_count + count_1 - 1

    # Filter by start and end mask
    for i in range(3):
        start_point[i] = streamline_c[i]
        end_point[i] = streamline_c[(total_count - 1)*3 + i]
    if streamline_ends_in_masks(start_point, end_point, start_mask, end_mask):
        return total_count

    return 0


cdef void pool(float* seeds, float spacing, float* peaks, unsigned int* bundle_mask, unsigned int* start_mask, unsigned int* end_mask, float* random, float* streamline_c, int* total_count):
    for k in range(5000):
        total_count[k] = process_seedpoint(&seeds[k*3], spacing, peaks, bundle_mask, start_mask, end_mask, random, &streamline_c[k*250])
    return

def pool_process_seedpoint(np_seeds, spacing, np_peaks, np_bundle_mask, np_start_mask, np_end_mask):
    cdef int i;
    cdef int count = 0;
    num_points_each = []
    streamlines = []

    cdef float* seeds = <float*>malloc(5000*3*sizeof(float))
    cdef float* peaks = <float*>malloc(73*87*73*3*sizeof(float))
    cdef unsigned int* bundle_mask = <unsigned int*>malloc(73*87*73*sizeof(unsigned int))
    cdef unsigned int* start_mask = <unsigned int*>malloc(73*87*73*sizeof(unsigned int))
    cdef unsigned int* end_mask = <unsigned int*>malloc(73*87*73*sizeof(unsigned int))
    cdef float* streamline_c = <float*>malloc(5000*250*sizeof(float))
    cdef int total_count[5000]

    cdef float random[2000]

    np_peaks = np_peaks.reshape(73*87*73*3)
    np_bundle_mask = np_bundle_mask.reshape(73*87*73)
    np_start_mask = np_start_mask.reshape(73*87*73)
    np_end_mask = np_end_mask.reshape(73*87*73)
    np_seeds = np_seeds.reshape(5000*3)

    for i in range(73*87*73*3):
        peaks[i] = np_peaks[i]
    for i in range(73*87*73):
        bundle_mask[i] = np_bundle_mask[i]
        start_mask[i] = np_start_mask[i]
        end_mask[i] = np_end_mask[i]
    for i in range(5000*3):
        seeds[i] = np_seeds[i]

    for i in range(2000):
        random[i] = np.random.normal(0, 0.15, 1)

    pool(seeds, spacing, peaks, bundle_mask, start_mask, end_mask, random, streamline_c, total_count)

    for k in range(5000):
        streamline = []
        for i in range(total_count[k]):
            streamline.append([streamline_c[k*250 + i*3 + 0], streamline_c[k*250 + i*3 + 1], streamline_c[k*250 + i*3 + 2]])
        streamlines.append(streamline)

    free(seeds)
    free(peaks)
    free(bundle_mask)
    free(start_mask)
    free(end_mask)

    return streamlines
