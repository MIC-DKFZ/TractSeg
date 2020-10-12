
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from os import getpid
import numpy as np
import nibabel as nib
from dipy.tracking.streamline import compress_streamlines as compress_streamlines_dipy
from dipy.segment.metric import ResampleFeature
from dipy.tracking.metrics import spline
from dipy.tracking import utils as utils_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import length as sl_length

from tractseg.libs import utils
from tractseg.libs import peak_utils

# Global variables needed for shared memory of parallel fiber compression
global _COMPRESSION_ERROR_THRESHOLD
_COMPRESSION_ERROR_THRESHOLD = None
global _FIBER_BATCHES
_FIBER_BATCHES = None


def compress_fibers_worker_shared_mem(idx):
    """
    Worker Functions for multithreaded compression.

    Function that runs in parallel must be on top level (not in class/function) otherwise it can
    not be pickled.
    """
    streamlines_chunk = _FIBER_BATCHES[idx]  # shared memory; by using indices each worker accesses only his part
    result = compress_streamlines_dipy(streamlines_chunk, tol_error=_COMPRESSION_ERROR_THRESHOLD)
    # print('PID {}, DONE'.format(getpid()))
    return result


def compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=-1):
    import psutil
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
        return streamlines
    fiber_batches = list(utils.chunks(streamlines, chunk_size))

    global _COMPRESSION_ERROR_THRESHOLD
    global _FIBER_BATCHES
    _COMPRESSION_ERROR_THRESHOLD = error_threshold
    _FIBER_BATCHES = fiber_batches

    # print("Main program using: {} GB".format(round(Utils.mem_usage(print_usage=False), 3)))
    pool = multiprocessing.Pool(processes=nr_processes)

    #Do not pass in data (doubles amount of memory needed), but only idx of shared memory
    #  (needs only as much memory as single thread version (only main thread needs memory, others almost 0).
    #  Shared memory version also faster (around 20-30%?).
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
    affine = np.abs(affine)
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

    todo: use dipy.io.streamline.save_tractogram to save streamlines

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

    # Create a new header with the correct affine and nr of streamlines
    hdr = {}
    hdr['voxel_sizes'] = vox_sizes
    hdr['voxel_order'] = vox_order
    hdr['dimensions'] = shape
    hdr['voxel_to_rasmm'] = affine
    hdr['nb_streamlines'] = len(streamlines)

    nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), out_file, header=hdr)


def convert_tck_to_trk(filename_in, filename_out, reference_affine, reference_shape,
                       compress_err_thr=0.1, smooth=None, nr_cpus=-1, tracking_format="trk_legacy"):

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


def create_empty_tractogram(filename_out, reference_file,
                            tracking_format="trk_legacy"):

    ref_img = nib.load(reference_file)
    reference_affine = ref_img.affine
    reference_shape = ref_img.get_fdata().shape[:3]

    streamlines = []

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
    Smooth streamlines

    Args:
        streamlines: list of streamlines
        smoothing_factor: 10: slight smoothing,  100: very smooth from beginning to end

    Returns:
        smoothed streamlines
    """
    streamlines_smooth = []
    for sl in streamlines:
        streamlines_smooth.append(spline(sl, s=smoothing_factor))
    return streamlines_smooth


def get_streamline_statistics(streamlines, subsample=False, raw=False):
    """
    Get streamlines statistics in mm

    Args:
        streamlines: list of streamlines
        subsample: Do not evaluate all points to increase runtime
        raw: if True returns list of fibers length and spaces

    Returns:
        (mean streamline length, mean space between two following points, max space between two following points)
    """
    if subsample:  # subsample for faster processing
        STEP_SIZE = 20
    else:
        STEP_SIZE = 1

    lengths = []
    spaces = []  # spaces between 2 points
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
        return lengths, spaces
    else:
        return np.array(lengths).mean(), np.array(spaces).mean(), np.array(spaces).max()


def filter_streamlines_leaving_mask(streamlines, mask):
    """
    Remove all streamlines that exit the mask
    """
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

    def _get_most_aligned_peak(pred, orig):
        orig = np.array(orig)
        angle1 = abs(peak_utils.angle_last_dim(pred, orig[0]))
        angle2 = abs(peak_utils.angle_last_dim(pred, orig[1]))
        angle3 = abs(peak_utils.angle_last_dim(pred, orig[2]))
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

    best_orig = _get_most_aligned_peak(peaks_pred,
                                      [peaks_orig[:, :, :, 0:3],
                                       peaks_orig[:, :, :, 3:6],
                                       peaks_orig[:, :, :, 6:9]])
    return best_orig


def get_weighted_mean_of_peaks(best_orig, tom, weight=0.5):
    """
    Calculate weighted mean between best_orig peaks and TOM peaks.

    Args:
        best_orig: original peaks
        tom: prior
        weight: how much to apply prior (0: only original signal, 1: only prior)

    Returns:
        weighted mean
    """
    angles = peak_utils.angle_last_dim(best_orig, tom)
    # make sure to take mean along smaller angle (<90 degree), not along the bigger one (>90 degree)
    tom[angles < 0] *= -1   # flip peak
    stacked = np.stack([best_orig, tom])
    return np.average(stacked, axis=0, weights=[1 - weight, weight])


def add_to_each_streamline(streamlines, scalar):
    """
    Add scalar value to each coordinate of each streamline
    """
    sl_new = []
    for sl in streamlines:
        sl_new.append(np.array(sl) + scalar)
    return sl_new


def add_to_each_streamline_axis(streamlines, scalar, axis="x"):
    sl_new = []
    for sl in streamlines:
        s = np.array(sl)
        if axis == "x":
            s[:, 0] += scalar
        elif axis == "y":
            s[:, 1] += scalar
        elif axis == "z":
            s[:, 2] += scalar
        sl_new.append(s)
    return sl_new


def flip(streamlines, axis="x"):
    new_sl = []
    for sl in streamlines:
        tmp = np.copy(sl)
        if axis == "x":
            tmp[:, 0] = tmp[:, 0] * -1
        elif axis == "y":
            tmp[:, 1] = tmp[:, 1] * -1
        elif axis == "z":
            tmp[:, 2] = tmp[:, 2] * -1
        else:
            raise ValueError("Unsupported axis")
        new_sl.append(tmp)
    return new_sl


def transform_point(p, affine):
    """
    Apply affine to point p

    Args:
        p: [x, y, z]
        affine: [4x4] matrix

    Returns:
        [x, y, z]
    """
    M = affine[:3, :3]
    offset = affine[:3, 3]
    return M.dot(p) + offset


def invert_streamlines(streamlines, reference_img, affine, axis="x"):
    """
    Invert streamlines. If inverting image voxel order (img[::-1]) we can do this inversion to the streamlines and
    the result properly fits to the inverted image.

    Args:
        streamlines:
        reference_img: 3d array
        affine: 4x4 matrix
        axis: x | y | z

    Returns:
        streamlines
    """

    img_shape = np.array(reference_img.shape)
    img_center_voxel_space = (img_shape - 1) / 2.
    img_center_mm_space = transform_point(img_center_voxel_space, affine)

    # affine_invert = np.eye(4)
    affine_invert = np.copy(affine)
    affine_invert[0, 3] = 0
    affine_invert[1, 3] = 0
    affine_invert[2, 3] = 0

    affine_invert[0, 0] = 1
    affine_invert[1, 1] = 1
    affine_invert[2, 2] = 1

    if axis == "x":
        affine_invert[0, 0] = -1
        affine_invert[0, 3] = img_center_mm_space[1] * 2
    elif axis == "y":
        affine_invert[1, 1] = -1
        affine_invert[1, 3] = img_center_mm_space[1] * 2
    elif axis == "z":
        affine_invert[2, 2] = -1
        affine_invert[2, 3] = img_center_mm_space[1] * 2
    else:
        raise ValueError("invalid axis")

    return list(transform_streamlines(streamlines, affine_invert))


def resample_to_same_distance(streamlines, max_nr_points=10, ANTI_INTERPOL_MULT=1):
    dist = sl_length(streamlines).max() / max_nr_points
    new_streamlines = []
    for sl in streamlines:
        l = sl_length(sl)
        nr_segments = int(l / dist)
        sl_new = set_number_of_points(sl, nb_points=nr_segments * ANTI_INTERPOL_MULT)
        new_streamlines.append(sl_new)
    return new_streamlines


def pad_sl_with_zeros(streamlines, target_len, pad_point):
    new_streamlines = []
    for sl in streamlines:
        new_sl = list(sl) + [pad_point] * (target_len - len(sl))
        new_streamlines.append(new_sl)
    return new_streamlines


def get_idxs_of_closest_points(streamlines, target_point):
    idxs = []
    for sl in streamlines:
        dists = []
        for idx, p in enumerate(sl):
            dist = abs(np.linalg.norm(p - target_point))
            dists.append(dist)
        idxs.append(np.array(dists).argmin())
    return idxs
