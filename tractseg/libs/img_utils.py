
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import joblib
from joblib import Parallel, delayed
from os.path import join
from pkg_resources import resource_filename

import psutil
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from dipy.align.imaffine import AffineMap

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import exp_utils
from tractseg.data import dataset_specific_utils


def pad_3d_image(image, pad_size, pad_value=None):
    """
    IMPORTANT: numbers in pad_size should be even numbers; they are always divided by 2 and then rounded to floor !
    -> pad 3 -> would result in padding_left: 1 and padding_right: 1  => 1 gets lost

    Args:
        image: 3D array
        pad_size: must be a np array with 3 entries, one for each dimension of the image
        pad_value: value for padding. Use 0 if None.

    Returns:
        padded array
    """
    image_shape = image.shape
    new_shape = np.array(list(image_shape)) + pad_size
    if pad_value is None:
        pad_value = image[0, 0, 0]
    new_image = np.ones(new_shape) * pad_value
    new_image[int(pad_size[0]/2.) : int(pad_size[0]/2.)+image_shape[0],
              int(pad_size[1]/2.) : int(pad_size[1]/2.)+image_shape[1],
              int(pad_size[2]/2.) : int(pad_size[2]/2.)+image_shape[2]] = image
    return new_image


def pad_4d_image(image, pad_size, pad_value=None):
    """
        IMPORTANT: numbers in pad_size should be even numbers; they are always divided by 2 and then rounded to floor !
        -> pad 3 -> would result in padding_left: 1 and padding_right: 1  => 1 gets lost

        Args:
            image: 4D array
            pad_size: must be a np array with 4 entries, one for each dimension of the image
            pad_value: value for padding. Use 0 if None.

        Returns:
            padded array
    """
    image_shape = image.shape
    new_shape = np.array(list(image_shape)) + pad_size
    if pad_value is None:
        pad_value = image[0, 0, 0, 0]
    new_image = np.ones(new_shape) * pad_value
    new_image[int(pad_size[0]/2.) : int(pad_size[0]/2.) + image_shape[0],
              int(pad_size[1]/2.) : int(pad_size[1]/2.) + image_shape[1],
              int(pad_size[2]/2.) : int(pad_size[2]/2.) + image_shape[2],
              int(pad_size[3]/2.) : int(pad_size[3]/2.) + image_shape[3]] = image
    return new_image


def pad_4d_image_left(image, pad_size, new_shape, pad_value=None):
    """
    This can be used if we want to pad by uneven numbers; you specify padding on the right side of
    each dimension (left is then automatically filled up).

    Args:
        image: 4D array
        pad_size: must be a np array with 4 entries, one for each dimension of the image
        new_shape:
        pad_value: value for padding. Use 0 if None.

    Returns:
        padded array
    """
    image_shape = image.shape
    # new_shape = np.array(list(image_shape)) + pad_size
    new_shape = np.array(new_shape)
    if pad_value is None:
        pad_value = image[0, 0, 0, 0]
    new_image = np.ones(new_shape).astype(image.dtype) * pad_value
    new_image[int(pad_size[0]) : int(pad_size[0]) + image_shape[0],
              int(pad_size[1]) : int(pad_size[1]) + image_shape[1],
              int(pad_size[2]) : int(pad_size[2]) + image_shape[2],
              int(pad_size[3]) : int(pad_size[3]) + image_shape[3]] = image
    return new_image


def remove_small_blobs(img, threshold=1, debug=True):
    """
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.
    """
    # Also considers diagonal elements for determining if a element belongs to a blob
    # mask, number_of_blobs = ndimage.label(img, structure=np.ones((3, 3, 3)))
    mask, number_of_blobs = ndimage.label(img)
    if debug:
        print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    #If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1:
        return img

    # Find largest blob, to make sure we do not remove everything
    #   Largest blob is actually the second largest, because largest is the background
    second_largest_blob_value = np.sort(counts)[-2]
    second_largest_blob_idx = np.where(counts==second_largest_blob_value)[0][0]
    if debug:
        print(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]

    for idx in remove_idx:
        if idx != second_largest_blob_idx:  # make sure to keep at least one blob
            mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        mask_after, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def postprocess_segmentations(data, bundles, blob_thr=50, hole_closing=None):
    """
    Postprocessing of segmentations. Fill holes and remove small blobs.

    hole_closing is deactivated per default because it incorrectly fills up the gyri (e.g. in AF).
    """
    skip_hole_closing = ["CST_right", "CST_left", "MCP"]
    increased_hole_closing = []  # not needed anymore because already done in bundle-specific postprocessing

    data_new = []
    for idx, bundle in enumerate(bundles):
        data_single = data[:,:,:,idx]

        #Fill holes
        if hole_closing is not None and bundle not in skip_hole_closing:
            size = hole_closing  # Working as expected (size 2-3 good value)
            if bundle in increased_hole_closing:
                size *= 2
            data_single = ndimage.binary_closing(data_single,
                                                 structure=np.ones((size, size, size))).astype(data_single.dtype)

        # Remove small blobs
        if blob_thr is not None:
            data_single = remove_small_blobs(data_single, threshold=blob_thr, debug=False)

        data_new.append(data_single)
    data_new = np.array(data_new).transpose(1, 2, 3, 0)
    return data_new


def has_two_big_blobs(img, bundle, debug=True):

    big_cluster_threshold = {
        "CA": 200,
        "FX_left": 100,
        "FX_right": 100
    }

    mask, number_of_blobs = ndimage.label(img)
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    #If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1:
        return False

    counts_sorted = np.sort(counts)[::-1][1:]

    if debug:
        print(counts_sorted)

    nr_big_clusters = len(counts_sorted[counts_sorted > big_cluster_threshold[bundle]])
    return nr_big_clusters >= 2


def bundle_specific_postprocessing(data, bundles):
    """
    For certain bundles checks if bundle contains two big blobs. Then it reduces the threshold for conversion to
    binary and applies hole closing.
    """
    bundles_thresholds = {
        "CA": 0.3,
        "FX_left": 0.4,
        "FX_right": 0.4,
    }

    data_new = []
    for idx, bundle in enumerate(bundles):
        data_single = data[:, :, :, idx]

        if bundle in list(bundles_thresholds.keys()):
            if has_two_big_blobs(data_single > 0.5, bundle, debug=False):
                print("INFO: Using bundle specific postprocessing for {} because bundle incomplete.".format(bundle))
                thr = bundles_thresholds[bundle]
            else:
                thr = 0.5
            data_single = data_single > thr

            size = 6
            data_single = ndimage.binary_closing(data_single,
                                                 structure=np.ones((size, size, size)))  # returns bool
        else:
            data_single = data_single > 0.5

        data_new.append(data_single)

    return np.array(data_new).transpose(1, 2, 3, 0).astype(np.uint8)


def resize_first_three_dims(img, order=0, zoom=0.62, nr_cpus=-1):

    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    return np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back


def resize_first_three_dims_singleCore(img, order=0, zoom=0.62, nr_cpus=-1):
    # Runtime 35ms
    img_sm = []
    for grad in range(img.shape[3]):
        #order: The order of the spline interpolation
        #  order=0 -> nearest interpolation; order=1 -> linear or bilinear interpolation?
        img_sm.append(ndimage.zoom(img[:, :, :, grad], zoom, order=order))
    img_sm = np.array(img_sm)
    return img_sm.transpose(1, 2, 3, 0)  # grads channel was in front -> put to back


def resize_first_three_dims_NUMPY(img, order=0, zoom=0.62):
    """
    Version of resize_first_three_dims using direct numpy indexing for storing results to test
    if it is faster. But it is not.
    Runtime 47ms
    """
    img_sm = None
    for grad in range(img.shape[3]):
        #order: The order of the spline interpolation
        #  order=0 -> nearest interpolation; order=1 -> linear or bilinear interpolation?
        grad_sm = ndimage.zoom(img[:, :, :, grad], zoom, order=order)
        if grad == 0:
            img_sm = np.zeros((grad_sm.shape[0], grad_sm.shape[1], grad_sm.shape[2], img.shape[3]))
        img_sm[:, :, :, grad] = grad_sm
    return img_sm


def create_multilabel_mask(classes, subject, labels_type=np.int16, dataset_folder="HCP", labels_folder="bundle_masks"):
    """
    One-hot encoding of all bundles in one big image
    """
    bundles = dataset_specific_utils.get_bundle_names(classes)

    #Masks sind immer HCP_highRes (später erst downsample)
    mask_ml = np.zeros((145, 174, 145, len(bundles)))
    background = np.ones((145, 174, 145))   # everything that contains no bundle

    # first bundle is background -> already considered by setting np.ones in the beginning
    for idx, bundle in enumerate(bundles[1:]):
        mask = nib.load(join(C.HOME, dataset_folder, subject, labels_folder, bundle + ".nii.gz"))
        mask_data = mask.get_fdata().astype(np.uint8)
        mask_ml[:, :, :, idx + 1] = mask_data
        background[mask_data == 1] = 0  # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml.astype(labels_type)


def save_multilabel_img_as_multiple_files(classes, img, affine, path, name="bundle_segmentations"):
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
        exp_utils.make_dir(join(path, name))
        nib.save(img_seg, join(path, name, bundle + ".nii.gz"))


def save_multilabel_img_as_multiple_files_peaks(flip_output_peaks, classes, img, affine, path, name="TOM"):
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        data = img[:, :, :, (idx*3):(idx*3)+3]

        if flip_output_peaks:
            data[:, :, :, 2] *= -1  # flip z Axis for correct view in MITK
            filename = bundle + "_f.nii.gz"
        else:
            filename = bundle + ".nii.gz"

        img_seg = nib.Nifti1Image(data, affine)
        exp_utils.make_dir(join(path, name))
        nib.save(img_seg, join(path, name, filename))


def save_multilabel_img_as_multiple_files_endings(classes, img, affine, path, name="endings_segmentations"):
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        img_seg = nib.Nifti1Image(img[:,:,:,idx], affine)
        exp_utils.make_dir(join(path, name))
        nib.save(img_seg, join(path, name, bundle + ".nii.gz"))


def simple_brain_mask(data):
    """
    Simple brain mask (for peak image). Does not matter if has holes
    because for cropping anyways only take min and max.

    Args:
        data: peak image (x, y, z, 9)

    Returns:
        brain mask (x, y, z)
    """
    data_max = data.max(axis=3)
    mask = data_max > 0.01
    mask = binary_dilation(mask, iterations=1)
    return mask.astype(np.uint8)


def probs_to_binary_bundle_specific(seg, bundles):
    """
    This is not used anymore at the moment.
    """
    assert len(bundles) == seg.shape[3], "dimensions seg and bundles do not match"

    bundles_thresholds = {
        "CA": 0.3,
        "CST_left": 0.4,
        "CST_right": 0.4,
        "FX_left": 0.4,
        "FX_right": 0.4,
    }

    # Thresholds optimised from Phantom
    # bundles_thresholds = {
    #     "CA": 0.3,
    #     "CC_1": 0.2,
    #     "CC_3": 0.3,
    #     "CC_7": 0.3,
    #     "CG_left": 0.3,
    #     "CG_right": 0.3,
    #     "CST_left": 0.4,
    #     "CST_right": 0.4,
    #     "MLF_left": 0.4,
    #     "MLF_right": 0.4,
    #     "FPT_left": 0.4,
    #     "FPT_right": 0.4,
    #     "FX_left": 0.2,
    #     "FX_right": 0.2,
    #     "ICP_left": 0.2,
    #     "ICP_right": 0.2,
    #     "ILF_left": 0.3,
    #     "ILF_right": 0.3,
    #     "MCP": 0.2,
    #     "POPT_left": 0.4,
    #     "POPT_right": 0.4,
    #     "SCP_left": 0.2,
    #     "SCP_right": 0.2,
    #     "SLF_I_left": 0.3,
    #     "SLF_I_right": 0.3,
    #     "SLF_III_left": 0.4,
    #     "SLF_III_right": 0.4,
    #     "UF_left": 0.2,
    #     "UF_right": 0.2,
    #     "T_PREF_left": 0.4,
    #     "T_PREF_right": 0.4,
    #     "T_PREM_left": 0.2,
    #     "T_PREM_right": 0.2,
    #     "T_PREC_left": 0.4,
    #     "T_PREC_right": 0.4,
    #     "T_POSTC_left": 0.3,
    #     "T_POSTC_right": 0.3,
    #     "ST_FO_left": 0.3,
    #     "ST_FO_right": 0.3,
    #     "ST_PREF_left": 0.4,
    #     "ST_PREF_right": 0.4,
    #     "ST_PREM_left": 0.3,
    #     "ST_PREM_right": 0.3,
    #     "ST_PREC_left": 0.4,
    #     "ST_PREC_right": 0.4,
    #     "ST_POSTC_left": 0.3,
    #     "ST_POSTC_right": 0.3,
    # }

    segs_binary = []
    for idx, bundle in enumerate(bundles):
        if bundle in bundles_thresholds:
            thr = bundles_thresholds[bundle]
        else:
            thr = 0.5
        segs_binary.append(seg[:,:,:,idx] > thr)

    return np.array(segs_binary).transpose(1, 2, 3, 0).astype(np.uint8)


def dilate_binary_mask(file_in, file_out, dilation=2):
    img = nib.load(file_in)
    data = img.get_fdata()

    for i in range(dilation):
        data = binary_dilation(data)

    data = data > 0.5

    img_out = nib.Nifti1Image(data.astype(np.uint8), img.affine)
    nib.save(img_out, file_out)


def peaks2fixel(peaks_file_in, fixel_dir_out):
    """
    Transform TOM peak file to mrtrix fixels format. That can then be transformed to spherical harmonics using
    fixel2sh.

    Args:
        peaks_file_in: (x,y,z,3)   (only 1 peak allowed per voxel)
        fixel_dir_out:

    Returns:
        Void
    """
    exp_utils.make_dir(fixel_dir_out)

    peaks_img = nib.load(peaks_file_in)
    peaks = peaks_img.get_fdata()
    s = peaks.shape

    directions = []
    index = np.zeros(list(s[:3]) + [2])
    amplitudes = []

    idx_ctr = 0
    for x in range(s[0]):
        for y in range(s[1]):
            for z in range(s[2]):
                peak = peaks[x, y, z]
                peak_len = np.linalg.norm(peak)
                if peak_len > 0:
                    peak_normalized = peak / (peak_len + 1e-20)
                    directions.append(peak_normalized)
                    amplitudes.append(peak_len)
                    index[x, y, z] = [1, idx_ctr]
                    idx_ctr += 1

    nib.save(nib.Nifti2Image(np.array(directions), np.eye(4)), join(fixel_dir_out, "directions.nii.gz"))
    nib.save(nib.Nifti2Image(index, peaks_img.affine), join(fixel_dir_out, "index.nii.gz"))
    nib.save(nib.Nifti2Image(np.array(amplitudes), np.eye(4)), join(fixel_dir_out, "amplitudes.nii.gz"))


def enforce_shape(data, target_shape=(91, 109, 91, 9)):
    """
    Cut and pad image to have same shape as target_shape (adapts first 3 dimensions, all further dimensions
    have to be the same in data and target_shape).
    """
    ss = data.shape  # source shape
    ts = target_shape  # target shape

    data_new = np.zeros(ts)

    # cut if too much
    if ss[0] > ts[0]:
        data = data[ss[0] - ts[0]:, :, :]
    if ss[1] > ts[1]:
        data = data[:, ss[1] - ts[1]:, :]
    if ss[2] > ts[2]:
        data = data[:, :, ss[2] - ts[2]:]

    # pad with zero if too small
    data_new[:data.shape[0], :data.shape[1], :data.shape[2]] = data
    return data_new


def apply_rotation_to_peaks(peaks, affine):
    """
    peaks: [x, y, z, 3]      image with one peak
    affine: [4, 4]
    """
    shape = peaks.shape
    peaks = peaks.reshape([-1, shape[3]])  # flatten spatial dim for matrix multiplication

    affine = affine[:3, :3]

    # Get rotation component of affine transformation
    len = np.linalg.norm(affine, axis=0)
    rotation = np.zeros((3,3))
    rotation[:, 0] = affine[:, 0] / len[0]
    rotation[:, 1] = affine[:, 1] / len[1]
    rotation[:, 2] = affine[:, 2] / len[2]

    # Apply rotation to bvecs
    rotated_peaks = peaks @ rotation

    rotated_peaks = rotated_peaks.reshape(shape)  # reshape back to 3d spatial dim
    return rotated_peaks


def change_spacing_4D(img_in, new_spacing=1.25):
    """
    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    data = img_in.get_fdata()
    old_shape = data.shape
    img_spacing = abs(img_in.affine[0, 0])

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)
    new_affine[0, 0] = new_spacing if img_in.affine[0, 0] > 0 else -new_spacing
    new_affine[1, 1] = new_spacing if img_in.affine[1, 1] > 0 else -new_spacing
    new_affine[2, 2] = new_spacing if img_in.affine[2, 2] > 0 else -new_spacing

    new_shape = np.floor(np.array(img_in.get_fdata().shape) * (img_spacing / new_spacing))
    new_shape = new_shape[:3]  # drop last dim

    new_data = []
    for i in range(data.shape[3]):
        affine_map = AffineMap(np.eye(4),
                               new_shape, new_affine,
                               old_shape, img_in.affine
                               )
        # Generally "nearest" a bit better results than "linear" interpolation
        res = affine_map.transform(data[:, :, :, i], interp="nearest")
        new_data.append(res)

    new_data = np.array(new_data).transpose(1, 2, 3, 0)
    img_new = nib.Nifti1Image(new_data, new_affine)

    return img_new


def flip_peaks(data, axis="x"):
    """
    Will flip sign of every third element in the 4th dimension.
    """
    if axis == "x":
        data[:, :, :, list(range(0,data.shape[3],3))] *= -1
    elif axis == "y":
        data[:, :, :, list(range(1,data.shape[3],3))] *= -1
    elif axis == "z":
        data[:, :, :, list(range(2,data.shape[3],3))] *= -1
    return data


def flip_axis(data, flip_axis, flip_peaks=False):
    """
    Will flip array ordering and if data is 4D (=peak image) will flip sign of every
    third element in the 4th dimension.
    """
    # todo important: change
    # is_peak_image = True if len(data.shape) > 3 else False
    # # is_peak_image = False
    # if is_peak_image:
    #     print(f"Flipping peaks: {is_peak_image}")
    if flip_axis == "x":
        data = data[::-1, :, :]
        if flip_peaks:
            data = flip_peaks(data, "x")
    elif flip_axis == "y":
        data = data[:, ::-1, :]
        if flip_peaks:
            data = flip_peaks(data, "y")
    elif flip_axis == "z":
        data = data[:, :, ::-1]
        if flip_peaks:
            data = flip_peaks(data, "z")
    return data


def get_flip_axis_to_match_MNI_space(affine):
    flip_axis = []

    if affine[0, 0] > 0:
        flip_axis.append("x")

    if affine[1, 1] < 0:
        flip_axis.append("y")

    if affine[2, 2] < 0:
        flip_axis.append("z")

    return flip_axis


def flip_axis_to_match_MNI_space(data, affine, flip_peaks=False):
    """
    Checks if affine of the image has the same signs on the diagonal as MNI space. If this is not the case it will
    invert the sign of the affine (not returned here) and invert the axis accordingly.
    Optionally can also flip the sign of the peaks in a peak image. But this is actually never needed, because
    the peaks themselves are in world space.
    """
    flip_axis_list = get_flip_axis_to_match_MNI_space(affine)

    for axis in flip_axis_list:
        data = flip_axis(data, axis, flip_peaks=flip_peaks)

    return data, flip_axis_list


def flip_affine(affine, flip_axis_list, data_shape):
    """
    apply flipping to affine
    """
    affine_flipped = affine.copy()  # could be returned if needed

    if "x" in flip_axis_list:
        affine_flipped[0, 0] = affine_flipped[0, 0] * -1
        affine_flipped[0, 3] -= (data_shape[0] - 1)  # this is needed to make it still align with unaltered fibers correctly

    if "y" in flip_axis_list:
        affine_flipped[1, 1] = affine_flipped[1, 1] * -1
        affine_flipped[1, 3] -= (data_shape[1] - 1)

    if "z" in flip_axis_list:
        affine_flipped[2, 2] = affine_flipped[2, 2] * -1
        affine_flipped[2, 3] -= (data_shape[2] - 1)

    return affine_flipped


def flip_peaks_to_correct_orientation_if_needed(peaks_input, do_flip=False):
    """
    We use a pretrained random forest classifier to detect if the orientation of the peak is the same
    orientation as the peaks used for training TractSeg. Otherwise detect along which axis they
    have to be flipped to have the right orientation and return the flipped peaks.

    NOTES:
    - Accuracy around 98%, but often failed on tumor cases
      -> if not really reliable is no benefit for UX
    - This random forest was trained on subjects with incorrectly rotated bvecs. Therefore would have to retrain it
      on subjects with correct bvecs before we can use it again
    - After fixing of bvecs rotation bug in general peaks are incorrectly flipped a lot less often. Therefore not so
      important anymore
    => Do not use anymore

    Args:
        peaks_input: nifti peak img
        do_flip: also return flipped data or only return if flip needed

    Returns:
        (4D numpy array (flipped peaks), flip orientation)
    """
    if sys.version_info[0] < 3:
        print("INFO: Peak orientation check not working on python 2, therefore it is skipped.")
        return peaks_input.get_fdata(), None
    else:
        peaks = change_spacing_4D(peaks_input, new_spacing=2.).get_fdata()
        #shape the classifier has been trained with
        peaks = enforce_shape(peaks, target_shape=(91, 109, 91, 9))

        peaks_x = peaks[int(peaks.shape[0] / 2.), :, :, :]
        peaks_y = peaks[:, int(peaks.shape[1] / 2.), :, :]
        peaks_z = peaks[:, :, int(peaks.shape[2] / 2.), :]
        X = [list(peaks_x.flatten()) + list(peaks_y.flatten()) + list(peaks_z.flatten())]
        X = np.nan_to_num(X)

        random_forest_path = resource_filename('tractseg.resources', 'random_forest_peak_orientation_detection.pkl')
        clf = joblib.load(random_forest_path)
        predicted_label = clf.predict(X)[0]
        # labels:
        #  ok: 0, x:1, y:2, z:3
        peaks_input_data = peaks_input.get_fdata()
        if do_flip:
            if predicted_label == 0:
                return peaks_input_data, None
            elif predicted_label == 1:
                return flip_peaks(peaks_input_data, axis="x"), "x"
            elif predicted_label == 2:
                return flip_peaks(peaks_input_data, axis="y"), "y"
            elif predicted_label == 3:
                return flip_peaks(peaks_input_data, axis="z"), "z"
        else:
            if predicted_label == 0:
                return peaks_input_data, None
            elif predicted_label == 1:
                return peaks_input_data, "x"
            elif predicted_label == 2:
                return peaks_input_data, "y"
            elif predicted_label == 3:
                return peaks_input_data, "z"


def get_image_spacing(img_path):
    img = nib.load(img_path)
    affine = img.affine
    return str(abs(round(affine[0, 0], 2)))


def scale_to_range(data, range=(0, 1)):
    return (range[1] - range[0]) * (data - data.min()) / (data.max() - data.min()) + range[0]
