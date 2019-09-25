
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh
from batchgenerators.augmentations.utils import elastic_deform_coordinates
from batchgenerators.augmentations.utils import rotate_coords_3d
from batchgenerators.augmentations.utils import rotate_coords_2d
from batchgenerators.augmentations.utils import scale_coords
from batchgenerators.augmentations.utils import create_matrix_rotation_x_3d
from batchgenerators.augmentations.utils import create_matrix_rotation_y_3d
from batchgenerators.augmentations.utils import create_matrix_rotation_z_3d
from batchgenerators.augmentations.utils import interpolate_img
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

from tractseg.libs import peak_utils


def rotate_multiple_peaks(data, angle_x, angle_y, angle_z):
    """
    Rotates the peaks by the given angles.

    data: 2D or 3D 3-peak image (9, x, y, [z])
    """
    def rotate_peaks(peaks, angle_x, angle_y, angle_z):
        rot_matrix = np.identity(3)
        rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
        rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
        # rotate clockwise -> wrong
        # peaks_rot = np.dot(peaks.reshape(3, -1).transpose(), rot_matrix).transpose().reshape(peaks.shape)
        # rotate counterclockwise -> this is correct
        peaks_rot = np.dot(rot_matrix, peaks.reshape(3, -1)).reshape(peaks.shape)
        return peaks_rot

    peaks_rot = np.zeros(data.shape)
    for i in range(3):
        peaks_rot[i*3:(i+1)*3, ...] = rotate_peaks(data[i*3:(i+1)*3, ...], angle_x, angle_y, angle_z)

    return peaks_rot


def rotate_multiple_tensors(data, angle_x, angle_y, angle_z):
    """
    Rotates the tensors by the given angles.

    data: 2D or 3D 3-tensor image (18, x, y, [z])
    """
    data = np.moveaxis(data, 0, -1)  # move channels to back

    def rotate_tensors(peaks, angle_x, angle_y, angle_z):
        rot_matrix = np.identity(3)
        rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
        rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)

        peaks = peak_utils.flat_tensor_to_matrix_tensor(peaks)  # (x, y, 3, 3)
        # rotate clockwise -> wrong
        # peaks_rot = rot_matrix.T @ peaks @ rot_matrix  # (x, y, 3, 3)
        # rotate counterclockwise -> this is correct
        peaks_rot = rot_matrix @ peaks @ rot_matrix.T  # (x, y, 3, 3)
        peaks_rot = peak_utils.matrix_tensor_to_flat_tensor(peaks_rot)
        return peaks_rot

    peaks_rot = np.zeros(data.shape)
    for i in range(3):
        peaks_rot[..., i*6:(i+1)*6] = rotate_tensors(data[..., i*6:(i+1)*6], angle_x, angle_y, angle_z)

    peaks_rot = np.moveaxis(peaks_rot, -1, 0)  # move channels to front
    return peaks_rot


def augment_spatial_peaks(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, slice_dir=None):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if np.random.uniform() < p_el_per_sample and do_elastic_deform:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        # NEW: initialize all because all needed for rotate_multiple_peaks (even if only rotating along one axis)
        a_x = 0
        a_y = 0
        a_z = 0

        if np.random.uniform() < p_rot_per_sample and do_rotation:
            if angle_x[0] == angle_x[1]:
                a_x = angle_x[0]
            else:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            if dim == 3:
                if angle_y[0] == angle_y[1]:
                    a_y = angle_y[0]
                else:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                if angle_z[0] == angle_z[1]:
                    a_z = angle_z[0]
                else:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if np.random.uniform() < p_scale_per_sample and do_scale:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]

        # NEW: Rotate Peaks / Tensors
        if dim > 2:
            raise ValueError("augment_spatial_peaks only supports 2D at the moment")

        sampled_2D_angle = a_x  # if 2D angle will always be a_x even if rotating other axis
        a_x = 0
        a_y = 0
        a_z = 0

        if slice_dir == 0:
            a_x = sampled_2D_angle
        elif slice_dir == 1:
            a_y = sampled_2D_angle
        elif slice_dir == 2:
            # Somehow we have to invert rotation direction for z to make align properly with rotated voxels.
            #  Unclear why this is the case. Maybe some different conventions for peaks and voxels??
            a_z = sampled_2D_angle * -1
        else:
            raise ValueError("invalid slice_dir passed as argument")

        data_aug = data_result[sample_id]
        if data_aug.shape[0] == 9:
            data_result[sample_id] = rotate_multiple_peaks(data_aug, a_x, a_y, a_z)
        elif data_aug.shape[0] == 18:
            data_result[sample_id] = rotate_multiple_tensors(data_aug, a_x, a_y, a_z)
        else:
            raise ValueError("Incorrect number of channels (expected 9 or 18)")

    return data_result, seg_result


# This is identical to batchgenerators.transforms.spatial_transforms.SpatialTransform except for another
# augment_spatial function, which also rotates the peaks when doing rotation.
class SpatialTransformPeaks(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True,
                 data_key="data", label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        #NEW: pass slice direction, because peak rotation depends on it
        slice_dir = data_dict.get("slice_dir")

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial_peaks(data, seg, patch_size=patch_size,
                                      patch_center_dist_from_border=self.patch_center_dist_from_border,
                                      do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                      do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                      angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                      border_mode_data=self.border_mode_data,
                                      border_cval_data=self.border_cval_data, order_data=self.order_data,
                                      border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                      order_seg=self.order_seg, random_crop=self.random_crop,
                                      p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                      p_rot_per_sample=self.p_rot_per_sample, slice_dir=slice_dir)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict

