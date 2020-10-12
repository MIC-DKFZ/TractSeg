"""
Convert a trk streamline file to a binary map.

Arguments:
    trk_file_in
    mask_file_out
    reference_file (file with same affine as DWI file, e.g. brain mask)

Example:
    python trk_2_binary.py CST_right.trk CST_right.nii.gz nodif_brain_mask.nii.gz
"""

import os, sys, inspect
from dipy.tracking import utils as utils_trk
from nibabel import trackvis
import nibabel as nib
import numpy as np
from scipy import ndimage
import logging
from dipy.tracking.vox2track import streamline_mapping

logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
logging.getLogger().setLevel(logging.INFO)


def get_number_of_points(streamlines):
    count = 0
    for sl in streamlines:
        count += len(sl)
    return count


def remove_small_blobs(img, threshold=1):
    """
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.
    """
    # if using structure=np.ones((3, 3, 3): Also considers diagonal elements for determining if a element
    # belongs to a blob -> not good, because leaves hardly any small blobs we can remove
    mask, number_of_blobs = ndimage.label(img)
    logging.debug('Number of blobs before filtering: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob
    logging.debug(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]

    for idx in remove_idx:
        mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    mask_after, number_of_blobs_after = ndimage.label(mask)
    logging.debug('Number of blobs after filtering: ' + str(number_of_blobs_after))
    return mask


if __name__ == '__main__':
    args = sys.argv[1:]
    file_in = args[0]
    file_out = args[1]
    ref_img_path = args[2]

    HOLE_CLOSING = 0

    # choose from "trk" or "trk_legacy"
    #  Use "trk_legacy" for zenodo dataset v1.1.0 and below
    #  Use "trk" for zenodo dataset v1.2.0
    tracking_format = "trk"

    ref_img = nib.load(ref_img_path)
    ref_affine = ref_img.affine
    ref_shape = ref_img.get_fdata().shape

    streams, hdr = trackvis.read(file_in)
    streamlines = [s[0] for s in streams]  # list of 2d ndarrays

    if tracking_format == "trk_legacy":
        streams, hdr = trackvis.read(file_in)
        streamlines = [s[0] for s in streams]
    else:
        sl_file = nib.streamlines.load(file_in)
        streamlines = sl_file.streamlines

    #Upsample Streamlines (very important, especially when using DensityMap Threshold. Without upsampling eroded results)
    max_seq_len = abs(ref_affine[0, 0] / 4)
    streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))

    # Remember: Does not count if a fibers has no node inside of a voxel -> upsampling helps, but not perfect
    # Counts the number of unique streamlines that pass through each voxel -> oversampling does not distort result
    dm = utils_trk.density_map(streamlines, ref_shape, affine=ref_affine)

    # Create Binary Map
    dm_binary = dm > 0  # Using higher Threshold problematic, because tends to remove valid parts (sparse fibers)
    dm_binary_c = dm_binary

    #Filter Blobs (might remove valid parts) -> do not use
    #dm_binary_c = remove_small_blobs(dm_binary_c, threshold=10)

    #Closing of Holes (not ideal because tends to remove valid holes, e.g. in MCP) -> do not use
    # size = 1
    # dm_binary_c = ndimage.binary_closing(dm_binary_c, structure=np.ones((size, size, size))).astype(dm_binary.dtype)

    #Save Binary Mask
    dm_binary_img = nib.Nifti1Image(dm_binary_c.astype("uint8"), ref_affine)
    nib.save(dm_binary_img, file_out)
