
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.libs import peak_utils


def get_seg_single_img_3_directions(Config, model, subject=None, data=None, scale_to_world_shape=True,
                                    only_prediction=False, batch_size=1):
    from tractseg.libs import trainer

    prob_slices = []
    directions = ["x", "y", "z"]
    for idx, direction in enumerate(directions):
        Config.SLICE_DIRECTION = direction
        print("Processing direction ({} of 3)".format(idx+1))

        if subject:
            dataManagerSingle = DataLoaderInference(Config, subject=subject)  # runtime on HCP data 0s
        else:
            dataManagerSingle = DataLoaderInference(Config, data=data)  # runtime on HCP data 0s

        img_probs, img_y = trainer.predict_img(Config, model, dataManagerSingle, probs=True,
                                               scale_to_world_shape=scale_to_world_shape,
                                               only_prediction=only_prediction,
                                               batch_size=batch_size)    # (x, y, z, nr_classes)
        prob_slices.append(img_probs)

    probs_x, probs_y, probs_z = prob_slices
    # runtime on HCP data: 0s
    probs_x = probs_x[..., None]  # (x, y, z, nr_classes, 1)
    probs_y = probs_y[..., None]
    probs_z = probs_z[..., None]
    # runtime on HCP data: 1.4s
    probs_combined = np.concatenate((probs_x, probs_y, probs_z), axis=4)    # (x, y, z, nr_classes, 3)
    return probs_combined, img_y


def mean_fusion(threshold, img, probs=True):
    """
    Merge along last axis by mean.

    Args:
        threshold: Threshold for binarization of probabilities
        img: 5D Image with probability per direction (x, y, z, nr_classes, 3)
        probs: Return binary image or probabilities

    Returns:
        4D image (x, y, z, nr_classes)
    """
    probs_mean = img.mean(axis=4)
    if not probs:
        probs_mean[probs_mean >= threshold] = 1
        probs_mean[probs_mean < threshold] = 0
        probs_mean = probs_mean.astype(np.int16)
    return probs_mean


def mean_fusion_peaks(img, nr_cpus=-1):
    """
    Calculating mean in tensor space (if simply taking mean in peak space most voxels look fine but a few are
    completely wrong (e.g. some voxels in transition to lateral projections of CST)).

    Args:
        img: 5D Image with probability per direction (x, y, z, nr_classes, 3)

    Returns:
        4D image (x, y, z, nr_classes)
    """
    from joblib import Parallel, delayed

    nr_classes = int(img.shape[3] / 3)

    def process_bundle(idx):
        print("idx: {}".format(idx))
        dirs_per_bundle = []
        for jdx in range(3):  # 3 orientations
            peak = img[:, :, :, idx*3:idx*3+3, jdx]
            tensor = peak_utils.peaks_to_tensors(peak)
            dirs_per_bundle.append(tensor)
        merged_tensor = np.array(dirs_per_bundle).mean(axis=0)
        merged_peak = peak_utils.tensors_to_peaks(merged_tensor)
        return merged_peak

    n_jobs = 5 if nr_cpus == -1 else min(nr_cpus, 5)
    merged_peaks_all = Parallel(n_jobs=n_jobs)(delayed(process_bundle)(idx) for idx in range(nr_classes))

    merged_peaks_all = np.array(merged_peaks_all).transpose(1, 2, 3, 0, 4)
    s = merged_peaks_all.shape
    merged_peaks_all = merged_peaks_all.reshape([s[0], s[1], s[2], s[3] * s[4]])

    return merged_peaks_all


def majority_fusion(threshold, img, probs=None):
    """
    Use majority voting instead of mean.
    Mean slightly better results (+0.002 Dice)
    -> use Mean
    """
    img[img >= threshold] = 1
    img[img < threshold] = 0
    probs_combined = img.astype(np.int16)
    probs_sum = probs_combined.sum(axis=4)
    probs_result = np.zeros(probs_sum.shape)
    probs_result[probs_sum >= 2] = 1   #majority is at least 2 of 3
    probs_result[probs_sum < 2] = 0
    return probs_result.astype(np.int16)
