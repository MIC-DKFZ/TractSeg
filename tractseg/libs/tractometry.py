
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.interpolation import map_coordinates
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils


def _get_length_best_orig_peak(predicted_img, orig_img, x, y, z):
    predicted = predicted_img[x, y, z, :]       # 1 peak
    orig = [orig_img[x, y, z, 0:3], orig_img[x, y, z, 3:6], orig_img[x, y, z, 6:9]]     # 3 peaks

    angle1 = abs(np.dot(predicted, orig[0]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[0]) + 1e-7))
    angle2 = abs(np.dot(predicted, orig[1]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[1]) + 1e-7))
    angle3 = abs(np.dot(predicted, orig[2]) / (np.linalg.norm(predicted) * np.linalg.norm(orig[2]) + 1e-7))

    argmax = np.argmax([angle1, angle2, angle3])
    best_peak_len = np.linalg.norm(orig[argmax])
    return best_peak_len


def _orient_to_same_start_region(streamlines, beginnings):
    # (we could also use dipy.tracking.streamline.orient_by_streamline instead)
    streamlines = fiber_utils.add_to_each_streamline(streamlines, 0.5)
    streamlines_new = []
    for idx, sl in enumerate(streamlines):
        startpoint = sl[0]
        # Flip streamline if not in right order
        if beginnings[int(startpoint[0]), int(startpoint[1]), int(startpoint[2])] == 0:
            sl = sl[::-1, :]
        streamlines_new.append(sl)
    streamlines_new = fiber_utils.add_to_each_streamline(streamlines_new, -0.5)
    return streamlines_new


def evaluate_along_streamlines(scalar_img, streamlines, beginnings, nr_points, dilate=0, predicted_peaks=None,
                               affine=None):
    # Runtime:
    # - default:                2.7s (test),    56s (all),      10s (test 4 bundles, 100 points)
    # - map_coordinate order 1: 1.9s (test),    26s (all),       6s (test 4 bundles, 100 points)
    # - map_coordinate order 3: 2.2s (test),    33s (all),
    # - values_from_volume:     2.5s (test),    43s (all),
    # - AFQ:                      ?s (test),     ?s (all),      85s  (test 4 bundles, 100 points)
    # => AFQ a lot slower than others

    for i in range(dilate):
        beginnings = binary_dilation(beginnings)
    beginnings = beginnings.astype(np.uint8)
    streamlines = _orient_to_same_start_region(streamlines, beginnings)
    if predicted_peaks is not None:
        # scalar img can also be orig peaks
        best_orig_peaks = fiber_utils.get_best_original_peaks(predicted_peaks, scalar_img, peak_len_thr=0.00001)
        scalar_img = np.linalg.norm(best_orig_peaks, axis=-1)


    ### Sampling ###

    #################################### Sampling map_coordinates #####################
    values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)
    ###################################################################################

    #################################### Sampling values_from_volume ##################
    # streamlines = list(transform_streamlines(streamlines, affine))  # this has to be here; not remove previous one
    # values = np.array(values_from_volume(scalar_img, streamlines, affine=affine)).T
    ###################################################################################


    ### Aggregation ###

    #################################### Aggregating by MEAN ##########################
    # values_mean = np.array(values).mean(axis=1)
    # values_std = np.array(values).std(axis=1)
    # return values_mean, values_std
    ###################################################################################

    #################################### Aggregating by cKDTree #######################
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=100., metric=metric)
    clusters = qb.cluster(streamlines)
    centroids = Streamlines(clusters.centroids)
    if len(centroids) > 1:
        print("WARNING: number clusters > 1 ({})".format(len(centroids)))
    _, segment_idxs = cKDTree(centroids.data, 1, copy_data=True).query(streamlines, k=1)  # (2000, 20)

    values_t = np.array(values).T  # (2000, 20)

    # If we want to take weighted mean like in AFQ:
    # weights = dsa.gaussian_weights(Streamlines(streamlines))
    # values_t = weights * values_t
    # return np.sum(values_t, 0), None

    results_dict = defaultdict(list)
    for idx, sl in enumerate(values_t):
        for jdx, seg in enumerate(sl):
            results_dict[segment_idxs[idx, jdx]].append(seg)

    if len(results_dict.keys()) < nr_points:
        print("WARNING: found less than required points. Filling up with centroid values.")
        centroid_values = map_coordinates(scalar_img, np.array([centroids[0]]).T, order=1)
        for i in range(nr_points):
            if len(results_dict[i]) == 0:
                results_dict[i].append(np.array(centroid_values).T[0, i])

    results_mean = []
    results_std = []
    for key in sorted(results_dict.keys()):
        value = results_dict[key]
        if len(value) > 0:
            results_mean.append(np.array(value).mean())
            results_std.append(np.array(value).std())
        else:
            print("WARNING: empty segment")
            results_mean.append(0)
            results_std.append(0)

    return results_mean, results_std
    ###################################################################################

    #################################### AFQ (sampling + aggregation ##################
    # streamlines = list(transform_streamlines(streamlines, affine))  # this has to be here; not remove previous one
    # streamlines = Streamlines(streamlines)
    # weights = dsa.gaussian_weights(streamlines)
    # results_mean = dsa.afq_profile(scalar_img, streamlines, affine=affine, weights=weights)
    # results_std = None
    # return results_mean, results_std
    ###################################################################################
