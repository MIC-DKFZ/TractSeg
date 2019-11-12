
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
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import values_from_volume
import dipy.stats.analysis as dsa

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

    streamlines = list(transform_streamlines(streamlines, np.linalg.inv(affine)))

    for i in range(dilate):
        beginnings = binary_dilation(beginnings)
    beginnings = beginnings.astype(np.uint8)
    streamlines = _orient_to_same_start_region(streamlines, beginnings)
    if predicted_peaks is not None:
        # scalar img can also be orig peaks
        best_orig_peaks = fiber_utils.get_best_original_peaks(predicted_peaks, scalar_img, peak_len_thr=0.00001)
        scalar_img = np.linalg.norm(best_orig_peaks, axis=-1)

    algorithm = "distance_map"  # equal_dist | distance_map | cutting_plane | afq


    if algorithm == "equal_dist":
        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)
        ### Aggregation ###
        values_mean = np.array(values).mean(axis=1)
        values_std = np.array(values).std(axis=1)
        return values_mean, values_std


    if algorithm == "distance_map":  # cKDTree

        ### Sampling ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        values = map_coordinates(scalar_img, np.array(streamlines).T, order=1)

        ### Aggregating by cKDTree approach ###
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines)
        centroids = Streamlines(clusters.centroids)
        if len(centroids) > 1:
            print("WARNING: number clusters > 1 ({})".format(len(centroids)))
        _, segment_idxs = cKDTree(centroids.data, 1, copy_data=True).query(streamlines, k=1)  # (2000, 100)

        values_t = np.array(values).T  # (2000, 100)

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


    elif algorithm == "cutting_plane":
        # This will resample all streamline to have equally distant points (resulting in a different number of points
        # in each streamline). Then the "middle" of the tract will be estimated taking the middle element of the
        # centroid (estimated with QuickBundles). Then each streamline the point closest to the "middle" will be
        # calculated and points will be indexed for each streamline starting from the middle. Then averaging across
        # all streamlines will be done by taking the mean for points with same indices.

        ### Sampling ###
        streamlines = fiber_utils.resample_to_same_distance(streamlines, max_nr_points=nr_points)
        # map_coordinates does not allow streamlines with different lengths -> use values_from_volume
        values = np.array(values_from_volume(scalar_img, streamlines, affine=np.eye(4))).T

        ### Aggregating by Cutting Plane approach ###
        # Resample to all fibers having same number of points -> needed for QuickBundles
        streamlines_resamp = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines_resamp)
        centroids = Streamlines(clusters.centroids)

        # index of the middle cluster
        middle_idx = int(nr_points / 2)
        middle_point = centroids[0][middle_idx]
        # For each streamline get idx for the point which is closest to the middle
        segment_idxs = fiber_utils.get_idxs_of_closest_points(streamlines, middle_point)

        # Align along the middle and assign indices
        segment_idxs_eqlen = []
        base_idx = 1000  # use higher index to avoid negative numbers for area below middle
        for idx, sl in enumerate(streamlines):
            sl_middle_pos = segment_idxs[idx]
            before_elems = sl_middle_pos
            after_elems = len(sl) - sl_middle_pos
            # indices for one streamline e.g. [998, 999, 1000, 1001, 1002, 1003]; 1000 is middle
            r = range((base_idx - before_elems), (base_idx + after_elems))
            segment_idxs_eqlen.append(r)
        segment_idxs = segment_idxs_eqlen

        # Calcuate maximum number of indices to not result in more indices than nr_points.
        # (this could be case if one streamline is very off-center and therefore has a lot of points only on one
        # side. In this case the values too far out of this streamline will be cut off).
        max_idx = base_idx + int(nr_points / 2)
        min_idx = base_idx - int(nr_points / 2)

        # Group by segment indices
        results_dict = defaultdict(list)
        for idx, sl in enumerate(values):
            for jdx, seg in enumerate(sl):
                current_idx = segment_idxs[idx][jdx]
                if current_idx >= min_idx and current_idx < max_idx:
                    results_dict[current_idx].append(seg)

        # If values missing fill up with centroid values
        if len(results_dict.keys()) < nr_points:
            print("WARNING: found less than required points. Filling up with centroid values.")
            centroid_sl = [centroids[0]]
            centroid_sl = np.array(centroid_sl).T
            centroid_values = map_coordinates(scalar_img, centroid_sl, order=1)
            for idx, seg_idx in enumerate(range(min_idx, max_idx)):
                if len(results_dict[seg_idx]) == 0:
                    results_dict[seg_idx].append(np.array(centroid_values).T[0, idx])

        # Aggregate by mean
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


    elif algorithm == "afq":
        ### sampling + aggregation ###
        streamlines = fiber_utils.resample_fibers(streamlines, nb_points=nr_points)
        streamlines = Streamlines(streamlines)
        weights = dsa.gaussian_weights(streamlines)
        results_mean = dsa.afq_profile(scalar_img, streamlines, affine=np.eye(4), weights=weights)
        results_std = np.zeros(nr_points)
        return results_mean, results_std
