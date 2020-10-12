"""
For the tractogram of a bundle creates two masks: One for startpoints and one for endpoints.

Approach:
Use DBscan clustering for dividing all endpoints into start and ending. Uses only a subset of all points
because of runtime.
Then train random forest on those 2 clusters and use random forest to divide all points (not only subset) into
two classes.

Arguments:
    reference_image_path
    tractogram_path
    output_mask_path   (without file ending)

Example:
    python create_endpoints_mask_with_clustering.py brain_mask.nii.gz CST_right.trk CST_right_mask
"""

import sys

import nibabel as nib
import numpy as np
from nibabel import trackvis

from dipy.tracking.streamline import transform_streamlines
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as sk_shuffle


def select_two_biggest_clusters(labels, points):
    hist = np.histogram(labels, len(np.unique(labels)))[0]  # histogram of cluster sizes
    label_idx = np.argsort(hist)[-2:]  # get index of the two biggest elements
    biggest_labels = np.unique(labels)[label_idx]

    print("Labels:")
    print(np.unique(labels))
    print("Histogram of cluster labels:")
    print(hist)
    print("Labels of the two biggest clusters: {}".format(biggest_labels))

    cluster_A = points[labels == biggest_labels[0]]
    cluster_B = points[labels == biggest_labels[1]]
    return cluster_A, cluster_B


def cluster(points, algorithm=DBSCAN):
    print("Running {}...".format(algorithm))
    if algorithm == "KMeans":
        # not good at finding clusters if close together
        labels = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit_predict(points)
    elif algorithm == "DBSCAN":
        # no fixed number of labels; slow with high eps
        labels = DBSCAN(eps=3.0, n_jobs=-1).fit_predict(points)
    # labels = SpectralClustering(n_clusters=2, n_jobs=-1).fit_predict(points)  # slow (> 1min)
    # labels = AgglomerativeClustering(n_clusters=2).fit_predict(points)  # fast
    points_start, points_end = select_two_biggest_clusters(labels, points)
    return points_start, points_end


def percental_difference_between_two_numbers(a, b):
    """
    How much percent is smaller number smaller than bigger number.
    If 1    -> both number equal size
    If 0.5  -> smaller number is half the size of bigger number
    """
    if a <= b:
        return float(a) / b
    else:
        return float(b) / a


args = sys.argv[1:]
ref_img_in = args[0]
file_in = args[1]
file_out = args[2]

ref_img = nib.load(ref_img_in)
ref_img_shape = ref_img.get_fdata().shape

streams, hdr = trackvis.read(file_in)
streamlines = [s[0] for s in streams]
streamlines = transform_streamlines(streamlines, np.linalg.inv(ref_img.affine))

mask_start = np.zeros(ref_img_shape)
mask_end = np.zeros(ref_img_shape)

if len(streamlines) > 0:

    startpoints = []
    endpoints = []
    for streamline in streamlines:
        startpoints.append(streamline[0])
        endpoints.append(streamline[-1])

    points = np.array(startpoints + endpoints)
    # Subsample points to make clustering faster
    #  Has to be at least 50k to work properly for very big tracts like CC (otherwise points too far apart for DBSCAN)
    NR_POINTS_FOR_CLUSTERING = 50000
    if points.shape[0] > NR_POINTS_FOR_CLUSTERING:
        idxs = np.random.choice(points.shape[0], NR_POINTS_FOR_CLUSTERING, False, None)
        points_subset = np.array(points[idxs])
    else:
        points_subset = points

    #Clustering
    points_start, points_end = cluster(points_subset, algorithm="DBSCAN")
    difference = percental_difference_between_two_numbers(len(points_start), len(points_end))
    if difference < 0.5:
        print("\nWARNING: DBSCAN did not find equally sized cluster (smaller cluster is only {}% of bigger one) -> "
              "rerun with KMeans\n".format(round(difference, 4)))
        points_start, points_end = cluster(points_subset, algorithm="KMeans")

    print("Training Random Forest and Predicting...")
    X = np.concatenate((points_start, points_end), axis=0)
    y = np.concatenate((np.zeros(points_start.shape[0]), np.ones(points_end.shape[0])))
    X, y = sk_shuffle(X, y, random_state=9)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X, y)
    labels = clf.predict(points)

    print("Creating Binary Masks...")
    for idx, point in enumerate(points):
        if labels[idx] == 0:
            mask_start[int(point[0]), int(point[1]), int(point[2])] = 1
        else:
            mask_end[int(point[0]), int(point[1]), int(point[2])] = 1

    mask_start = mask_start > 0.5
    mask_end = mask_end > 0.5

else:
    print("Bundle contains 0 streamlines -> returning empty image")

dm_binary_img = nib.Nifti1Image(mask_start.astype("uint8"), ref_img.affine)
nib.save(dm_binary_img, file_out + "_beginnings.nii.gz")
dm_binary_img = nib.Nifti1Image(mask_end.astype("uint8"), ref_img.affine)
nib.save(dm_binary_img, file_out + "_endings.nii.gz")
