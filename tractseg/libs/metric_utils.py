
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

from tractseg.data import dataset_specific_utils
from tractseg.libs import peak_utils


def my_f1_score(y_true, y_pred):
    """
    Binary f1. Same results as sklearn f1 binary.
    """
    intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def my_f1_score_macro(y_true, y_pred):
    """
    Macro f1. Same results as sklearn f1 macro.

    Args:
        y_true: (n_samples, n_classes)
        y_pred: (n_samples, n_classes)

    Returns:

    """
    f1s = []
    for i in range(y_true.shape[1]):
        intersect = np.sum(y_true[:, i] * y_pred[:, i])  # works because all multiplied by 0 gets 0
        denominator = np.sum(y_true[:, i]) + np.sum(y_pred[:, i])  # works because all multiplied by 0 gets 0
        f1 = (2 * intersect) / (denominator + 1e-6)
        f1s.append(f1)
    return np.mean(np.array(f1s))


def convert_seg_image_to_one_hot_encoding(image):
    """
    Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (x, y, z, n_classes)
    """
    classes = np.unique(image)
    out_image = np.zeros([len(classes)] + list(image.shape), dtype=image.dtype)
    for i, c in enumerate(classes):
        out_image[i][image == c] = 1

    dims = list(range(len(out_image.shape)))
    dims_reordered = [dims[-1]] + dims[:-1]  # put last element to the front

    return out_image.transpose(dims_reordered)  # put class dimension to the back


def calc_overlap(groundtruth, prediction):
    """
    Expects 2 classes: 0 and 1  (otherwise not working)

    IMPORTANT: Because we can not calc this when no 1 in sample, we do not get 1.0 even if
    we compare groundtruth with groundtruth.

    Identical with sklearn recall with average="binary"

    Args:
        groundtruth: 1D array
        prediction: 1D array

    Returns:
        overlap
    """
    groundtruth = groundtruth.astype(np.int32)
    prediction = prediction.astype(np.int32)
    overlap_mask = np.logical_and(prediction == 1, groundtruth == 1)
    if np.count_nonzero(groundtruth) == 0:
        # print("WARNING: could not calc overlap, because division by 0 -> return 0")
        return 0
    else:
        return np.count_nonzero(overlap_mask) / np.count_nonzero(groundtruth)


def calc_overreach(groundtruth, prediction):
    """
    Expects 2 classes: 0 and 1  (otherwise not working)
    """
    groundtruth = groundtruth.astype(np.int32)
    prediction = prediction.astype(np.int32)
    overreach_mask = np.logical_and(groundtruth == 0, prediction == 1)

    if np.count_nonzero(groundtruth) == 0:
        # print("WARNING: could not calc overreach, because division by 0 -> return 0")
        return 0
    else:
        return np.count_nonzero(overreach_mask) / np.count_nonzero(groundtruth)


def normalize_last_element(metrics, length, type):
    for key, value in metrics.items():
        if key.endswith("_" + type):
            metrics[key][-1] /= float(length)
    return metrics


def normalize_last_element_general(metrics, length):
    for key, value in metrics.items():
        metrics[key][-1] /= float(length)
    return metrics


def add_empty_element(metrics):
    for key, value in metrics.items():
        metrics[key].append(0)
    return metrics


def calculate_metrics(metrics, y, class_probs, loss, f1=None, f1_per_bundle=None, type="train", threshold=0.5):
    """
    Add metrics to metric dict.

    Args:
        metrics: metric dict
        y: ground truth (n_samples, n_classes)
        class_probs: predictions (n_samples, n_classes)
        loss:
        f1:
        f1_per_bundle:
        type:
        threshold:

    Returns:
        updated metric dict
    """
    if f1 is None:
        pred_class = (class_probs >= threshold).astype(np.int16)
        y = (y >= threshold).astype(np.int16)

    metrics["loss_" + type][-1] += loss
    if f1 is None:
        metrics["f1_macro_" + type][-1] += my_f1_score_macro(y, pred_class)
    else:
        metrics["f1_macro_" + type][-1] += f1
        if f1_per_bundle is not None:
            for key in f1_per_bundle:
                if "f1_" + key + "_" + type not in metrics:
                    metrics["f1_" + key + "_" + type] = [0]
                metrics["f1_" + key + "_" + type][-1] += f1_per_bundle[key]

    return metrics


def add_to_metrics(metrics, metr_batch, type, metric_types):
    for key in metric_types:
        metrics[key + "_" + type][-1] += metr_batch[key]
    return metrics


def calculate_metrics_onlyLoss(metrics, loss, type="train"):
    metrics["loss_" + type][-1] += loss
    return metrics


def calculate_metrics_each_bundle(metrics, y, class_probs, bundles, f1=None, threshold=0.5):
    if f1 is None:
        pred_class = (class_probs >= threshold).astype(np.int16)
        y = (y >= threshold).astype(np.int16)

        for idx, bundle in enumerate(bundles):
            metrics[bundle][-1] += f1_score(y[:, idx], pred_class[:, idx], average="binary")
    else:
        for idx, bundle in enumerate(bundles):
            metrics[bundle][-1] += f1[bundle]

    return metrics


def calc_peak_dice_onlySeg(classes, y_pred, y_true):
    """
    Create binary mask of peaks by simple thresholding. Then calculate Dice.
    """
    score_per_bundle = {}
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]      # [x,y,z,3]

        # 0.1 -> keep some outliers, but also some holes already; 0.2 also ok (still looks like e.g. CST)
        #  Resulting dice for 0.1 and 0.2 very similar
        y_pred_binary = np.abs(y_pred_bund).sum(axis=-1) > 0.2
        y_true_binary = np.abs(y_true_bund).sum(axis=-1) > 1e-3

        f1 = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), average="binary")
        score_per_bundle[bundle] = f1

    return score_per_bundle


def calc_peak_dice(classes, y_pred, y_true, max_angle_error=[0.9]):
    score_per_bundle = {}
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]  # (x,y,z,3)

        angles = abs(peak_utils.angle_last_dim(y_pred_bund, y_true_bund))
        angles_binary = angles > max_angle_error[0]

        gt_binary = y_true_bund.sum(axis=-1) > 0

        f1 = f1_score(gt_binary.flatten(), angles_binary.flatten(), average="binary")
        score_per_bundle[bundle] = f1

    return score_per_bundle


def calc_peak_dice_pytorch(classes, y_pred, y_true, max_angle_error=[0.9]):
    """
    Calculate angle between groundtruth and prediction and keep the voxels where
    angle is smaller than MAX_ANGLE_ERROR.

    From groundtruth generate a binary mask by selecting all voxels with len > 0.

    Calculate Dice from these 2 masks.

    -> Penalty on peaks outside of tract or if predicted peak=0
    -> no penalty on very very small with right direction -> bad
    => Peak_dice can be high even if peaks inside of tract almost missing (almost 0)

    Args:
        y_pred:
        y_true:
        max_angle_error: 0.7 ->  angle error of 45° or less; 0.9 ->  angle error of 23° or less
                         Can be list with several values -> calculate for several thresholds

    Returns:

    """
    from tractseg.libs import pytorch_utils

    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)

    #Single threshold
    if len(max_angle_error) == 1:
        score_per_bundle = {}
        bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
        for idx, bundle in enumerate(bundles):
            # if bundle == "CST_right":
            y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x, y, z, 3]

            angles = pytorch_utils.angle_last_dim(y_pred_bund, y_true_bund)
            gt_binary = y_true_bund.sum(dim=-1) > 0
            gt_binary = gt_binary.view(-1)  # [bs*x*y]

            angles_binary = angles > max_angle_error[0]
            angles_binary = angles_binary.view(-1)

            f1 = pytorch_utils.f1_score_binary(gt_binary, angles_binary)
            score_per_bundle[bundle] = f1

        return score_per_bundle

    #multiple thresholds
    else:
        score_per_bundle = {}
        bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
        for idx, bundle in enumerate(bundles):
            y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x, y, z, 3]

            angles = pytorch_utils.angle_last_dim(y_pred_bund, y_true_bund)
            gt_binary = y_true_bund.sum(dim=-1) > 0
            gt_binary = gt_binary.view(-1)  # [bs*x*y]

            score_per_bundle[bundle] = []
            for threshold in max_angle_error:
                angles_binary = angles > threshold
                angles_binary = angles_binary.view(-1)

                f1 = pytorch_utils.f1_score_binary(gt_binary, angles_binary)
                score_per_bundle[bundle].append(f1)

        return score_per_bundle


def calc_peak_length_dice(classes, y_pred, y_true, max_angle_error=[0.9], max_length_error=0.1):
    score_per_bundle = {}
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]  # [x, y, z, 3]

        angles = abs(peak_utils.angle_last_dim(y_pred_bund, y_true_bund))

        lenghts_pred = np.linalg.norm(y_pred_bund, axis=-1)
        lengths_true = np.linalg.norm(y_true_bund, axis=-1)
        lengths_binary = abs(lenghts_pred - lengths_true) < (max_length_error * lengths_true)
        lengths_binary = lengths_binary.flatten()

        gt_binary = y_true_bund.sum(axis=-1) > 0
        gt_binary = gt_binary.flatten()  # [bs*x*y]

        angles_binary = angles > max_angle_error[0]
        angles_binary = angles_binary.flatten()

        combined = lengths_binary * angles_binary

        f1 = my_f1_score(gt_binary, combined)
        score_per_bundle[bundle] = f1
    return score_per_bundle


def calc_peak_length_dice_pytorch(classes, y_pred, y_true, max_angle_error=[0.9], max_length_error=0.1):
    import torch
    from tractseg.libs import pytorch_utils

    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)


    #Single threshold
    score_per_bundle = {}
    bundles = dataset_specific_utils.get_bundle_names(classes)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[..., (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[..., (idx * 3):(idx * 3) + 3].contiguous() # [x, y, z, 3]

        angles = pytorch_utils.angle_last_dim(y_pred_bund, y_true_bund)

        lenghts_pred = torch.norm(y_pred_bund, 2., -1)
        lengths_true = torch.norm(y_true_bund, 2, -1)
        lengths_binary = torch.abs(lenghts_pred-lengths_true) < (max_length_error * lengths_true)
        lengths_binary = lengths_binary.view(-1)

        gt_binary = y_true_bund.sum(dim=-1) > 0
        gt_binary = gt_binary.view(-1)  # [bs*x*y]

        angles_binary = angles > max_angle_error[0]
        angles_binary = angles_binary.view(-1)

        combined = lengths_binary * angles_binary

        f1 = pytorch_utils.f1_score_binary(gt_binary, combined)
        score_per_bundle[bundle] = f1
    return score_per_bundle


def unconfound(y, confound, group_data=False):
    """
    This will remove the influence "confound" has on "y".

    If the data is made up of two groups, the group label (indicating the group) must be the first column of
    'confound'. The group label will be considered when fitting the linear model, but will not be considered when
    calculating the residuals.

    Args:
        y: [samples, targets]
        confound: [samples, confounds]
        group_data: if the data is made up of two groups (e.g. for t-test) or is just
                    one group (e.g. for correlation analysis)
    Returns:
        y_correct: [samples, targets]
    """
    # Demeaning beforehand or using intercept=True has similar effect
    #y = demean(y)
    #confound = demean(confound)

    lr = LinearRegression(fit_intercept=True).fit(confound, y)  # lr.coef_: [targets, confounds]
    if group_data:
        y_predicted_by_confound = lr.coef_[:, 1:] @ confound[:, 1:].T
    else:
        y_predicted_by_confound = lr.coef_ @ confound.T  # [targets, samples]
    y_corrected = y.T - y_predicted_by_confound
    return y_corrected.T  # [samples, targets]
