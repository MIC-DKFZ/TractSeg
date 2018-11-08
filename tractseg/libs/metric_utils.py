#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import f1_score
# from medpy import metric

from tractseg.libs import exp_utils


def my_f1_score(y_true, y_pred):
    '''
    Binary f1

    Tested: same results as sklearn f1 binary
    '''
    intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def my_f1_score_macro(y_true, y_pred):
    '''
    Macro f1

    y_true: [n_samples, n_classes]
    y_pred: [n_samples, n_classes]

    Tested: same results as sklearn f1 macro
    '''
    f1s = []
    for i in range(y_true.shape[1]):
        intersect = np.sum(y_true[:, i] * y_pred[:, i])  # works because all multiplied by 0 gets 0
        denominator = np.sum(y_true[:, i]) + np.sum(y_pred[:, i])  # works because all multiplied by 0 gets 0
        f1 = (2 * intersect) / (denominator + 1e-6)
        f1s.append(f1)
    return np.mean(np.array(f1s))


def convert_seg_image_to_one_hot_encoding(image):
    '''
    Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (x, y, z, n_classes)
    '''
    classes = np.unique(image)
    out_image = np.zeros([len(classes)] + list(image.shape), dtype=image.dtype)
    for i, c in enumerate(classes):
        out_image[i][image == c] = 1

    dims = list(range(len(out_image.shape)))
    dims_reordered = [dims[-1]] + dims[:-1]  # put last element to the front

    return out_image.transpose(dims_reordered)  # put class dimension to the back


def calc_overlap(groundtruth, prediction):
    '''
    Expects 2 classes: 0 and 1  (otherwise not working)

    IMPORTANT: Because we can not calc this, when no 1 in sample, we do not get 1.0 even if
    we compare groundtruth with groundtruth (when I tried that, I got 0.89)

    Identical with recall with average="binary"

    :param groundtruth: 1D array
    :param prediction: 1D array
    :return:
    '''
    # ensure int type
    groundtruth = groundtruth.astype(np.int32)
    prediction = prediction.astype(np.int32)
    overlap_mask = np.logical_and(prediction == 1, groundtruth == 1)
    if np.count_nonzero(groundtruth) == 0:
        # print("WARNING: could not calc overlap, because division by 0 -> return 0")
        return 0  # ok, because we sum these up -> do not change sum  -> not quite right
    else:
        return np.count_nonzero(overlap_mask) / np.count_nonzero(groundtruth)


def calc_overreach(groundtruth, prediction):
    '''
    Expects 2 classes: 0 and 1  (otherwise not working)

    :param groundtruth: 1D array
    :param prediction: 1D array
    :return:
    '''
    # ensure int type
    groundtruth = groundtruth.astype(np.int32)
    prediction = prediction.astype(np.int32)
    overreach_mask = np.logical_and(groundtruth == 0, prediction == 1)

    if np.count_nonzero(groundtruth) == 0:
        # print("WARNING: could not calc overreach, because division by 0 -> return 0")
        return 0   # ok, because we sum these up -> do not change sum  -> not quite right
    else:
        # return np.count_nonzero(overreach_mask) / np.count_nonzero(prediction)  # FALSCH!!
        return np.count_nonzero(overreach_mask) / np.count_nonzero(groundtruth)


def normalize_last_element(metrics, length, type):
    '''

    :param metrics:
    :param length:
    :param type:  "train" or "test"
    :return:
    '''
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
    '''
    y -> Ground Truth

    y: [n_samples, n_classes]
    class_probs: [n_samples, n_classes]

    class_probs -> Predictions
    '''

    if f1 is None:
        class_probs[class_probs >= threshold] = 1                     # bit slow
        class_probs[class_probs < threshold] = 0                      # bit slow
        pred_class = class_probs.astype(np.int16)     #is float32     #slow

        y[y >= threshold] = 1                         # bit slow
        y[y < threshold] = 0                          # bit slow
        y = y.astype(np.int16)    #is int16           #slow

    metrics["loss_"+type][-1] += loss
    if f1 is None:
        metrics["f1_macro_"+type][-1] += my_f1_score_macro(y, pred_class)
    else:
        metrics["f1_macro_"+type][-1] += f1
        if f1_per_bundle is not None:
            for key in f1_per_bundle:
                if "f1_"+key+"_"+type not in metrics:
                    metrics["f1_" + key + "_" + type] = [0]
                metrics["f1_" + key + "_" + type][-1] += f1_per_bundle[key]

    return metrics


def calculate_metrics_onlyLoss(metrics, loss, type="train"):
    metrics["loss_"+type][-1] += loss
    return metrics


def calculate_metrics_each_bundle(metrics, y, class_probs, bundles, f1=None, threshold=0.5):
    '''
    bundles -> have to be in same order as classes in predictions
    y -> Ground Truth
    class_probs -> Predictions
    '''

    if f1 is None:
        class_probs[class_probs >= threshold] = 1
        class_probs[class_probs < threshold] = 0
        pred_class = class_probs.astype(np.int16)

        y[y >= threshold] = 1
        y[y < threshold] = 0
        y = y.astype(np.int16)

        for idx, bundle in enumerate(bundles):
            metrics[bundle][-1] += f1_score(y[:,idx], pred_class[:,idx], average="binary")
    else:
        for idx, bundle in enumerate(bundles):
            metrics[bundle][-1] += f1[bundle]

    return metrics


def average_metric_all_bundles(metrics_all):
    '''
    For each experiment: Takes last element of each metric
        -> then: take average
    => Average of all metrics for all experiments
    :param metrics_all: list of metrics dictionaries
    :return:
    '''

    metrics_avg = {}

    for metric_key in metrics_all[0]:

        elems = []
        for experiment in metrics_all:
            elems.append(experiment[metric_key][-1])

        metrics_avg[metric_key] = sum(elems) / len(elems)

    return metrics_avg


def calc_peak_dice_onlySeg(Config, y_pred, y_true):
    '''
    Create binary mask of peaks by simple thresholding. Then calculate Dice.

    :param y_pred:
    :param y_true:
    :return:
    '''

    score_per_bundle = {}
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
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


def calc_peak_dice(Config, y_pred, y_true, max_angle_error=[0.9]):
    '''

    :param y_pred:
    :param y_true:
    :param max_angle_error:  0.7 ->  angle error of 45° or less; 0.9 ->  angle error of 23° or less
    :return:
    '''

    def angle(a, b):
        '''
        Calculate the angle between two 1d-arrays (2 vectors) along the last dimension

        without anything further: 1->0°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)
        '''
        return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def angle_last_dim(a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension

        without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)

        return: one dimension less then input
        '''
        # print(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return abs(np.einsum('...i,...i', a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-7))


    score_per_bundle = {}
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]      # [x,y,z,3]

        angles = angle_last_dim(y_pred_bund, y_true_bund)
        angles_binary = angles > max_angle_error[0]

        gt_binary = y_true_bund.sum(axis=-1) > 0

        f1 = f1_score(gt_binary.flatten(), angles_binary.flatten(), average="binary")
        score_per_bundle[bundle] = f1

    return score_per_bundle


def calc_peak_dice_pytorch(Config, y_pred, y_true, max_angle_error=[0.9]):
    '''
    Calculate angle between groundtruth and prediction and keep the voxels where
    angle is smaller than MAX_ANGLE_ERROR.

    From groundtruth generate a binary mask by selecting all voxels with len > 0.

    Calculate Dice from these 2 masks.

    -> Penalty on peaks outside of tract or if predicted peak=0
    -> no penalty on very very small with right direction -> bad
    => Peak_dice can be high even if peaks inside of tract almost missing (almost 0)

    :param y_pred:
    :param y_true:
    :param max_angle_error:  0.7 ->  angle error of 45° or less; 0.9 ->  angle error of 23° or less
                             Can be list with several values -> calculate for several thresholds
    :return:
    '''
    import torch
    from tractseg.libs.pytorch_einsum import einsum
    from tractseg.libs import pytorch_utils

    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)

    def angle_last_dim(a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension

        without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)

        return: one dimension less then input
        '''
        return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))

    #Single threshold
    if len(max_angle_error) == 1:
        score_per_bundle = {}
        bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
        for idx, bundle in enumerate(bundles):
            # if bundle == "CST_right":
            y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()      # [x,y,z,3]

            angles = angle_last_dim(y_pred_bund, y_true_bund)
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
        bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
        for idx, bundle in enumerate(bundles):
            # if bundle == "CST_right":
            y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
            y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]

            angles = angle_last_dim(y_pred_bund, y_true_bund)
            gt_binary = y_true_bund.sum(dim=-1) > 0
            gt_binary = gt_binary.view(-1)  # [bs*x*y]

            score_per_bundle[bundle] = []
            for threshold in max_angle_error:
                angles_binary = angles > threshold
                angles_binary = angles_binary.view(-1)

                f1 = pytorch_utils.f1_score_binary(gt_binary, angles_binary)
                score_per_bundle[bundle].append(f1)

        return score_per_bundle


def calc_peak_length_dice(Config, y_pred, y_true, max_angle_error=[0.9], max_length_error=0.1):
    '''

    :param y_pred:
    :param y_true:
    :param max_angle_error:  0.7 ->  angle error of 45° or less; 0.9 ->  angle error of 23° or less
    :return:
    '''

    def angle_last_dim(a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension

        without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)

        return: one dimension less then input
        '''
        # print(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return abs(np.einsum('...i,...i', a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-7))


    score_per_bundle = {}
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]      # [x,y,z,3]

        angles = angle_last_dim(y_pred_bund, y_true_bund)

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


def calc_peak_length_dice_pytorch(Config, y_pred, y_true, max_angle_error=[0.9], max_length_error=0.1):
    '''
    Ca

    :param y_pred:
    :param y_true:
    :param max_angle_error:  0.7 ->  angle error of 45° or less; 0.9 ->  angle error of 23° or less
                             Can be list with several values -> calculate for several thresholds
    :return:
    '''
    import torch
    from tractseg.libs.pytorch_einsum import einsum
    from tractseg.libs import pytorch_utils

    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

    def angle_last_dim(a, b):
        '''
        Calculate the angle between two nd-arrays (array of vectors) along the last dimension

        without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
        np.arccos -> returns degree in pi (90°: 0.5*pi)

        return: one dimension less then input
        '''
        if len(a.shape) == 4:
            return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))
        else:
            return torch.abs(einsum('abcde,abcde->abcd', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))

    #Single threshold
    score_per_bundle = {}
    bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]
    for idx, bundle in enumerate(bundles):
        y_pred_bund = y_pred[..., (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[..., (idx * 3):(idx * 3) + 3].contiguous()      # [x,y,z,3]

        angles = angle_last_dim(y_pred_bund, y_true_bund)

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









