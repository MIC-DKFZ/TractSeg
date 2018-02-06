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


from __future__ import division
import numpy as np
from sklearn.metrics import f1_score
from tractseg.libs.ExpUtils import ExpUtils
# from medpy import metric

class MetricUtils:

    @staticmethod
    def my_f1_score(y_true, y_pred):
        '''
        Binary f1

        Tested: same results as sklearn f1 binary
        '''
        intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
        denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
        f1 = (2 * intersect) / (denominator + 1e-6)
        return f1

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def normalize_last_element(metrics, length, type):
        '''

        :param metrics:
        :param length:
        :param type:  "train" or "test"
        :return:
        '''
        for key, value in metrics.iteritems():
            if key.endswith("_" + type):
                metrics[key][-1] /= float(length)
        return metrics

    @staticmethod
    def normalize_last_element_general(metrics, length):
        for key, value in metrics.iteritems():
            metrics[key][-1] /= float(length)
        return metrics

    @staticmethod
    def add_empty_element(metrics):
        for key, value in metrics.iteritems():
            metrics[key].append(0)
        return metrics

    @staticmethod
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
            metrics["f1_macro_"+type][-1] += MetricUtils.my_f1_score_macro(y, pred_class)
        else:
            metrics["f1_macro_"+type][-1] += f1
            if f1_per_bundle is not None:
                metrics["f1_CA_" + type][-1] += f1_per_bundle["CA"]
                # metrics["f1_FX_left_" + type][-1] += f1_per_bundle["FX_left"]
                # metrics["f1_FX_right_" + type][-1] += f1_per_bundle["FX_right"]
                metrics["f1_CST_right_" + type][-1] += f1_per_bundle["CST_right"]
                metrics["f1_UF_left_" + type][-1] += f1_per_bundle["UF_left"]

        return metrics

    @staticmethod
    def calculate_metrics_onlyLoss(metrics, loss, type="train"):
        metrics["loss_"+type][-1] += loss
        return metrics

    @staticmethod
    def calculate_metrics_each_bundle(metrics, y, class_probs, bundles, threshold=0.5):
        '''
        bundles -> have to be in same order as classes in predictions
        y -> Ground Truth
        class_probs -> Predictions
        '''

        class_probs[class_probs >= threshold] = 1
        class_probs[class_probs < threshold] = 0
        pred_class = class_probs.astype(np.int16)

        y[y >= threshold] = 1
        y[y < threshold] = 0
        y = y.astype(np.int16)

        for idx, bundle in enumerate(bundles):
            metrics[bundle][-1] += f1_score(y[:,idx], pred_class[:,idx], average="binary")
        return metrics

    @staticmethod
    def average_metric_all_bundles(metrics_all):
        '''
        For each experiment: Takes last element of each metric
            -> then: take average
        => Average of all metrics for all experiments
        :param metrics_all: list of metrics dictionaries
        :return:
        '''

        metrics_avg = {}
        metric_keys = metrics_all[0].keys()

        for metric_key in metric_keys:

            elems = []
            for experiment in metrics_all:
                elems.append(experiment[metric_key][-1])

            metrics_avg[metric_key] = sum(elems) / len(elems)

        return metrics_avg

    @staticmethod
    def calc_peak_dice_onlySeg(y_pred, y_true):
        '''
        Create binary mask of peaks by simple thresholding. Then calculate Dice.

        :param y_pred:
        :param y_true:
        :return:
        '''

        score_per_bundle = {}
        bundles = ExpUtils.get_bundle_names()[1:]
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

    @staticmethod
    def calc_peak_dice(y_pred, y_true, max_angle_error=0.9):
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
        bundles = ExpUtils.get_bundle_names()[1:]
        for idx, bundle in enumerate(bundles):
            y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3]
            y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3]      # [x,y,z,3]

            angles = angle_last_dim(y_pred_bund, y_true_bund)
            angles_binary = angles > max_angle_error

            gt_binary = y_true_bund.sum(axis=-1) > 0

            f1 = f1_score(gt_binary.flatten(), angles_binary.flatten(), average="binary")
            score_per_bundle[bundle] = f1

        return score_per_bundle











