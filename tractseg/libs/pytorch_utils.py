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
import torch
import torch.nn as nn


def save_checkpoint(path, **kwargs):
    for key, value in list(kwargs.items()):
        if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
            kwargs[key] = value.state_dict()

    torch.save(kwargs, path)


def load_checkpoint(path, **kwargs):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    for key, value in list(kwargs.items()):
        if key in checkpoint:
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                value.load_state_dict(checkpoint[key])
            else:
                kwargs[key] = checkpoint[key]

    return kwargs


def f1_score_macro(y_true, y_pred, per_class=False, threshold=0.5):
    '''
    Macro f1

    y_true: [bs, classes, x, y]
    y_pred: [bs, classes, x, y]

    Tested: same results as sklearn f1 macro
    '''
    y_true = y_true.byte()
    y_pred = y_pred > threshold

    if len(y_true.size()) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)

        y_true = y_true.contiguous().view(-1, y_true.size()[3])  # [bs*x*y, classes]
        y_pred = y_pred.contiguous().view(-1, y_pred.size()[3])
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

        y_true = y_true.contiguous().view(-1, y_true.size()[4])  # [bs*x*y, classes]
        y_pred = y_pred.contiguous().view(-1, y_pred.size()[4])

    f1s = []
    for i in range(y_true.size()[1]):
        intersect = torch.sum(y_true[:, i] * y_pred[:, i])  # works because all multiplied by 0 gets 0
        denominator = torch.sum(y_true[:, i]) + torch.sum(y_pred[:, i])  # works because all multiplied by 0 gets 0
        #Have to cast to float here (for python3 (?)) otherwise always 0
        f1 = (2 * intersect.float()) / (denominator.float() + 1e-6)
        f1s.append(f1)
    if per_class:
        return np.array(f1s)
    else:
        return np.mean(np.array(f1s))


def f1_score_binary(y_true, y_pred):
    '''
    Binary f1

    y_true: [bs*x*y], binary
    y_pred: [bs*x*y], binary

    Tested: same results as sklearn f1 binary
    '''
    # y_true = y_true.byte()
    # y_pred = y_pred > 0.5

    # y_true = y_true.contiguous().view(-1)  # [bs*x*y]
    # y_pred = y_pred.contiguous().view(-1)

    intersect = torch.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = torch.sum(y_true) + torch.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect.float()) / (denominator.float() + 1e-6)
    return f1


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input


def soft_sample_dice(net_output, gt, eps=1e-6):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()


def soft_batch_dice(net_output, gt, eps=1e-6):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()


def MSE_weighted(y_pred, y_true, weights):
    loss = weights * ((y_pred - y_true) ** 2)
    return torch.mean(loss)


def angle_last_dim(a, b):
    '''
    Calculate the angle between two nd-arrays (array of vectors) along the last dimension

    without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
    np.arccos -> returns degree in pi (90°: 0.5*pi)

    return: one dimension less then input
    '''
    from tractseg.libs.pytorch_einsum import einsum

    return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))


def angle_second_dim(a, b):
    '''
    Not working !
    RuntimeError: invalid argument 2: input is not contiguous (and

    Calculate the angle between two nd-arrays (array of vectors) along the second dimension

    without anything further: 1->0°, 0.9->23°, 0.7->45°, 0->90°
    np.arccos -> returns degree in pi (90°: 0.5*pi)

    return: one dimension less then input
    '''
    from tractseg.libs.pytorch_einsum import einsum

    return torch.abs(einsum('abcd,abcd->acd', a, b) / (torch.norm(a, 2., 1) * torch.norm(b, 2, 1) + 1e-7))


def angle_loss(y_pred, y_true, weights):
    '''
    Not working! Have to replace np.mean by torch.mean (?)

    :param y_pred:
    :param y_true:
    :param weights:  [bs, classes, x, y, z]
    :return:
    '''
    # faster if no permute?
    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)
    weights = weights.permute(0, 2, 3, 1)

    scores = []
    nr_of_classes = int(y_true.shape[-1] / 3.)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]
        weights_bund = weights[:, :, :, (idx * 3)].contiguous()  # [x,y,z]

        angles = angle_last_dim(y_pred_bund, y_true_bund)

        angles_weighted = angles / weights_bund
        scores.append(torch.mean(angles_weighted))

    #BUG: in pytorch 0.4 this does not work anymore: have to use torch for taking mean which is derivable
    return -np.mean(scores)


def angle_length_loss(y_pred, y_true, weights):
    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)
    weights = weights.permute(0, 2, 3, 1)

    # Single threshold

    # score_per_bundle = {}
    # bundles = exp_utils.get_bundle_names(Config.CLASSES)[1:]

    nr_of_classes = int(y_true.shape[-1] / 3.)
    scores = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]
        weights_bund = weights[:, :, :, (idx * 3)].contiguous()  # [x,y,z]

        angles = angle_last_dim(y_pred_bund, y_true_bund)
        angles_weighted = angles / weights_bund
        #norm lengths to 0-1 to be more equal to angles?? -> peaks are already around 1 -> ok
        lengths = (torch.norm(y_pred_bund, 2., -1) - torch.norm(y_true_bund, 2, -1)) ** 2
        lenghts_weighted = lengths * weights_bund

        # Divide by weights.max otherwise lens would be way bigger
        #   Would also work: just divide by inverted weights_bund
        #   -> keeps foreground the same and penalizes the background less
        #   (weights.max just simple way of getting the current weight factor
        #   (because weights_bund is tensor, but we want scalar))
        #   Flip angles to make it a minimization problem
        combined = -angles_weighted + lenghts_weighted / weights_bund.max()

        #Would this work? (mathematically not the same (!), but correct concept?)
        # combined = weights_bund * (-angles + lengths)

        scores[idx] = torch.mean(combined)

    return torch.mean(scores)


def angle_loss_faster(y_pred, y_true, weights):
    '''
    Not working !
    RuntimeError: invalid argument 2: input is not contiguous (and 'abcd,abcd->acd' also in numpy wrong)

    :param y_pred:
    :param y_true:
    :param weights:
    :return:
    '''
    scores = []
    nr_of_classes = int(y_true.shape[-1] / 3.)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[:, (idx * 3):(idx * 3) + 3, :, :].contiguous()
        y_true_bund = y_true[:, (idx * 3):(idx * 3) + 3, :, :].contiguous()  # [x,y,z,3]
        weights_bund = weights[:, (idx * 3), :, :].contiguous()  # [x,y,z]

        angles = angle_second_dim(y_pred_bund, y_true_bund)

        angles_weighted = angles / weights_bund
        scores.append(torch.mean(angles_weighted))

    #BUG: in pytorch 0.4 this does not work anymore: have to use torch for taking mean which is derivable
    return -np.mean(scores)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    # nonlinearity = nn.ReLU()
    nonlinearity = nn.LeakyReLU()

    if batchnorm:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    # nonlinearity = nn.ReLU()
    nonlinearity = nn.LeakyReLU()

    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer


def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    # nonlinearity = nn.ReLU()
    nonlinearity = nn.LeakyReLU()

    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    # nonlinearity = nn.ReLU()
    nonlinearity = nn.LeakyReLU()

    layer = nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer