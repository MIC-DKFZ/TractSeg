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

import torch
import numpy as np

class PytorchUtils:

    @staticmethod
    def save_checkpoint(path, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                kwargs[key] = value.state_dict()

        torch.save(kwargs, path)

    @staticmethod
    def load_checkpoint(path, **kwargs):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

        for key, value in kwargs.items():
            if key in checkpoint:
                if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                    value.load_state_dict(checkpoint[key])
                else:
                    kwargs[key] = checkpoint[key]

        return kwargs

    @staticmethod
    def f1_score_macro(y_true, y_pred, per_class=False):
        '''
        Macro f1

        y_true: [bs, classes, x, y]
        y_pred: [bs, classes, x, y]

        Tested: same results as sklearn f1 macro
        '''
        y_true = y_true.byte()
        y_pred = y_pred > 0.5

        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)

        y_true = y_true.contiguous().view(-1, y_true.size()[3])  # [bs*x*y, classes]
        y_pred = y_pred.contiguous().view(-1, y_pred.size()[3])

        f1s = []
        for i in range(y_true.size()[1]):
            intersect = torch.sum(y_true[:, i] * y_pred[:, i])  # works because all multiplied by 0 gets 0
            denominator = torch.sum(y_true[:, i]) + torch.sum(y_pred[:, i])  # works because all multiplied by 0 gets 0
            f1 = (2 * intersect) / (denominator + 1e-6)
            f1s.append(f1)
        if per_class:
            return np.array(f1s)
        else:
            return np.mean(np.array(f1s))

    @staticmethod
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
        f1 = (2 * intersect) / (denominator + 1e-6)
        return f1


    @staticmethod
    def sum_tensor(input, axes, keepdim=False):
        axes = np.unique(axes)
        if keepdim:
            for ax in axes:
                input = input.sum(ax, keepdim=True)
        else:
            for ax in sorted(axes, reverse=True):
                input = input.sum(ax)
        return input

    @staticmethod
    def soft_dice(net_output, gt, eps=1e-6):
        axes = tuple(range(2, len(net_output.size())))
        intersect = PytorchUtils.sum_tensor(net_output * gt, axes, keepdim=False)
        denom = PytorchUtils.sum_tensor(net_output + gt, axes, keepdim=False)
        return - (2 * intersect / (denom + eps)).mean()

