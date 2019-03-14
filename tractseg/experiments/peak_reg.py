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

import numpy as np
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "peak_regression"

    CLASSES = "20"  # All / 11 / 20 / CST_right
    BATCH_SIZE = 44     #torch 0.3: 44
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1  # nr of epochs
    LABELS_TYPE = np.float32
    TRAINING_SLICE_DIRECTION = "y"
    GET_PROBS = True
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
    UPSAMPLE_TYPE = "nearest"

