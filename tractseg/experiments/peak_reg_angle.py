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
from tractseg.experiments.peak_reg import Config as PeakRegConfig

class Config(PeakRegConfig):

    CLASSES = "All_Part1"  # All_Part1 / All_Part2 / All_Part3 / All_Part4

    LOSS_WEIGHT = 1  # None not possible for PeakReg experiments
    LOSS_WEIGHT_LEN = -1
    LOSS_FUNCTION = "angle_loss"
    METRIC_TYPES = ["loss", "f1_macro"]
    BEST_EPOCH_SELECTION = "loss"

    NUM_EPOCHS = 150
