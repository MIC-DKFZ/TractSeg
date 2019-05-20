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

from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "endings_segmentation"

    CLASSES = "All_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1

    # BATCH_SIZE = 30         # for all 72 (=144) classes we need smaller batch size because of memory limit
    BATCH_SIZE = 28          # Using torch 1.0 batch_size had to be still fit in memory

    FEATURES_FILENAME = "12g90g270g_CSD_BX"

    # 150ep would easily be enough
