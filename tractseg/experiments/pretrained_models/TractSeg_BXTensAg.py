#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base_legacy.tract_seg_legacy import Config as TractSegConfig

#todo: remove this model when CSD+BX input properly integrated

class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET_FOLDER = "HCP_preproc"
    NR_OF_GRADIENTS = 18
    FEATURES_FILENAME = "12g90g270g_BX"
    P_SAMP = 0.4

