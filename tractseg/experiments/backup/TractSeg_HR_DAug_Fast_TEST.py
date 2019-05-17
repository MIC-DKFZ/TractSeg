#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base_legacy.tract_seg_legacy import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    # DATASET_FOLDER = "HCP_preproc_bedpostX"
    DATASET_FOLDER = "HCP_preproc"
    BATCH_SIZE = 47

    #pad to 144 (not to multiple of 16)

    # p_scale_per_sample=0.8, for resample 0.2
    DAUG_RESAMPLE = False
    DAUG_GAUSSIAN_BLUR = True
