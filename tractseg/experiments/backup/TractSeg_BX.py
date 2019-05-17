#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.base_legacy.tract_seg_legacy import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    # DATA_AUGMENTATION = False
    MODEL = "UNet_Pytorch_DeepSup"

    DATASET_FOLDER = "HCP_preproc_bedpostX"
    # DATASET_FOLDER = "HCP_preproc"
    BATCH_SIZE = 47

    #pad to 144 (not to multiple of 16)

    P_SAMP = 0.4
    DAUG_RESAMPLE = False
    DAUG_GAUSSIAN_BLUR = True

    NR_OF_GRADIENTS = 18
    FEATURES_FILENAME = "125mm_bedpostx_tensor"