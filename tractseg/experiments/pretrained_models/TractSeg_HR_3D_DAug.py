#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.tract_seg import Config as TractSegConfig

class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    MODEL = "UNet3D_Pytorch_DeepSup_sm"
    DIM = "3D"
    FP16 = False
    UPSAMPLE_TYPE = "trilinear"

    BATCH_SIZE = 1
    UNET_NR_FILT = 8

    DATA_AUGMENTATION = True


"""
Memory consumption
NR_FILT=8: 10900MB

System RAM running full (>30GB) from DAug -> super slow (without DAug: GPU utility 90%)
"""