#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.data import dataset_specific_utils
from tractseg.experiments.tract_seg import Config as TractSegConfig

class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET = "HCP_all"
    DATASET_FOLDER = "HCP_preproc_all"
    FEATURES_FILENAME = "32g90g270g_CSD_BX"
    CLASSES = "xtract"
    NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(CLASSES)[1:])
    RESOLUTION = "1.25mm"
    LABELS_FILENAME = "bundle_masks_xtract_thr001"

    NUM_EPOCHS = 300
    EPOCH_MULTIPLIER = 0.5

    DAUG_ROTATE = True
    SPATIAL_TRANSFORM = "SpatialTransformPeaks"
    # rotation: 2*np.pi = 360 degree  (-> 0.8 ~ 45 degree, 0.4 ~ 22 degree))
    DAUG_ROTATE_ANGLE = (-0.4, 0.4)