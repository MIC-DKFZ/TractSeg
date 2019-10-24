#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DAUG_ROTATE = True
    SPATIAL_TRANSFORM = "SpatialTransformPeaks"
    # rotation: 2*np.pi = 360 degree  (-> 0.8 ~ 45 degree, 0.4 ~ 22 degree))
    DAUG_ROTATE_ANGLE = (-0.4, 0.4)

