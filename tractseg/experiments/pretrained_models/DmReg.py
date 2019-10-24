#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.data import dataset_specific_utils
from tractseg.experiments.dm_reg import Config as DmRegConfig

class Config(DmRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
