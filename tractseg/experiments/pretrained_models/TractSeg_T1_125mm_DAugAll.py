import os
from tractseg.experiments.base_legacy.tract_seg_legacy import Config as TractSegConfig

class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "T1"
    NR_OF_GRADIENTS = 1
