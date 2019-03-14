import os
from tractseg.experiments.tract_seg import Config as TractSegConfig

class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "T1"
    NR_OF_GRADIENTS = 1
