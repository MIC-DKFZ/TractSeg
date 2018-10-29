import os
from tractseg.experiments.old_3.LowResConfig import Config as LowResConfig

class Config(LowResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CLASSES = "20"
    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
