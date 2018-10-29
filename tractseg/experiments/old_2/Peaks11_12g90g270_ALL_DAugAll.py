import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    INFO_2 = "using AngleLenthLoss, PeakLengthDice"
