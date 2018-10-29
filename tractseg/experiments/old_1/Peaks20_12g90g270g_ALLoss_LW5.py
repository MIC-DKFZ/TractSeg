import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CLASSES = "20"

    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = 2

    INFO_2 = "using AngleLengthLoss, PeakLengthDice, LW5 constant"
