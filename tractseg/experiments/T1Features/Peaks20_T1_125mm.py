import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CLASSES = "20"
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"

    FEATURES_FILENAME = "T1"
    NR_OF_GRADIENTS = 1

    NUM_EPOCHS = 150
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1

