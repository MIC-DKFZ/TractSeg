import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    INFO_2 = "using AngleLenthLoss, PeakLengthDice"
    CLASSES = "CST_right"
    NUM_EPOCHS = 1000
    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True
