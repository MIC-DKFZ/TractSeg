import os
from tractseg.experiments.old_3.LowResConfig import Config as LowResConfig

class Config(LowResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    PEAK_DICE_LEN_THR = 0.05
    CLASSES = "All"

    INFO_2 = "using AngleLengthLoss, PeakLengthDice"