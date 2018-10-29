import os
from tractseg.experiments.old_3.LowResConfig import Config as LowResConfig

class Config(LowResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CLASSES = "20"
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
