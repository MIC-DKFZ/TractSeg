import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    PEAK_DICE_LEN_THR = 0.05

    INFO_2 = "using AngleLenthLoss, PeakLengthDice"