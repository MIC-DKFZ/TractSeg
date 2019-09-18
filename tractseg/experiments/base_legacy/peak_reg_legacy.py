
import numpy as np
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "peak_regression"

    CLASSES = "20"  # All / 11 / 20 / CST_right
    BATCH_SIZE = 44     #torch 0.3: 44
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1  # nr of epochs
    LABELS_TYPE = "float"
    TRAINING_SLICE_DIRECTION = "y"
    GET_PROBS = True
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
    UPSAMPLE_TYPE = "nearest"

    LR_SCHEDULE = False