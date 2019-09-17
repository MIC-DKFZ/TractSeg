
import numpy as np
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "peak_regression"

    CLASSES = "20"  # All / 11 / 20 / CST_right
    BATCH_SIZE = 44     #torch 0.3: 44
    # LOSS_WEIGHT = 5
    # LOSS_WEIGHT_LEN = -1  # nr of epochs
    LOSS_WEIGHT = 10
    LOSS_WEIGHT_LEN = 400  # nr of epochs
    LABELS_TYPE = "float"
    TRAINING_SLICE_DIRECTION = "y"
    GET_PROBS = True
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
    UPSAMPLE_TYPE = "nearest"
    LOSS_FUNCTION = "angle_length_loss"

    FEATURES_FILENAME = "12g90g270g_CSD_BX"

    # BEST_EPOCH_SELECTION = "loss"
    NUM_EPOCHS = 250
    METRIC_TYPES = ["loss", "f1_macro", "angle_err"]