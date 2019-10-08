
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "peak_regression"
    CLASSES = "20"
    BATCH_SIZE = 44
    LOSS_WEIGHT = 10
    LOSS_WEIGHT_LEN = 400  # nr of epochs
    LABELS_TYPE = "float"
    TRAINING_SLICE_DIRECTION = "y"
    GET_PROBS = True
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"
    UPSAMPLE_TYPE = "nearest"
    LOSS_FUNCTION = "angle_length_loss"
    FEATURES_FILENAME = "12g90g270g_CSD_BX"
    NUM_EPOCHS = 250
    METRIC_TYPES = ["loss", "f1_macro", "angle_err"]