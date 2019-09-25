
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "endings_segmentation"
    CLASSES = "All_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1
    BATCH_SIZE = 28  # for all 72 (=144) classes we need smaller batch size because of memory limit
    FEATURES_FILENAME = "12g90g270g_CSD_BX"
    NUM_EPOCHS = 150  # easily enough if using plateau LR schedule
