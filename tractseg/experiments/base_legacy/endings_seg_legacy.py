
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "endings_segmentation"

    CLASSES = "All_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1

    # BATCH_SIZE = 30         # for all 72 (=144) classes we need smaller batch size because of memory limit
    BATCH_SIZE = 28          # Using torch 1.0 batch_size had to be still fit in memory

    LR_SCHEDULE = False