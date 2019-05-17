import os
from tractseg.experiments.base_legacy.endings_seg_legacy import Config as EndingsSegConfig


class Config(EndingsSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True