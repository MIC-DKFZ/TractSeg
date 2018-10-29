import os
from tractseg.experiments.tract_seg_lowres import Config as TractSegConfig_LowRes


class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 150
    CLASSES = "20"

    LOSS_FUNCTION = "soft_sample_dice"
    BATCH_NORM = True
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 40
    OPTIMIZER = "Adamax"

