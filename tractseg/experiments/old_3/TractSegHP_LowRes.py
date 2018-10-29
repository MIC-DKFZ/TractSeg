import os
from tractseg.experiments.old_3.LowResClassificationConfig import Config as LowResClassificationConfig


class Config(LowResClassificationConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    NUM_EPOCHS = 500
    LOSS_WEIGHT_LEN = -1
    LOSS_WEIGHT = 1
    TRAINING_SLICE_DIRECTION = "xyz"
    BATCH_SIZE = 50
