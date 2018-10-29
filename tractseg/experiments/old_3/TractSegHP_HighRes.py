import os
from tractseg.experiments.tract_seg import Config as HighResClassificationConfig


class Config(HighResClassificationConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    NUM_EPOCHS = 500
    LOSS_WEIGHT_LEN = -1
    LOSS_WEIGHT = 1
    TRAINING_SLICE_DIRECTION = "xyz"
    BATCH_SIZE = 50
