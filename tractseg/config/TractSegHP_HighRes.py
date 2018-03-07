import os
from tractseg.config.HighResClassificationHP import HP as HighResClassificationHP
from tractseg.config.LowResClassificationHP import HP as LowResClassificationHP
import numpy as np

class HP(HighResClassificationHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    NUM_EPOCHS = 500
    LOSS_WEIGHT = 1
    LOSS_WEIGHT_LEN = -1
    TRAINING_SLICE_DIRECTION = "xyz"
