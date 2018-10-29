import os
from tractseg.experiments.old_3.LowResClassificationConfig import Config as LowResClassificationConfig


class Config(LowResClassificationConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 400
    MODEL = "UNet_Pytorch_weighted"
    CLASSES = "20_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1
    TRAINING_SLICE_DIRECTION = "xyz"
