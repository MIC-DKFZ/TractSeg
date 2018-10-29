import os
from tractseg.experiments.peak_reg import Config as PeakRegConfig

#Name of original exp: Peaks20_12g90g270g_ALLoss

class Config(PeakRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1  # nr of epochs

    NUM_EPOCHS = 500

    DATA_AUGMENTATION = True
    DAUG_SCALE = True
    DAUG_NOISE = True
    DAUG_ELASTIC_DEFORM = False
    DAUG_RESAMPLE = True
    DAUG_ROTATE = False