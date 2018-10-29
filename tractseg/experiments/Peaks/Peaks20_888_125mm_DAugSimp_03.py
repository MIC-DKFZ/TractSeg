import os
from tractseg.experiments.peak_reg_lowres import Config as PeakRegConfig_LowRes

#Name of original exp: Peaks20_12g90g270g_ALLoss

class Config(PeakRegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    LOSS_WEIGHT = 10
    LOSS_WEIGHT_LEN = 400  # nr of epochs

    NUM_EPOCHS = 250

    DATA_AUGMENTATION = True