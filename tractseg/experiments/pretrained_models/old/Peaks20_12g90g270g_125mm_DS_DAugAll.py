import os
from tractseg.experiments.base_legacy.peak_reg_legacy import Config as PeakRegConfig

#Name of original exp: Peaks20_12g90g270g_ALLoss

class Config(PeakRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    LOSS_WEIGHT = 10
    LOSS_WEIGHT_LEN = 400  # nr of epochs


