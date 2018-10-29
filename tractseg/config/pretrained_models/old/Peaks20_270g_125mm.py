import os
from tractseg.config.peak_reg import Config as PeakRegConfig


class Config(PeakRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "270g_125mm_peaks"
    NUM_EPOCHS = 150
