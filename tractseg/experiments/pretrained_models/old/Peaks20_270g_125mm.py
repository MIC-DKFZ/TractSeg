import os
from tractseg.experiments.base_legacy.peak_reg_legacy import Config as PeakRegConfig


class Config(PeakRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "270g_125mm_peaks"
    NUM_EPOCHS = 150
