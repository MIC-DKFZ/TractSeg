import os
from tractseg.config.PeakRegHP import HP as PeakRegHP


class HP(PeakRegHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "270g_125mm_peaks"
    NUM_EPOCHS = 150
