import os
from tractseg.experiments.endings_seg import Config as EndingsSegConfig


class Config(EndingsSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 150
    FEATURES_FILENAME = "270g_125mm_peaks"
