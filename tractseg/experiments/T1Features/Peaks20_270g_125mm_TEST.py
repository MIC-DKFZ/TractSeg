import os
from tractseg.experiments.LowRes import Config as LowRes
from tractseg.experiments.PeakReg import Config as PeakReg

class Config(PeakReg):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CLASSES = "20"
    INFO_2 = "Test_info3"
    # FEATURES_FILENAME = "270g_125mm_peaks"
    # NUM_EPOCHS = 150
    # LOSS_WEIGHT = 5
    # LOSS_WEIGHT_LEN = -1
