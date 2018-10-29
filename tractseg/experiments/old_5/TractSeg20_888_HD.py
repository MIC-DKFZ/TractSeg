import os
from tractseg.experiments.tract_seg_lowres import Config as TractSegConfig_LowRes


class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = False
    # NUM_EPOCHS = 500

    CLASSES = "20"  # All / 11 / CST_right