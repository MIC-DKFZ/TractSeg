import os
from tractseg.config.tract_seg import HP as TractSegHP

class HP(TractSegHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    FEATURES_FILENAME = "T1"
    NR_OF_GRADIENTS = 1

    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True