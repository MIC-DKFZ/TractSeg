import os
from tractseg.config.tract_seg import HP as TractSegHP


class HP(TractSegHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = True
    FEATURES_FILENAME = "T1_Peaks12g90g270g"
    NR_OF_GRADIENTS = 10

    NUM_EPOCHS = 500

    NORMALIZE_PER_CHANNEL = True