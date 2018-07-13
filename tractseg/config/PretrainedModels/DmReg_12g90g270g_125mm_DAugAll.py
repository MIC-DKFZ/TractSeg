import os
from tractseg.config.DmRegHP import HP as DmRegHP


class HP(DmRegHP):

    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True
