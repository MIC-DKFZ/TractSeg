import os
from tractseg.config.dm_reg import Config as DmRegConfig


class Config(DmRegConfig):

    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True
