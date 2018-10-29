import os
from tractseg.experiments.old_3.TractSegConfig_LowRes import Config as TractSegConfig_LowRes


class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = False
    FEATURES_FILENAME = "T1_25mm"
    NR_OF_GRADIENTS = 1
    NUM_EPOCHS = 150

    LOSS_FUNCTION = "soft_batch_dice"
    BATCH_NORM = True
    LEARNING_RATE = 1e-4        #0.001=1e-3
    BATCH_SIZE = 40