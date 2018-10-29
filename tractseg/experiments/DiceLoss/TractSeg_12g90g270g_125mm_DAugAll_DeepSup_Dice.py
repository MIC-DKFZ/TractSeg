import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    LOSS_FUNCTION = "soft_batch_dice"
    BATCH_NORM = True
    LEARNING_RATE = 1e-4  # 0.001=1e-3
    BATCH_SIZE = 40
    OPTIMIZER = "Adam"
