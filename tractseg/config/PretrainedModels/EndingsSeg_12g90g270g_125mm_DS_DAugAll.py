import os
from tractseg.config.endings_seg import HP as EndingsSegHP


class HP(EndingsSegHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"
