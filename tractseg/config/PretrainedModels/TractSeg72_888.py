import os
from tractseg.config.tract_seg_lowres import HP as TractSegHP_LowRes

class HP(TractSegHP_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    INPUT_DIM = (80, 80)