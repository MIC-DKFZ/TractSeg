import os
from tractseg.experiments.tract_seg_lowres import Config as TractSegConfig_LowRes
from tractseg.libs.Config import Config as C

class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"
