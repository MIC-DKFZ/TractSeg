import os
from tractseg.experiments.tract_seg_lowres import Config as TractSegConfig_LowRes


class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    CLASSES = "20"
    NUM_EPOCHS = 100
    DATA_AUGMENTATION = False
    MODEL = "UNet_Pytorch_DeepSup"

    USE_VISLOGGER = False        # use ts_env_py3 env for that