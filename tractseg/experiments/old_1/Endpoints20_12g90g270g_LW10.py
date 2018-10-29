import os
from tractseg.experiments.tract_seg import Config as HighResClassificationConfig


class Config(HighResClassificationConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    MODEL = "UNet_Pytorch_weighted"
    CLASSES = "20_endpoints"
    LOSS_WEIGHT = 10
