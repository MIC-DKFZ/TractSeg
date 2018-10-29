import os
from tractseg.experiments.peak_reg import Config as HighResConfig


class Config(HighResConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    CALC_F1 = True
    BATCH_SIZE = 2


