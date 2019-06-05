import os
from tractseg.experiments.peak_reg_angle import Config as PeakRegConfig

class Config(PeakRegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    CLASSES = "All"

