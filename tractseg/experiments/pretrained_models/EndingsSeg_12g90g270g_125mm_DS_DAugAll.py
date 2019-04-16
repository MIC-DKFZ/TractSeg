import os
from tractseg.experiments.endings_seg import Config as EndingsSegConfig


class Config(EndingsSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
