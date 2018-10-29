import os
from tractseg.experiments.old_3.TractSegConfig_HighRes import Config as TractSegConfig_HighRes


class Config(TractSegConfig_HighRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
