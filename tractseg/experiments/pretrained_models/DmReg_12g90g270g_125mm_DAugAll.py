import os
from tractseg.libs import exp_utils
from tractseg.experiments.dm_reg import Config as DmRegConfig


class Config(DmRegConfig):

    EXP_NAME = os.path.basename(__file__).split(".")[0]

    # CLASSES = "AutoPTX"
    # NR_OF_CLASSES = len(exp_utils.get_bundle_names(CLASSES)[1:])
