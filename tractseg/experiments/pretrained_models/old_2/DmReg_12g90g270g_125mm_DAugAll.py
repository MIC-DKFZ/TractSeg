import os
from tractseg.experiments.base_legacy.dm_reg_legacy import Config as DmRegConfig


class Config(DmRegConfig):

    EXP_NAME = os.path.basename(__file__).split(".")[0]

    # CLASSES = "AutoPTX"
    # NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(CLASSES)[1:])
