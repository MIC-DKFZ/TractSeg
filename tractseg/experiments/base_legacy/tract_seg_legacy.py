
from tractseg.libs import exp_utils
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "tract_segmentation"

    # CLASSES = "AutoPTX"
    # NR_OF_CLASSES = len(exp_utils.get_bundle_names(CLASSES)[1:])

    LR_SCHEDULE = False