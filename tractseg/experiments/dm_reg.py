
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "dm_regression"
    LABELS_TYPE = "float"
    THRESHOLD = 0.01
    FEATURES_FILENAME = "12g90g270g_CSD_BX"
