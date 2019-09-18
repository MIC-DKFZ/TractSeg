
import numpy as np
from tractseg.experiments.base import Config as BaseConfig


class Config(BaseConfig):

    EXPERIMENT_TYPE = "dm_regression"

    LABELS_TYPE = "float"

    THRESHOLD = 0.01  # Binary: 0.5, Regression: 0.01 ?

    LR_SCHEDULE = False

    # DATASET = "HCP"  # HCP / HCP_32g
    # RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz