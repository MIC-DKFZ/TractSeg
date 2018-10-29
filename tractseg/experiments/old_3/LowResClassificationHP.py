import os
from tractseg.libs import exp_utils
import numpy as np
from tractseg.experiments.base import Config as BaseConfig

class Config(BaseConfig):

    DAUG_SCALE = True
    DAUG_ELASTIC_DEFORM = False
    DAUG_ROTATE = False
    DAUG_RESAMPLE = False
    DAUG_NOISE = True

    DATASET = "HCP_32g"  # HCP / HCP_32g
    RESOLUTION = "2.5mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "32g_25mm_peaks"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    CLASSES = "20"  # All / 11 / CST_right

    # MODEL = "UNet_Pytorch"
    # LABELS_TYPE = np.int16