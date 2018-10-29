from tractseg.config.base import HP as BaseHP
import numpy as np

class HP(BaseHP):

    EXPERIMENT_TYPE = "dm_regression"

    MODEL = "UNet_Pytorch_Regression"
    LABELS_TYPE = np.float32

    # DATASET = "HCP"  # HCP / HCP_32g
    # RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz