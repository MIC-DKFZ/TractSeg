import os
from tractseg.libs.ExpUtils import ExpUtils
import numpy as np
from tractseg.config.BaseHP import HP as BaseHP

class HP(BaseHP):

    DATASET = "HCP_32g"  # HCP / HCP_32g
    RESOLUTION = "2.5mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "32g_25mm_peaks"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    CLASSES = "All"  # All / 11 / CST_right

    MODEL = "UNet_Pytorch"
    LABELS_TYPE = np.int16