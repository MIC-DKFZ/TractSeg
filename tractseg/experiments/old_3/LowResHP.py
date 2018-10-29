from tractseg.experiments.base import Config as BaseConfig
import numpy as np
class Config(BaseConfig):

    DAUG_SCALE = True
    DAUG_ELASTIC_DEFORM = False
    DAUG_ROTATE = False
    DAUG_RESAMPLE = False
    DAUG_NOISE = True

    DATASET = "HCP_32g"  # HCP / HCP_32g
    RESOLUTION = "2.5mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "32g_25mm_peaks"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    CLASSES = "11"      # All / 11 / CST_right

    MODEL = "UNet_Pytorch_Regression"
    GET_PROBS = True

    LOSS_WEIGHT = 10
    LOSS_WEIGHT_LEN = 400  # nr of epochs
    LABELS_TYPE = np.float32