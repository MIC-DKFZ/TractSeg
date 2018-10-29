from tractseg.config.base import HP as BaseHP
import numpy as np

class HP(BaseHP):

    EXPERIMENT_TYPE = "peak_regression"

    MODEL = "UNet_Pytorch_Regression"
    CLASSES = "20"  # All / 11 / 20 / CST_right
    BATCH_SIZE = 44     #torch 0.3: 44
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1  # nr of epochs
    LABELS_TYPE = np.float32
    TRAINING_SLICE_DIRECTION = "y"
    GET_PROBS = True
    INFO_2 = "using AngleLengthLoss, PeakLengthDice"

    DAUG_ELASTIC_DEFORM = False

    # DATASET = "HCP"  # HCP / HCP_32g
    # RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz