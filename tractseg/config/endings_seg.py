from tractseg.config.base import HP as BaseHP

class HP(BaseHP):

    EXPERIMENT_TYPE = "endings_segmentation"

    MODEL = "UNet_Pytorch"
    # CLASSES = "20_endpoints"
    CLASSES = "All_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1
    BATCH_SIZE = 30         #for all 72 (=144) classes we need smaller batch size because of memory limit

    # DATASET = "HCP"
    # RESOLUTION = "1.25mm"
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
