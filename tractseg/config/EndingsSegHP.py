from tractseg.config.BaseHP import HP as BaseHP

class HP(BaseHP):

    EXPERIMENT_TYPE = "endings_segmentation"

    MODEL = "UNet_Pytorch_weighted"
    CLASSES = "20_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1

    # DATASET = "HCP"
    # RESOLUTION = "1.25mm"
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
