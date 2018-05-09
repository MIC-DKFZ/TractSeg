from tractseg.config.BaseHP import HP as BaseHP

class HP(BaseHP):

    EXPERIMENT_TYPE = "tract_segmentation"

    # DATASET = "HCP"
    # RESOLUTION = "1.25mm"
    # FEATURES_FILENAME = "12g90g270g"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz