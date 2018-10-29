from tractseg.config.tract_seg import Config as TractSegConfig

class Config(TractSegConfig):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"