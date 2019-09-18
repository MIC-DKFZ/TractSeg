
from tractseg.experiments.base_legacy.endings_seg_legacy import Config as EndingsSegConfig


class Config(EndingsSegConfig):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"