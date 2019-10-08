
from tractseg.experiments.base_legacy.peak_reg_legacy import Config as PeakRegConfig


class Config(PeakRegConfig):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"
