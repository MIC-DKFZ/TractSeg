
from tractseg.experiments.base_legacy.dm_reg_legacy import Config as DmRegConfig


class Config(DmRegConfig):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"