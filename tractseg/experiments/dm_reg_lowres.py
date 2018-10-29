from tractseg.experiments.dm_reg import Config as DmRegConfig
import numpy as np

class Config(DmRegConfig):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"