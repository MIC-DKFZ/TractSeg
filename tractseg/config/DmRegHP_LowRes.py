from tractseg.config.DmRegHP import HP as DmRegHP
import numpy as np

class HP(DmRegHP):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"