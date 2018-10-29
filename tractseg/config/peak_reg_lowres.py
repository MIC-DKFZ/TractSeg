from tractseg.config.peak_reg import HP as PeakRegHP

class HP(PeakRegHP):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"