from tractseg.config.tract_seg import HP as TractSegHP

class HP(TractSegHP):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"