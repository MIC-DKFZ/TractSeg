from tractseg.config.endings_seg import HP as EndingsSegHP

class HP(EndingsSegHP):

    DATASET = "HCP_32g"
    RESOLUTION = "2.5mm"
    FEATURES_FILENAME = "32g_25mm_peaks"