from tractseg.experiments.peak_reg import Config as HighResConfig


# ExpRunner --train=False --test=False --lw --config=Peaks_tmp

class Config(HighResConfig):
    EXP_NAME = "Peaks11_12g90g270_ALL_LThr05_DAugSimp"
    # CLASSES = "All"
    FEATURES_FILENAME = "270g_125mm_peaks"


