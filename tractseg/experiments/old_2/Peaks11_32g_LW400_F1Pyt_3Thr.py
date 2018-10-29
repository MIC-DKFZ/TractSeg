from tractseg.experiments.old_3.LowResConfig import Config as LowResConfig

class Config(LowResConfig):
    PEAK_DICE_THR = [0.6, 0.8, 0.95]
    # PEAK_DICE_THR = [0.9]
    CALC_F1 = True
    LOSS_WEIGHT_LEN = 400


