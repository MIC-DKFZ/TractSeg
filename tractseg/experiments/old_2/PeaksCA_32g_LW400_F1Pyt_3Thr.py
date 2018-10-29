from tractseg.experiments.old_3.LowResConfig import Config as LowResConfig

class Config(LowResConfig):
    NUM_EPOCHS = 2000
    PEAK_DICE_THR = [0.6, 0.8, 0.95]
    CALC_F1 = True
    LOSS_WEIGHT_LEN = 400
    CLASSES = "CA"  # All / 11 / CST_right


