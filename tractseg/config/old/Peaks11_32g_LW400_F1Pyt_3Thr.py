from tractseg.config.HighResHP import HP as HighResHP
from tractseg.config.LowResHP import HP as LowResHP

class HP(LowResHP):
    PEAK_DICE_THR = [0.6, 0.8, 0.95]
    # PEAK_DICE_THR = [0.9]
    CALC_F1 = True
    LOSS_WEIGHT_LEN = 400


