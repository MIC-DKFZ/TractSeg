
from tractseg.experiments.peak_reg import Config as PeakRegConfig

class Config(PeakRegConfig):

    CLASSES = "All_Part1"  # All_Part1 | All_Part2 | All_Part3 | All_Part4
    LOSS_WEIGHT = 1  # None not possible for PeakReg experiments
    LOSS_WEIGHT_LEN = -1
    LOSS_FUNCTION = "angle_loss"
    METRIC_TYPES = ["loss", "f1_macro"]
    BEST_EPOCH_SELECTION = "loss"
    NUM_EPOCHS = 150

    FP16 = False  # loss always NaN for Peak Reg with fp16