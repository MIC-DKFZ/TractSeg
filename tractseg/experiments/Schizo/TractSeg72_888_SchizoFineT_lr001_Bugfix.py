import os
from tractseg.experiments.tract_seg_lowres import Config as TractSegConfig_LowRes
from tractseg.libs.system_config import SystemConfig as C

class Config(TractSegConfig_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    LEARNING_RATE = 0.001   # orig: 0.001
    LOAD_WEIGHTS = True
    # WEIGHTS_PATH = os.path.join(C.EXP_PATH, "TractSeg20_888_DAugAll_DS/best_weights_ep223.npz")
    WEIGHTS_PATH = os.path.join(C.EXP_PATH, "TractSeg_Schizo_2mm/best_weights_ep102.npz")
    RESET_LAST_LAYER = True
