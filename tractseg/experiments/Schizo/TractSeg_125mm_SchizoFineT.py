import os
from tractseg.experiments.tract_seg import Config as TractSegConfig
from tractseg.libs.system_config import SystemConfig as C

#This is TractSeg_12g90g270g_125mm_DAugAll
class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    LEARNING_RATE = 0.0001   # orig: 0.001
    LOAD_WEIGHTS = True
    WEIGHTS_PATH = os.path.join(C.EXP_PATH, "TractSeg_Schizo_125mm/best_weights_ep64.npz")
    # WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes", "TractSeg_T1_12g90g270g_125mm_DAugAll", "best_weights_ep126.npz")
    RESET_LAST_LAYER = True
