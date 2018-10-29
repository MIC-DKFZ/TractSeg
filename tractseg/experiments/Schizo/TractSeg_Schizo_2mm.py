import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    NUM_EPOCHS = 250
    DATA_AUGMENTATION = True
    MODEL = "UNet_Pytorch_DeepSup"

    DATASET = "Schizo"
    DATASET_FOLDER = "Schizo"
    FEATURES_FILENAME = "30g_2mm_peaks"
    RESOLUTION = "2mm"