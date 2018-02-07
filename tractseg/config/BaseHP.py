from os.path import join
import numpy as np
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.Config import Config as C

#Settings and Hyperparameters
class HP:
    EXP_MULTI_NAME = ""             #CV Parent Dir name # leave empty for Single Bundle Experiment
    EXP_NAME = "HCP_TEST"           # HCP_TEST
    MODEL = "UNet_Pytorch"          # UNet_Lasagne / UNet_Pytorch
    NUM_EPOCHS = 500
    DATA_AUGMENTATION = True
    DAUG_INFO = "Elastic(90,120)(9,11) - Scale(0.9, 1.5) - CenterDist60 - DownsampScipy(0.5,1) - Contrast(0.7,1.3) - Gaussian(0,0.05) - BrightnessMult(0.7,1.3) - RotateUltimate(-0.8,0.8)"
    DATASET = "HCP"  # HCP / HCP_32g
    RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "270g_125mm_peaks"  # 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    LABELS_FILENAME = "bundle_masks_72"     # bundle_masks / bundle_masks_72 / bundle_masks_45       #Only used when using DataManagerNifti
    DATASET_FOLDER = "HCP"  # HCP / TRACED / HCP_fusion_npy_270g_125mm / HCP_fusion_npy_32g_25mm
    LABELS_FOLDER = "bundle_masks"  # bundle_masks / bundle_masks_dm
    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    BATCH_SIZE = 50  #max:  #Pytorch: 50  #Lasagne: 56  #Lasagne combined: 42  #Pytorch UpSample: 56   #Pytorch_SE_r16: 45    #Pytorch_SE_r64: 45
    LEARNING_RATE = 0.001  # 0.002 #LR find: 0.000143 ?  # 0.001
    UNET_NR_FILT = 64
    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "HCP100_45B_UNet_x_DM_lr002_slope2_dec992_ep800/best_weights_ep64.npz")    # Can be absolute path or relative like "exp_folder/weights.npz"
    WEIGHTS_PATH = ""   # if empty string: autoloading the best_weights in get_best_weights_path()
    TYPE = "single_direction"       # single_direction / combined
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []
    TRAIN = True
    TEST = True         # python ExpRunner.py --train=False --seg=False --test=True --lw=True
    SEGMENT = False
    GET_PROBS = False   # python ExpRunner.py --train=False --seg=False --probs=True --lw=True
    OUTPUT_MULTIPLE_FILES = False

    #Unimportant / rarly changed:
    LABELS_TYPE = np.int16  # Binary: np.int16, Regression: np.float32
    THRESHOLD = 0.5  # Binary: 0.5, Regression: 0.01 ?
    TEST_TIME_DAUG = False
    SLICE_DIRECTION = "x"  #no effect at the moment     # x, y, z  (combined needs z)
    USE_VISLOGGER = False
    INFO = "no Dropout, Deconv, 72bundles, LeakyRelu, "
    SAVE_WEIGHTS = True
    NR_OF_CLASSES = len(ExpUtils.get_bundle_names())
    SEG_INPUT = "Peaks"     # Gradients/ Peaks
    NR_SLICES = 1           # adapt manually: NR_OF_GRADIENTS in UNet.py and get_batch... in train() and in get_seg_prediction()
    PRINT_FREQ = 20
    NORMALIZE_DATA = True
    BEST_EPOCH = 0
    INPUT_DIM = (144, 144)
    VERBOSE = True