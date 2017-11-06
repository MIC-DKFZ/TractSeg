import argparse
from os.path import join
import numpy as np
from libs.Config import Config as C
from libs.ExpUtils import ExpUtils
from libs.Mrtrix import Mrtrix

import warnings
warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings

#Settings and Hyperparameters
class HP:
    EXP_MULTI_NAME = ""              #CV Parent Dir name # leave empty for Single Bundle Experiment
    EXP_NAME = "HCP_TEST"

    MODEL = "UNet_Pytorch"     # UNet_Multilabel_diceScore / UNet_Pytorch
    NUM_EPOCHS = 300
    DATA_AUGMENTATION = True
    # DAUG_INFO = "Elastic(90,120)(9,11) - Scale(0.9, 1.5) - CenterDist60 - DownsampScipy(0.5,1) - Contrast(0.7,1.3) - Gaussian(0,0.05) - BrightnessMult(0.7,1.3) - RotateUltimate(-0.8,0.8) - Mirror"
    DAUG_INFO = "Elastic(90,120)(9,11) - Scale(0.9, 1.5) - CenterDist60 - DownsampScipy(0.5,1) - Contrast(0.7,1.3) - Gaussian(0,0.05) - BrightnessMult(0.7,1.3) - RotateUltimate(-0.8,0.8)"

    DATASET = "HCP"  # HCP / HCP_32g                     #OLD: / HCP_2mm / HCP_2.5mm (needed for Probmap Training) / TRACED / Phantom
    RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)                 #Old: 2mm
    FEATURES_FILENAME = "270g_125mm_peaks"  # 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz    #OLD: peaks.nii.gz / 32g/peaks.nii.gz / peaks_no_artifacts_hcpSize.nii.gz
    LABELS_FILENAME = "bundle_masks"     # bundle_masks / bundle_masks_45       #Only used when using DataManagerNifti
    DATASET_FOLDER = "HCP"  # HCP / TRACED / HCP_fusion_npy_270g_125mm / HCP_fusion_npy_32g_25mm
    LABELS_FOLDER = "bundle_masks"  # bundle_masks / bundle_masks_dm / bundle_masks_traced

    # DATA_PATH = ""  # put empty string to create slices on the fly
    # DATA_PATH = join(C.HOME, "HCP_aggregated", "HCP100_125mm_45Bun_xyz")
    # DATA_PATH = join(C.HOME, "HCP_aggregated", "HCP100_125mm_45Bun_x")    # HCP100_125mm_45Bun_x_DM /  HCP100_125mm_45Bun_x

    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    SLICE_DIRECTION = "x"  #no effect at the moment     # x, y, z  (combined needs z)
    BATCH_SIZE = 46  # Lasagne: 56  # Lasagne combined: 42   #Pytorch: 46
    LEARNING_RATE = 0.002  # UNet: 0.002     #Simon: 0.0001
    UNET_NR_FILT = 64
    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "HCP100_45B_UNet_x_DM_lr002_slope2_dec992_ep800/best_weights_ep64.npz")    # Can be absolute path or relative like "exp_folder/weights.npz"
    WEIGHTS_PATH = ""   # autoloading the best_weights in read_program_parameters()

    TYPE = "single_direction"       # single_direction / combined / 3D
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []

    TRAIN = True
    TEST = True  # python ExpRunner.py --train=False --seg=False --test=True --lw=True
    SEGMENT = False
    GET_PROBS = False  # python ExpRunner.py --train=False --seg=False --probs=True --lw=True
    PREDICT_IMG = None  # python ExpRunner.py --train=False --test=False --lw=True --predict_img=/mnt/jakob/E130-Personal/Wasserthal/VISIS/s01/243g_25mm/peaks.nii.gz
    PREDICT_IMG_OUT = None

    # For DM Regression
    # Also adapt LABELS_FOLDER
    LABELS_TYPE = np.int16  # Binary: np.int16, Regression: np.float32
    THRESHOLD = 0.5           # Binary: 0.5, Regression: 0.01 ?
    TEST_TIME_DAUG = False

    #Unimportant / rarly changed:
    USE_VISLOGGER = False
    INFO = "74 BNew, DMNifti, newSplit, 90gAnd270g, NormBeforeDAug, Fusion: 32gAnd270g"
    SAVE_WEIGHTS = True
    NR_OF_CLASSES = len(ExpUtils.get_bundle_names())
    FRAMEWORK = "Lasagne"
    SEG_INPUT = "Peaks"  # alt    # Gradients/ Peaks
    NR_SLICES = 1           # adapt manually: NR_OF_GRADIENTS in UNet.py and get_batch... in train() and in get_seg_prediction()
    PRINT_FREQ = 20
    BUNDLE = "CST_right"
    SHUFFLE = True
    NORMALIZE_DATA = True
    BEST_EPOCH = 0
    INPUT_DIM = (144, 144)

parser = argparse.ArgumentParser(description="Process some integers.",
                                    epilog="Written by Jakob Wasserthal. Please reference TODO")
#todo: make input optional -> if no input => expert training mode
parser.add_argument("-i", metavar="filename", dest="input", help="Diffusion Input image (Nifti image)")
#https://stackoverflow.com/questions/20048048/argparse-default-option-based-on-another-option
parser.add_argument("-o", metavar="filename", dest="output", help="Output segmentation file")
parser.add_argument("--output_multiple_files", action="store_true", help="Create extra output file for each bundle", default=False)
parser.add_argument("--bvals", metavar="filename", help="bvals file. Default is './bvals'")
parser.add_argument("--bvecs", metavar="filename", help="bvecs file. Default is './bvecs'")
parser.add_argument("--train", action="store_true", help="Train network", default=True)
parser.add_argument("--test", action="store_true", help="Test network", default=True)
parser.add_argument("--seg", action="store_true", help="Create binary segmentation", default=False)   #todo: better API
parser.add_argument("--probs", action="store_true", help="Create probmap segmentation", default=False)   #todo: better API
parser.add_argument("--lw", action="store_true", help="Load weights of pretrained net", default=False)   #todo: better API
parser.add_argument("--en", metavar="name", help="Experiment name")
parser.add_argument("--fold", metavar="N", help="Which fold to train when doing CrossValidation", type=int, default=0)
parser.add_argument('--version', action='version', version='TractQuerier 1.0')
args = parser.parse_args()

HP.PREDICT_IMG = args.input is not None
HP.TRAIN = args.train
HP.TEST = args.test
HP.seg = args.seg
HP.probs = args.probs
HP.lw = args.lw
HP.en = args.en
HP.fold = args.fold

HP.MULTI_PARENT_PATH = join(C.EXP_PATH, HP.EXP_MULTI_NAME)
HP.EXP_PATH = join(C.EXP_PATH, HP.EXP_MULTI_NAME, HP.EXP_NAME)

#todo: set path to delivered pretrained weights
if HP.WEIGHTS_PATH == "":
    HP.WEIGHTS_PATH = ExpUtils.get_best_weights_path(HP.EXP_PATH, HP.LOAD_WEIGHTS)



if HP.PREDICT_IMG:
    print("Segmenting bundles...")
    Mrtrix.create_brain_mask(args.input)
    Mrtrix.create_fods(args.input)

    #todo: run Experiment Runner
    # python ~/dev/dl-tracking/ExpRunner.py --train=False --test=False --lw=True \
    # --en=HCP_fold0\
    # --predict_img=peaks.nii.gz \
    # --predict_img_out=bundle_segmentation.nii.gz
else:
    #todo train
    a = 0