
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

from tractseg.data import dataset_specific_utils
from tractseg.libs.system_config import SystemConfig as C


class Config:
    """
    Settings and hyperparameters
    """

    # input data
    EXPERIMENT_TYPE = "tract_segmentation"  # tract_segmentation|endings_segmentation|dm_regression|peak_regression
    EXP_NAME = "HCP_TEST"
    EXP_MULTI_NAME = ""  # CV parent directory name; leave empty for single bundle experiment
    DATASET_FOLDER = "HCP_preproc"
    LABELS_FOLDER = "bundle_masks"
    MULTI_PARENT_PATH = join(C.EXP_PATH, EXP_MULTI_NAME)
    EXP_PATH = join(C.EXP_PATH, EXP_MULTI_NAME, EXP_NAME)  # default path
    CLASSES = "All"
    NR_OF_GRADIENTS = 9
    NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(CLASSES)[1:])
    INPUT_DIM = None  # autofilled
    DATASET = "HCP"  # HCP | HCP_32g | Schizo
    RESOLUTION = "1.25mm"  # 1.25mm|2.5mm
    # 12g90g270g | 270g_125mm_xyz | 270g_125mm_peaks | 90g_125mm_peaks | 32g_25mm_peaks | 32g_25mm_xyz
    FEATURES_FILENAME = "12g90g270g"
    LABELS_FILENAME = ""  # autofilled
    LABELS_TYPE = "int"
    THRESHOLD = 0.5  # Binary: 0.5, Regression: 0.01

    # hyperparameters
    MODEL = "UNet_Pytorch_DeepSup"
    DIM = "2D"  # 2D | 3D
    BATCH_SIZE = 47
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LR_SCHEDULE_PATIENCE = 20
    UNET_NR_FILT = 64
    EPOCH_MULTIPLIER = 1  # 2D: 1, 3D: 12 for lowRes, 3 for highRes
    NUM_EPOCHS = 250
    SLICE_DIRECTION = "y"  # x | y | z  ("combined" needs z)
    TRAINING_SLICE_DIRECTION = "xyz"  # y | xyz
    LOSS_FUNCTION = "default"  # default | soft_batch_dice
    OPTIMIZER = "Adamax"
    LOSS_WEIGHT = None  # None = no weighting
    LOSS_WEIGHT_LEN = -1  # -1 = constant over all epochs
    BATCH_NORM = False
    WEIGHT_DECAY = 0
    USE_DROPOUT = False
    DROPOUT_SAMPLING = False
    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "My_experiment/best_weights_ep64.npz")
    WEIGHTS_PATH = ""  # if empty string: autoloading the best_weights in get_best_weights_path()
    SAVE_WEIGHTS = True
    TYPE = "single_direction"  # single_direction | combined
    CV_FOLD = 0
    VALIDATE_SUBJECTS = []
    TRAIN_SUBJECTS = []
    TEST_SUBJECTS = []
    TRAIN = True
    TEST = True
    SEGMENT = False
    GET_PROBS = False
    OUTPUT_MULTIPLE_FILES = False
    RESET_LAST_LAYER = False
    UPSAMPLE_TYPE = "bilinear"  # bilinear | nearest
    BEST_EPOCH_SELECTION = "f1"  # f1 | loss
    METRIC_TYPES = ["loss", "f1_macro"]
    FP16 = True
    PEAK_DICE_THR = [0.95]
    PEAK_DICE_LEN_THR = 0.05
    FLIP_OUTPUT_PEAKS = False  # flip peaks along z axis to make them compatible with MITK
    USE_VISLOGGER = False
    SEG_INPUT = "Peaks"  # Gradients | Peaks
    NR_SLICES = 1
    PRINT_FREQ = 20
    NORMALIZE_DATA = True
    NORMALIZE_PER_CHANNEL = False
    BEST_EPOCH = 0
    VERBOSE = True
    CALC_F1 = True
    ONLY_VAL = False
    TEST_TIME_DAUG = False
    PAD_TO_SQUARE = True
    INPUT_RESCALING = False  # Resample data to different resolution (instead of doing in preprocessing))

    # data augmentation
    DATA_AUGMENTATION = True
    DAUG_SCALE = True
    DAUG_NOISE = True
    DAUG_NOISE_VARIANCE = (0, 0.05)
    DAUG_ELASTIC_DEFORM = True
    DAUG_ALPHA = (90., 120.)
    DAUG_SIGMA = (9., 11.)
    DAUG_RESAMPLE = False  # does not improve validation dice (if using Gaussian_blur) -> deactivate
    DAUG_RESAMPLE_LEGACY = False  # does not improve validation dice (at least on AutoPTX) -> deactivate
    DAUG_GAUSSIAN_BLUR = True
    DAUG_BLUR_SIGMA = (0, 1)
    DAUG_ROTATE = False
    DAUG_ROTATE_ANGLE = (-0.2, 0.2)  # rotation: 2*np.pi = 360 degree  (0.4 ~= 22 degree, 0.2 ~= 11 degree))
    DAUG_MIRROR = False
    DAUG_FLIP_PEAKS = False
    SPATIAL_TRANSFORM = "SpatialTransform"  # SpatialTransform|SpatialTransformPeaks
    P_SAMP = 1.0
    DAUG_INFO = "-"
    INFO = "-"

    # for inference
    PREDICT_IMG = False
    PREDICT_IMG_OUTPUT = None
    TRACTSEG_DIR = "tractseg_output"
    KEEP_INTERMEDIATE_FILES = False
    CSD_RESOLUTION = "LOW"  # HIGH | LOW
    NR_CPUS = -1
