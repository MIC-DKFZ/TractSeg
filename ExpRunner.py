#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import pickle
import sys
import time
from os.path import join
from pprint import pprint
import nibabel as nib
import numpy as np
from libs.Config import Config as C
from libs.ExpUtils import ExpUtils
from libs.ImgUtils import ImgUtils
from libs.MetricUtils import MetricUtils
from libs.Trainer import Trainer
from libs.DatasetUtils import DatasetUtils
from libs.DirectionMerger import DirectionMerger


'''
Adapt for Predict by file:
- search for "Adapt for PredictByFile" -> 2 places
'''

'''
Adapt for Fusion Training:
- BATCH_SIZE=42
- TYPE=combined
- FEATURES_FILENAME=270g_125mm_xyz  (-> needed for testing and seg)
- DATASET_FOLDER=HCP_fusion_npy_270g_125mm   (-> needed for testing and seg)
- for creating npy files from probmaps: Slicer.py
- if mean: adapt SlicesBatchGeneratorRandomNpyImg_fusionMean

Adapt for 32g_25mm prediction:
- DATASET=HCP_32g
- FEATURES_FILENAME=32g_25mm_peaks
- 32g_25mm_peaks not available on new Cluster at the moment
'''

import warnings
warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings

#Hyperparameters
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
    # SLOPE = -1  # 2  # Slope bigger -> lossWeights decay less / remain bigger over epochs
    # W_DECAY_LEN = -1  # 100
    # Each epoch LR is 97% of LR of epoch before
    #   0.95 -> 1/10 after 45ep, 1/2 after 10ep; 0.9 -> 1/10 after 20 ep; 0.8 -> 1/10 after 10 ep (so bei ma_arbeit)
    #   0.97 -> 1/10 after 70ep; 0.98 -> 1/10 after 110ep; 0.992 -> 1/10 after 280ep
    # LR_DECAY = 1.00   #not using anymore, because not better when using Adam


HP = ExpUtils.read_program_parameters(sys.argv[1:], HP)
HP.TRAIN_SUBJECTS, HP.VALIDATE_SUBJECTS, HP.TEST_SUBJECTS = ExpUtils.get_cv_fold(HP.CV_FOLD)
# HP.TRAIN_SUBJECTS, HP.VALIDATE_SUBJECTS, HP.TEST_SUBJECTS = ExpUtils.get_cv_fold_TRACED(HP.CV_FOLD)
EXP_NAME_ORIG = HP.EXP_NAME  # store beginning of exp_name for multi_bundle experiments

print("Hyperparameters:")
ExpUtils.print_HPs(HP)

if HP.TRAIN:
    HP.EXP_PATH = ExpUtils.create_experiment_folder(HP.EXP_NAME, HP.MULTI_PARENT_PATH, HP.TRAIN)
    # if HP.TYPE == "single_direction" and not HP.DATA_PATH:
    #     ExpUtils.make_dir(join(HP.EXP_PATH, "data"))
    #     Slicer.create_files(HP, HP.SLICE_DIRECTION, HP.TRAIN_SUBJECTS, HP.VALIDATE_SUBJECTS, HP.TEST_SUBJECTS)

# if HP.TYPE == "combined" and not HP.DATA_PATH:
#     ExpUtils.make_dir(join(HP.EXP_PATH, "combined"))
#     Slicer.create_prob_files(HP, HP.BUNDLE, HP.TRAIN_SUBJECTS, HP.VALIDATE_SUBJECTS, HP.TEST_SUBJECTS)

DataManagerSingleSubjectById = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectById")
DataManagerSingleSubjectByFile = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectByFile")

DataManagerTraining = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerTrainingNiftiImgs")


def test_whole_subject(HP, model, subjects, type):

    # Metrics traditional
    metrics = {
        "loss_" + type: [0],
        "acc_" + type: [0],
        "f1_binary_" + type: [0],
        "f1_samples_" + type: [0],
        "f1_micro_" + type: [0],
        "f1_macro_" + type: [0],
    }

    # Metrics per bundle
    metrics_bundles = {}
    for bundle in ExpUtils.get_bundle_names():
        metrics_bundles[bundle] = [0]

    for subject in subjects:
        print("{} subject {}".format(type, subject))
        start_time = time.time()

        dataManagerSingle = DataManagerSingleSubjectById(HP, subject=subject)
        trainerSingle = Trainer(model, dataManagerSingle)
        img_probs, img_y = trainerSingle.get_seg_single_img(HP, probs=True)
        # img_probs_xyz, img_y = DirectionMerger.get_seg_single_img_3_directions(HP, model, subject=subject)
        # igm_probs = DirectionMerger.mean_fusion(HP.THRESHOLD, img_probs_xyz, probs=True)

        print("Took {}s".format(round(time.time() - start_time, 2)))

        img_probs = np.reshape(img_probs, (-1, img_probs.shape[-1]))  #Flatten all dims except nrClasses dim
        img_y = np.reshape(img_y, (-1, img_y.shape[-1]))

        metrics = MetricUtils.calculate_metrics(metrics, img_y, img_probs, 0, type=type, threshold=HP.THRESHOLD)
        metrics_bundles = MetricUtils.calculate_metrics_each_bundle(metrics_bundles, img_y, img_probs, ExpUtils.get_bundle_names(), threshold=HP.THRESHOLD)

    metrics = MetricUtils.normalize_last_element(metrics, len(subjects), type=type)
    metrics_bundles = MetricUtils.normalize_last_element_general(metrics_bundles, len(subjects))

    print("WHOLE SUBJECT:")
    pprint(metrics)
    print("WHOLE SUBJECT BUNDLES:")
    pprint(metrics_bundles)


    with open(join(HP.EXP_PATH, "score_" + type + "-set.txt"), "w") as f:
        pprint(metrics, f)
        f.write("\n\nWeights: {}\n".format(HP.WEIGHTS_PATH))
        f.write("type: {}\n\n".format(type))
        pprint(metrics_bundles, f)

    pickle.dump(metrics, open(join(HP.EXP_PATH, "score_" + type + ".pkl"), "wb"))

    return metrics


dataManager = DataManagerTraining(HP)
ModelClass = getattr(importlib.import_module("models." + HP.MODEL), HP.MODEL)
model = ModelClass(HP)
trainer = Trainer(model, dataManager)

if HP.TRAIN:
    print("Training...")
    metrics = trainer.train(HP)


#After Training
if HP.TRAIN:
    # have to load other weights, because after training it has the weights of the last epoch
    print("Loading best epoch: {}".format(HP.BEST_EPOCH))
    HP.WEIGHTS_PATH = HP.EXP_PATH + "/best_weights_ep" + str(HP.BEST_EPOCH) + ".npz"
    HP.LOAD_WEIGHTS = True
    # model_test = ModelClass(HP) #takes long; has to recompile model

    trainer.model.load_model(join(HP.EXP_PATH, HP.WEIGHTS_PATH))
    # print("Loading weights ... ({})".format(join(HP.EXP_PATH, HP.WEIGHTS_PATH)))
    # with np.load(join(HP.EXP_PATH, HP.WEIGHTS_PATH)) as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # L.layers.set_all_param_values(trainer.model.output, param_values)

    model_test = trainer.model
else:
    # Weight_path already set to best model (wenn reading program parameters) -> will be loaded automatically
    model_test = trainer.model


if HP.SEGMENT:
    ExpUtils.make_dir(join(HP.EXP_PATH, "segmentations"))
    # all_subjects = HP.VALIDATE_SUBJECTS #+ HP.TEST_SUBJECTS
    all_subjects = HP.TEST_SUBJECTS
    for subject in all_subjects:
        print("Get_segmentation subject {}".format(subject))
        start_time = time.time()

        dataManagerSingle = DataManagerSingleSubjectById(HP, subject=subject, use_gt_mask=False)
        trainerSingle = Trainer(model_test, dataManagerSingle)
        img_seg, img_y = trainerSingle.get_seg_single_img(HP, probs=False)  # only x or y or z
        # img_seg, img_y = DirectionMerger.get_seg_single_img_3_directions(HP, model, subject)  #returns probs not binary seg

        # ImgUtils.save_multilabel_img_as_multiple_files(HP, img_seg, subject)   # Save as several files
        img = nib.Nifti1Image(img_seg, ImgUtils.get_dwi_affine(HP.DATASET, HP.RESOLUTION))
        nib.save(img, join(HP.EXP_PATH, "segmentations", subject + "_segmentation.nii.gz"))
        print("took {}s".format(time.time() - start_time))


if HP.TEST:
    test_metrics = test_whole_subject(HP, model_test, HP.VALIDATE_SUBJECTS, "validate")
    test_metrics = test_whole_subject(HP, model_test, HP.TEST_SUBJECTS, "test")


if HP.GET_PROBS:
    ExpUtils.make_dir(join(HP.EXP_PATH, "probmaps"))
    # ExpUtils.make_dir(join(HP.EXP_PATH, "probmaps_32g_25mm"))
    all_subjects = HP.TEST_SUBJECTS
    # all_subjects = HP.TRAIN_SUBJECTS + HP.VALIDATE_SUBJECTS + HP.TEST_SUBJECTS
    for subject in all_subjects:
        print("Get_probs subject {}".format(subject))

        # dataManagerSingle = DataManagerSingleSubjectById(HP, subject=subject, use_gt_mask=False)
        # trainerSingle = Trainer(model_test, dataManagerSingle)
        # img_probs, img_y = trainerSingle.get_seg_single_img(HP, probs=True)
        img_probs, img_y = DirectionMerger.get_seg_single_img_3_directions(HP, model, subject=subject)

        #Save as one probmap for further combined training
        img = nib.Nifti1Image(img_probs, ImgUtils.get_dwi_affine(HP.DATASET, HP.RESOLUTION))
        nib.save(img, join(HP.EXP_PATH, "probmaps", subject + "_probmap.nii.gz"))
        # nib.save(img, join(HP.EXP_PATH, "probmaps_32g_25mm", subject + "_probmap.nii.gz"))

#Command:
# python ExpRunner.py --train=False --test=False --lw=True --predict_img=/mnt/jakob/E130-Personal/Wasserthal/HCP/599469/32g_25mm/peaks.nii.gz --predict_img_out=/mnt/jakob/E130-Personal/Wasserthal/tmp/Prediction.nii.gz
# python ExpRunner.py --train=False --test=False --lw=True --predict_img=/ad/wasserth/E130-Personal/Wasserthal/HCP/599469/270g_125mm/peaks.nii.gz --predict_img_out=/ad/wasserth/E130-Personal/Wasserthal/tmp/Prediction_gpuNode.nii.gz
if HP.PREDICT_IMG:
    start_time = time.time()
    data_img = nib.load(HP.PREDICT_IMG)
    data, transformation = DatasetUtils.pad_and_scale_img_to_square_img(data_img.get_data(), target_size=144)

    # dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
    # trainerSingle = Trainer(model, dataManagerSingle)
    # seg, gt = trainerSingle.get_seg_single_img(HP, probs=False, scale_to_world_shape=False)
    seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False)
    seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=False)

    seg = DatasetUtils.cut_and_scale_img_back_to_original_img(seg, transformation)
    print("Took {}s".format(round(time.time() - start_time, 2)))

    img = nib.Nifti1Image(seg, data_img.get_affine())
    # nib.save(img, join(C.NETWORK_DRIVE, "tmp", "Prediction.nii.gz"))
    nib.save(img, HP.PREDICT_IMG_OUT)
