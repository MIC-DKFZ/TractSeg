#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import pickle
import time
from os.path import join
from pprint import pprint
import nibabel as nib
import numpy as np
from libs.ExpUtils import ExpUtils
from libs.ImgUtils import ImgUtils
from libs.MetricUtils import MetricUtils
from libs.Trainer import Trainer
from libs.DatasetUtils import DatasetUtils
from libs.DirectionMerger import DirectionMerger


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

class ExpRunner():

    @staticmethod
    def experiment(HP):

        if HP.TRAIN:
            HP.EXP_PATH = ExpUtils.create_experiment_folder(HP.EXP_NAME, HP.MULTI_PARENT_PATH, HP.TRAIN)

        DataManagerSingleSubjectById = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectById")
        DataManagerTraining = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerTrainingNiftiImgs")

        def test_whole_subject(HP, model, subjects, type):

            # Metrics traditional
            metrics = {
                "loss_" + type: [0],
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
            test_whole_subject(HP, model_test, HP.VALIDATE_SUBJECTS, "validate")
            test_whole_subject(HP, model_test, HP.TEST_SUBJECTS, "test")

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


    @staticmethod
    def predict_img(HP):
        start_time = time.time()
        data_img = nib.load(join(HP.PREDICT_IMG_OUTPUT, "peaks.nii.gz"))
        data, transformation = DatasetUtils.pad_and_scale_img_to_square_img(data_img.get_data(), target_size=144)

        ModelClass = getattr(importlib.import_module("models." + HP.MODEL), HP.MODEL)
        model = ModelClass(HP)

        # DataManagerSingleSubjectByFile = getattr(importlib.import_module("libs." + "DataManagers"), "DataManagerSingleSubjectByFile")
        # dataManagerSingle = DataManagerSingleSubjectByFile(HP, data=data)
        # trainerSingle = Trainer(model, dataManagerSingle)
        # seg, gt = trainerSingle.get_seg_single_img(HP, probs=False, scale_to_world_shape=False)
        seg_xyz, gt = DirectionMerger.get_seg_single_img_3_directions(HP, model, data=data, scale_to_world_shape=False)
        seg = DirectionMerger.mean_fusion(HP.THRESHOLD, seg_xyz, probs=False)

        seg = DatasetUtils.cut_and_scale_img_back_to_original_img(seg, transformation)
        ExpUtils.print_verbose(HP, "Took {}s".format(round(time.time() - start_time, 2)))

        if HP.OUTPUT_MULTIPLE_FILES:
            ImgUtils.save_multilabel_img_as_multiple_files(seg, data_img.get_affine(), HP.PREDICT_IMG_OUTPUT)   # Save as several files
        else:
            img = nib.Nifti1Image(seg, data_img.get_affine())
            nib.save(img, join(HP.PREDICT_IMG_OUTPUT, "bundle_segmentations.nii.gz"))
