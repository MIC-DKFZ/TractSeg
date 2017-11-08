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

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import sys
import os
import re
from os.path import join
from libs.Config import Config as C
from libs.Subjects import get_all_subjects
from libs.Utils import Utils
import numpy as np
from pprint import pprint
import glob
import copy

import matplotlib
matplotlib.use('Agg') #Solves error with ssh and plotting

#https://www.quora.com/If-a-Python-program-already-has-numerous-matplotlib-plot-functions-what-is-the-quickest-way-to-convert-them-all-to-a-way-where-all-the-plots-can-be-produced-as-hard-images-with-minimal-modification-of-code
import matplotlib.pyplot as plt

#Might fix problems with matplotlib over ssh (failing after connection is open for longer)
#   (http://stackoverflow.com/questions/2443702/problem-running-python-matplotlib-in-background-after-ending-ssh-session)
plt.ioff()

class ExpUtils:

    @staticmethod
    def create_experiment_folder(experiment_name, multi_parent_path, train):
        '''
        Create a new experiment folder. If it already exist, create new one with increasing number at the end.

        If not training model (only predicting): Use existing folder
        '''

        if multi_parent_path != "":
            dir = join(multi_parent_path, experiment_name)
        else:
            dir = join(C.EXP_PATH, experiment_name)

        if not train:
            if os.path.exists(dir):
                return dir
            else:
                sys.exit('Testing target directory does not exist!')
        else:

            for i in range(40):
                if os.path.exists(dir):
                    # tailing_numbers = re.findall('x.*?([0-9]+)$', experiment_name) #not correct
                    tailing_numbers = re.findall('x([0-9]+)$', experiment_name) #find tailing numbers that start with a x
                    if len(tailing_numbers) > 0:
                        num = int(tailing_numbers[0])
                        if num < 10:
                            experiment_name = experiment_name[:-1] + str(num+1)
                        else:
                            experiment_name = experiment_name[:-2] + str(num+1)
                    else:
                        experiment_name += "_x2"

                    if multi_parent_path != "":
                        dir = join(multi_parent_path, experiment_name)
                    else:
                        dir = join(C.EXP_PATH, experiment_name)
                else:
                    os.makedirs(dir)
                    break
            return dir

    @staticmethod
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def print_HPs(HP):
        dict = copy.deepcopy(HP.__dict__)
        dict.pop("TRAIN_SUBJECTS", None)
        dict.pop("TEST_SUBJECTS", None)
        dict.pop("VALIDATE_SUBJECTS", None)
        pprint(dict)

    @staticmethod
    def get_best_weights_path(exp_path, load_weights):
        if load_weights:
            return glob.glob(exp_path + "/best_weights_ep*.npz")[0]
        else:
            return ""

    @staticmethod
    def get_bvals_bvecs_path(args):
        if args.bvals:
            bvals = join(os.path.dirname(args.input), args.bvals)
        else:
            bvals = join(os.path.dirname(args.input), "Diffusion.bvals")  # todo: change default to "bvals"
        if args.bvecs:
            bvecs = join(os.path.dirname(args.input), args.bvecs)
        else:
            bvecs = join(os.path.dirname(args.input), "Diffusion.bvecs")  # todo: change default to "bvecs"
        return bvals, bvecs

    @staticmethod
    def get_bundle_names():

        #Comment with Indices:
        # bundles = ["BG", "AF_left", "AF_right", "ATR_left", "ATR_right", 5 "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", 12 "CC_7",
        #            "CG_left", "CG_right", 15 "CST_left", 16 "CST_right", "EMC_left", "EMC_right", "FPT_left", "FPT_right", "FX_left", 22 "FX_right",
        #            "ICP_left", "ICP_right", 25 "IFO_left", 26 "IFO_right", "ILF_left", "ILF_right", "MCP", "OR_left", 31 "OR_right",
        #            "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", 39 "SLF_II_right",
        #            "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "UF_left", 45 "UF_right"]

        # New Big   (after Bram and with Projection Tracts)  (74 Tracts)
        bundles = ["AF_left", "AF_right", "ATR_left", "ATR_right", "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", "CC_7",
                   "CG_left", "CG_right", "CST_left", "CST_right", "EMC_left", "EMC_right", "MLF_left", "MLF_right",
                   "FPT_left", "FPT_right", "FX_left", "FX_right",
                   "ICP_left", "ICP_right", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "MCP", "OR_left", "OR_right",
                   "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", "SLF_II_right",
                   "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "UF_left", "UF_right", "CC",
                   "T_PREF_left", "T_PREF_right", "T_PREM_left", "T_PREM_right", "T_PREC_left", "T_PREC_right", "T_POSTC_left",
                   "T_POSTC_right", "T_PAR_left", "T_PAR_right", "T_OCC_left", "T_OCC_right", "ST_FO_left", "ST_FO_right", "ST_PREF_left",
                   "ST_PREF_right", "ST_PREM_left", "ST_PREM_right", "ST_PREC_left", "ST_PREC_right", "ST_POSTC_left", "ST_POSTC_right",
                   "ST_PAR_left", "ST_PAR_right", "ST_OCC_left", "ST_OCC_right"]

        return ["BG"] + bundles    #Add Background label (is always beginning of list)

    @staticmethod
    def get_ACT_noACT_bundle_names():
        ACT = ["AF_left", "AF_right", "ATR_left", "ATR_right", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", "CC_7",
               "CG_left", "CG_right", "CST_left", "CST_right", "EMC_left", "EMC_right", "MLF_left", "MLF_right",
               "FPT_left", "FPT_right", "FX_left", "FX_right",
               "ICP_left", "ICP_right", "ILF_left", "ILF_right", "MCP", "OR_left", "OR_right",
               "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", "SLF_II_right",
               "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "CC",
               "T_PREF_left", "T_PREF_right", "T_PREM_left", "T_PREM_right", "T_PREC_left", "T_PREC_right", "T_POSTC_left",
               "T_POSTC_right", "T_PAR_left", "T_PAR_right", "T_OCC_left", "T_OCC_right", "ST_FO_left", "ST_FO_right", "ST_PREF_left",
               "ST_PREF_right", "ST_PREM_left", "ST_PREM_right", "ST_PREC_left", "ST_PREC_right", "ST_POSTC_left", "ST_POSTC_right",
               "ST_PAR_left", "ST_PAR_right", "ST_OCC_left", "ST_OCC_right"]

        noACT = ["CA", "IFO_left", "IFO_right", "UF_left", "UF_right"]

        return ACT, noACT

    @staticmethod
    def add_background_class(data):
        '''
        List of 3D Array with bundle masks; shape: (nr_bundles, x,y,z)
        Calculate BG class (where no other class is 1) and add it at idx=0 to array.
        Returns with nr_bundles shape in last dim: (x,y,z,nr_bundles)
        :param data:
        :return:
        '''
        s = data[0].shape
        mask_ml = np.zeros((s[0], s[1], s[2], len(data) + 1))
        background = np.ones((s[0], s[1], s[2]))  # everything that contains no bundle

        for idx in range(len(data)):
            mask = data[idx]
            mask_ml[:, :, :, idx + 1] = mask
            background[mask == 1] = 0  # remove this bundle from background

        mask_ml[:, :, :, 0] = background
        return mask_ml

    @staticmethod
    def get_cv_fold(fold):
        '''
        Brauche train-test-validate wegen Best-model selection und wegen training von combined net
        :return:
        '''

        #For CV
        if fold == 0:
            train, validate, test = [0, 1, 2], [3], [4]
        elif fold == 1:
            train, validate, test = [1, 2, 3], [4], [0]
        elif fold == 2:
            train, validate, test = [2, 3, 4], [0], [1]
        elif fold == 3:
            train, validate, test = [3, 4, 0], [1], [2]
        elif fold == 4:
            train, validate, test = [4, 0, 1], [2], [3]

        # subjects = list(Utils.chunks(get_all_subjects()[:100], 10))   #10 folds
        subjects = list(Utils.chunks(get_all_subjects(), 21))   #5 folds a 21 subjects
        # => 5 fold CV ok (score only 1%-point worse than 10 folds (80 vs 60 train subjects) (10 Fold CV impractical!)

        subjects = np.array(subjects)
        return list(subjects[train].flatten()), list(subjects[validate].flatten()), list(subjects[test].flatten())

    @staticmethod
    def print_and_save(HP, text):
        print(text)
        with open(join(HP.EXP_PATH, "Log.txt"), "a") as f:  # a for append
            f.write(text)
            f.write("\n")

    @staticmethod
    def print_verbose(HP, text):
        if HP.VERBOSE:
            print(text)

    @staticmethod
    def XXX_create_exp_plot(metrics, path, exp_name, small=False):
        #tmp method to avoid matplotlib
        pass

    @staticmethod
    def create_exp_plot(metrics, path, exp_name, without_first_epochs=False):

        min_loss_test = np.min(metrics["loss_validate"])
        min_loss_test_epoch_idx = np.argmin(metrics["loss_validate"])
        description_loss = "min loss_validate: {} (ep {})".format(round(min_loss_test, 7), min_loss_test_epoch_idx)

        max_f1_test = np.max(metrics["f1_macro_validate"])
        max_f1_test_epoch_idx = np.argmax(metrics["f1_macro_validate"])
        description_f1 = "max f1_macro_validate: {} (ep {})".format(round(max_f1_test, 4), max_f1_test_epoch_idx)

        description = description_loss + " || " + description_f1


        fig, ax = plt.subplots(figsize=(17, 5))

        # does not properly work with ax.twinx()
        # fig.gca().set_position((.1, .3, .8, .6))  # [left, bottom, width, height]; between 0-1: where it should be in result

        plt.grid(b=True, which='major', color='black', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        ax2 = ax.twinx()  # create second scale

        #shrink current axis by 5% to make room for legend next to it
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height])

        if without_first_epochs:
            plt1, = ax.plot(range(5, len(metrics["loss_train"])), metrics["loss_train"][5:], "r:", label='loss train')
            plt2, = ax.plot(range(5, len(metrics["loss_validate"])), metrics["loss_validate"][5:], "r", label='loss val')
            plt3, = ax.plot(range(5, len(metrics["loss_test"])), metrics["loss_test"][5:], "r--", label='loss test')

            plt4, = ax2.plot(range(5, len(metrics["f1_macro_train"])), metrics["f1_macro_train"][5:], "g:", label='f1_macro_train')
            plt5, = ax2.plot(range(5, len(metrics["f1_macro_validate"])), metrics["f1_macro_validate"][5:], "g", label='f1_macro_val')
            plt6, = ax2.plot(range(5, len(metrics["f1_macro_test"])), metrics["f1_macro_test"][5:], "g--", label='f1_macro_test')

            plt.legend(handles=[plt1, plt2, plt3, plt4, plt5, plt6],
                       loc=2,
                       borderaxespad=0.,
                       bbox_to_anchor=(1.03, 1))  # wenn weiter von Achse weg soll: 1.05 -> 1.15

            fig_name = "metrics.png"

        else:
            plt1, = ax.plot(metrics["loss_train"], "r:", label='loss train')
            plt2, = ax.plot(metrics["loss_validate"], "r", label='loss val')
            plt3, = ax.plot(metrics["loss_test"], "r--", label='loss test')

            plt7, = ax2.plot(metrics["f1_macro_train"], "g:", label='f1_macro_train')
            plt8, = ax2.plot(metrics["f1_macro_validate"], "g", label='f1_macro_validate')
            plt9, = ax2.plot(metrics["f1_macro_test"], "g--", label='f1_macro_test')

            plt.legend(handles=[plt1, plt2, plt3, plt7, plt8, plt9],
                       loc=2,
                       borderaxespad=0.,
                       bbox_to_anchor=(1.03, 1))

            fig_name = "metrics_all.png"


        fig.text(0.12, 0.95, exp_name, size=12, weight="bold")
        fig.text(0.12, 0.02, description)
        fig.savefig(join(path, fig_name), dpi=100)
        plt.close()
