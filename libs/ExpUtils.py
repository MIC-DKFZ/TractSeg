#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import sys
import getopt
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
matplotlib.use('Agg') #maybe solves error I get with ssh after a while => schein das Problem gelöst zu haben

#https://www.quora.com/If-a-Python-program-already-has-numerous-matplotlib-plot-functions-what-is-the-quickest-way-to-convert-them-all-to-a-way-where-all-the-plots-can-be-produced-as-hard-images-with-minimal-modification-of-code
import matplotlib.pyplot as plt

#Might fix problems with matplotlib over ssh (failing after connection is open for longer)
#   (http://stackoverflow.com/questions/2443702/problem-running-python-matplotlib-in-background-after-ending-ssh-session)
plt.ioff()

class ExpUtils:

    @staticmethod
    def read_program_parameters(argv, HP):
        try:
            opts, args = getopt.getopt(argv, "", ["bs=", "np=", "pf=", "sl=", "en=", "lr=", "bun=", "lw=", "sw=", "wpath=",
                                                  "train=", "test=", "seg=", "probs=", "sdir=", "slope=", "normalize=",
                                                  "dataset=", "resolution=", "fold=", "type=", "enm=", "wdec=", "lrdec=",
                                                  "daug=", "nrfilt=", "predict_img=", "predict_img_out=", "vislogger="])
        except getopt.GetoptError:
            print('invalid parameters')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("--bs"):
                HP.BATCH_SIZE = int(arg)
            elif opt in ("--np"):
                HP.NUM_EPOCHS = int(arg)
            elif opt in ("--pf"):
                HP.PRINT_FREQ = int(arg)
            elif opt in ("--sl"):
                HP.SEQ_LEN = int(arg)
            elif opt in ("--en"):
                HP.EXP_NAME = arg
            elif opt in ("--enm"):
                HP.EXP_MULTI_NAME = arg
            elif opt in ("--lr"):
                HP.LEARNING_RATE = float(arg)
            elif opt in ("--bun"):
                HP.BUNDLE = arg
            elif opt in ("--lw"):
                HP.LOAD_WEIGHTS = arg == "True"
            elif opt in ("--sw"):
                HP.SAVE_WEIGHTS = arg == "True"
            elif opt in ("--wpath"):
                HP.WEIGHTS_PATH = arg
            elif opt in ("--train"):
                HP.TRAIN = arg == "True"
            elif opt in ("--test"):
                HP.TEST = arg == "True"
            elif opt in ("--seg"):
                HP.SEGMENT = arg == "True"
            elif opt in ("--probs"):
                HP.GET_PROBS = arg == "True"
            elif opt in ("--sdir"):
                HP.SLICE_DIRECTION = arg
            elif opt in ("--slope"):
                HP.SLOPE = int(arg)
            elif opt in ("--normalize"):
                HP.NORMALIZE_DATA = arg == "True"
            elif opt in ("--dataset"):
                HP.DATASET = arg
            elif opt in ("--resolution"):
                HP.RESOLUTION = arg
            elif opt in ("--fold"):
                HP.CV_FOLD= int(arg)
            elif opt in ("--type"):
                HP.TYPE = arg
            elif opt in ("--wdec"):
                HP.W_DECAY_LEN = int(arg)
            elif opt in ("--lrdec"):
                HP.LR_DECAY = float(arg)
            elif opt in ("--daug"):
                HP.DATA_AUGMENTATION = arg == "True"
            elif opt in ("--nrfilt"):
                HP.UNET_NR_FILT = int(arg)
            elif opt in ("--predict_img"):
                HP.PREDICT_IMG = arg
            elif opt in ("--predict_img_out"):
                HP.PREDICT_IMG_OUT = arg
            elif opt in ("--vislogger"):
                HP.USE_VISLOGGER = arg == "True"




        HP.MULTI_PARENT_PATH = join(C.EXP_PATH, HP.EXP_MULTI_NAME)
        HP.EXP_PATH = join(C.EXP_PATH, HP.EXP_MULTI_NAME, HP.EXP_NAME)
        if HP.WEIGHTS_PATH == "":
            HP.WEIGHTS_PATH = ExpUtils.get_best_weights_path(HP.EXP_PATH, HP.LOAD_WEIGHTS)

        if HP.RESOLUTION == "1.25mm":
            HP.INPUT_DIM = (144, 144)
        elif HP.RESOLUTION == "2mm" or HP.RESOLUTION == "2.5mm":
            HP.INPUT_DIM = (80, 80)
        return HP


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
            print("Getting best weights (path: {})".format(exp_path + "/best_weights_ep*.npz"))
            return glob.glob(exp_path + "/best_weights_ep*.npz")[0]
        else:
            return ""

    @staticmethod
    def get_bundle_names():


        #Comment with Indices:
        # bundles = ["BG", "AF_left", "AF_right", "ATR_left", "ATR_right", 5 "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", 12 "CC_7",
        #            "CG_left", "CG_right", 15 "CST_left", 16 "CST_right", "EMC_left", "EMC_right", "FPT_left", "FPT_right", "FX_left", 22 "FX_right",
        #            "ICP_left", "ICP_right", 25 "IFO_left", 26 "IFO_right", "ILF_left", "ILF_right", "MCP", "OR_left", 31 "OR_right",
        #            "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", 39 "SLF_II_right",
        #            "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "UF_left", 45 "UF_right"]

        #New Big    (pre Bram)      (used for many experiments)
        # bundles = ["AF_left", "AF_right", "ATR_left", "ATR_right", "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", "CC_7",
        #            "CG_left", "CG_right", "CST_left", "CST_right", "EMC_left", "EMC_right", "FPT_left", "FPT_right", "FX_left", "FX_right",
        #            "ICP_left", "ICP_right",  "IFO_left", "IFO_right", "ILF_left", "ILF_right", "MCP", "OR_left", "OR_right",
        #            "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", "SLF_II_right",
        #            "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "UF_left", "UF_right"]

        # New Big   (after Bram)
        # bundles = ["AF_left", "AF_right", "ATR_left", "ATR_right", "CA", "CC_1", "CC_2", "CC_3", "CC_4", "CC_5", "CC_6", "CC_7",
        #            "CG_left", "CG_right", "CST_left", "CST_right", "EMC_left", "EMC_right", "FPT_left", "FPT_right", "FX_left", "FX_right",
        #            "ICP_left", "ICP_right", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "MCP", "OR_left", "OR_right",
        #            "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_I_left", "SLF_I_right", "SLF_II_left", "SLF_II_right",
        #            "SLF_III_left", "SLF_III_right", "STR_left", "STR_right", "UF_left", "UF_right", "CC"]

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

        #Phantom
        # bundles = ["CA", "CC", "Cingulum_left", "Cingulum_right", "CP", "CST_left", "CST_right", "Fornix", "FPT_left", "FPT_right",
        #            "ICP_left", "ICP_right", "ILF_left", "ILF_right", "IOFF_left", "IOFF_right", "MCP", "OR_left", "OR_right",
        #            "POPT_left", "POPT_right", "SCP_left", "SCP_right", "SLF_left", "SLF_right", "UF_left", "UF_right"]

        #TRACED OLD
        # bundles = ["CC_2", "CC_7", "CG_left", "CG_right", "CST_left", "CST_right", "FX",  "IFO_left", "IFO_right",
        #            "SFO_left", "SFO_right", "ILF_left", "ILF_right", "SLF_left", "SLF_right", "UF_left", "UF_right"]

        #TRACED
        # bundles = ["UF_left", "UF_right", "FX_left", "FX_right", "CC_2", "CG_left", "CG_right", "CST_left", "CST_right", "CC_7",
        #            "ILF_left", "ILF_right", "SLF_left", "SLF_right", "IFO_left", "IFO_right"]

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

        #For Dev
        # train, validate, test = [0, 1, 2, 3, 4, 5, 6, 7], [8], [9]
        # train, validate, test = [0, 1, 2, 3, 4, 5], [8], [9]  #HCP80
        # train, validate, test = [0, 1, 2], [8], [9]     #HCP50

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
        #       -> and by selecting best epoch during CV we get +1% -> in the end same performance if 3 or 4 folds for training

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
    def XXX_create_exp_plot(metrics, path, exp_name, small=False, only_f1=False):
        #tmp method to avoid matplotlib
        a = 0

    @staticmethod
    def create_exp_plot(metrics, path, exp_name, small=False, only_f1=False):

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

        if small:
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

            fig_name = "metrics_small.png"

        elif only_f1:
            plt1, = ax2.plot(range(5, len(metrics["f1_binary_train"])), metrics["f1_binary_train"][5:], "g:", label='f1_binary_train')
            plt2, = ax2.plot(range(5, len(metrics["f1_binary_validate"])), metrics["f1_binary_validate"][5:], "g", label='f1_binary_val')
            plt3, = ax2.plot(range(5, len(metrics["f1_binary_test"])), metrics["f1_binary_test"][5:], "g--", label='f1_binary_test')

            plt4, = ax.plot(range(5, len(metrics["f1_micro_train"])), metrics["f1_micro_train"][5:], "k:", label='f1_micro_train')
            plt5, = ax.plot(range(5, len(metrics["f1_micro_validate"])), metrics["f1_micro_validate"][5:], "k", label='f1_micro_val')
            plt6, = ax.plot(range(5, len(metrics["f1_micro_test"])), metrics["f1_micro_test"][5:], "k--", label='f1_micro_test')

            plt7, = ax2.plot(range(5, len(metrics["f1_macro_train"])), metrics["f1_macro_train"][5:], "b:", label='f1_macro_train')
            plt8, = ax2.plot(range(5, len(metrics["f1_macro_validate"])), metrics["f1_macro_validate"][5:], "b", label='f1_macro_val')
            plt9, = ax2.plot(range(5, len(metrics["f1_macro_test"])), metrics["f1_macro_test"][5:], "b--", label='f1_macro_test')

            plt.legend(handles=[plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9],
                       loc=2,
                       borderaxespad=0.,
                       bbox_to_anchor=(1.03, 1))

            fig_name = "metrics_f1.png"

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

            fig_name = "metrics.png"


        fig.text(0.12, 0.95, exp_name, size=12, weight="bold")
        fig.text(0.12, 0.02, description)
        fig.savefig(join(path, fig_name), dpi=100)
        plt.close()
