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

# import abc
# import six

#
# #Python 2 and 3 compatible
# @six.add_metaclass(abc.ABCMeta)
# class BaseModel():
# # class BaseModel(metaclass=abc.ABCMeta):
#
#     def __init__(self, HP):
#         self.HP = HP
#
#         #Abstract instance variables that have to be defined by the network
#         self.train = None
#         self.predict = None
#         self.net = None
#         self.get_probs = None
#         self.save_model = None
#         self.load_model = None
#
#         self.create_network()
#
#     @abc.abstractmethod
#     def create_network(self):
#         '''
#         Create networks.
#         Needs to define the abstact instance variables
#         '''
#         return None
#

import os
import glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import importlib

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.libs.MetricUtils import MetricUtils
from tractseg.libs.PytorchUtils import conv2d

class BaseModel:
    def __init__(self, HP):
        self.HP = HP

        #Abstract instance variables that have to be defined by the network
        self.train = None
        self.predict = None
        self.net = None
        self.get_probs = None
        self.save_model = None
        self.load_model = None

        self.create_network()

    def create_network(self):
        # torch.backends.cudnn.benchmark = True     #not faster

        def train(X, y, weight_factor=10):
            X = torch.tensor(X, dtype=torch.float32).to(device)   # X: (bs, features, x, y)   y: (bs, classes, x, y)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            net.train()
            outputs, outputs_sigmoid = net(X)  # forward     # outputs: (bs, classes, x, y)

            if weight_factor > 1:
                # weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, self.HP.INPUT_DIM[0], self.HP.INPUT_DIM[1])).cuda()
                weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, y.shape[2], y.shape[3])).cuda()
                bundle_mask = y > 0
                weights[bundle_mask.data] *= weight_factor  # 10
                if self.HP.EXPERIMENT_TYPE == "peak_regression":
                    loss = criterion(outputs, y, weights)
                else:
                    loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
            else:
                if self.HP.LOSS_FUNCTION == "soft_sample_dice" or self.HP.LOSS_FUNCTION == "soft_batch_dice":
                    #todo important: change
                    loss = criterion(outputs_sigmoid, y)
                    # loss = criterion(outputs_sigmoid, y) + nn.BCEWithLogitsLoss()(outputs, y)
                else:
                    loss = criterion(outputs, y)

            loss.backward()  # backward
            optimizer.step()  # optimise

            if self.HP.EXPERIMENT_TYPE == "peak_regression":
                # f1 = PytorchUtils.f1_score_macro(y.data, outputs.data, per_class=True)
                # f1_a = MetricUtils.calc_peak_dice_pytorch(self.HP, outputs.data, y.data, max_angle_error=self.HP.PEAK_DICE_THR)
                f1 = MetricUtils.calc_peak_length_dice_pytorch(self.HP, outputs.detach(), y.detach(),
                                                               max_angle_error=self.HP.PEAK_DICE_THR, max_length_error=self.HP.PEAK_DICE_LEN_THR)
                # f1 = (f1_a, f1_b)
            elif self.HP.EXPERIMENT_TYPE == "dm_regression":   #density map regression
                f1 = PytorchUtils.f1_score_macro(y.detach()>0.5, outputs.detach(), per_class=True)
            else:
                f1 = PytorchUtils.f1_score_macro(y.detach(), outputs_sigmoid.detach(), per_class=True, threshold=self.HP.THRESHOLD)


            if self.HP.USE_VISLOGGER:
                probs = outputs_sigmoid.detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            else:
                probs = None    #faster

            return loss.item(), probs, f1

        def test(X, y, weight_factor=10):
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.float32).to(device)

            if self.HP.DROPOUT_SAMPLING:
                net.train()
            else:
                net.train(False)
            outputs, outputs_sigmoid = net(X)  # forward

            if weight_factor > 1:
                # weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, self.HP.INPUT_DIM[0], self.HP.INPUT_DIM[1])).cuda()
                weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, y.shape[2], y.shape[3])).cuda()
                bundle_mask = y > 0
                weights[bundle_mask.data] *= weight_factor  # 10
                if self.HP.EXPERIMENT_TYPE == "peak_regression":
                    loss = criterion(outputs, y, weights)
                else:
                    loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
            else:
                if self.HP.LOSS_FUNCTION == "soft_sample_dice" or self.HP.LOSS_FUNCTION == "soft_batch_dice":
                    #todo important: change
                    loss = criterion(outputs_sigmoid, y)
                    # loss = criterion(outputs_sigmoid, y) + nn.BCEWithLogitsLoss()(outputs, y)
                else:
                    loss = criterion(outputs, y)

            if self.HP.EXPERIMENT_TYPE == "peak_regression":
                # f1 = PytorchUtils.f1_score_macro(y.data, outputs.data, per_class=True)
                # f1_a = MetricUtils.calc_peak_dice_pytorch(self.HP, outputs.data, y.data, max_angle_error=self.HP.PEAK_DICE_THR)
                f1 = MetricUtils.calc_peak_length_dice_pytorch(self.HP, outputs.detach(), y.detach(),
                                                               max_angle_error=self.HP.PEAK_DICE_THR, max_length_error=self.HP.PEAK_DICE_LEN_THR)
                # f1 = (f1_a, f1_b)
            elif self.HP.EXPERIMENT_TYPE == "dm_regression":   #density map regression
                f1 = PytorchUtils.f1_score_macro(y.detach()>0.5, outputs.detach(), per_class=True)
            else:
                f1 = PytorchUtils.f1_score_macro(y.detach(), outputs_sigmoid.detach(), per_class=True, threshold=self.HP.THRESHOLD)

            # probs = outputs_sigmoid.detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = None  # faster
            return loss.item(), probs, f1

        def predict(X):
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).to(device)

            if self.HP.DROPOUT_SAMPLING:
                net.train()
            else:
                net.train(False)
            outputs, outputs_sigmoid = net(X)  # forward
            probs = outputs_sigmoid.detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            return probs

        def save_model(metrics, epoch_nr):
            max_f1_idx = np.argmax(metrics["f1_macro_validate"])
            max_f1 = np.max(metrics["f1_macro_validate"])
            if epoch_nr == max_f1_idx and max_f1 > 0.01:  # saving to network drives takes 5s (to local only 0.5s) -> do not save so often
                print("  Saving weights...")
                for fl in glob.glob(join(self.HP.EXP_PATH, "best_weights_ep*")):  # remove weights from previous epochs
                    os.remove(fl)
                try:
                    #Actually is a pkl not a npz
                    PytorchUtils.save_checkpoint(join(self.HP.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"), unet=net)
                except IOError:
                    print("\nERROR: Could not save weights because of IO Error\n")
                self.HP.BEST_EPOCH = epoch_nr

        def load_model(path):
            PytorchUtils.load_checkpoint(path, unet=net)

        def print_current_lr():
            for param_group in optimizer.param_groups:
                ExpUtils.print_and_save(self.HP, "current learning rate: {}".format(param_group['lr']))


        if self.HP.SEG_INPUT == "Peaks" and self.HP.TYPE == "single_direction":
            NR_OF_GRADIENTS = self.HP.NR_OF_GRADIENTS
            # NR_OF_GRADIENTS = 9
            # NR_OF_GRADIENTS = 9 * 5
            # NR_OF_GRADIENTS = 9 * 9
            # NR_OF_GRADIENTS = 33
        elif self.HP.SEG_INPUT == "Peaks" and self.HP.TYPE == "combined":
            self.HP.NR_OF_GRADIENTS = 3*self.HP.NR_OF_CLASSES
        else:
            self.HP.NR_OF_GRADIENTS = 33

        if self.HP.LOSS_FUNCTION == "soft_sample_dice":
            criterion = PytorchUtils.soft_sample_dice
        elif self.HP.LOSS_FUNCTION == "soft_batch_dice":
            criterion = PytorchUtils.soft_batch_dice
        elif self.HP.EXPERIMENT_TYPE == "peak_regression":
            criterion = PytorchUtils.angle_length_loss
        else:
            # weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, self.HP.INPUT_DIM[0], self.HP.INPUT_DIM[1])).cuda()
            # weights[:, 5, :, :] *= 10     #CA
            # weights[:, 21, :, :] *= 10    #FX_left
            # weights[:, 22, :, :] *= 10    #FX_right
            # criterion = nn.BCEWithLogitsLoss(weight=weights)
            criterion = nn.BCEWithLogitsLoss()

        NetworkClass = getattr(importlib.import_module("tractseg.models." + self.HP.MODEL), self.HP.MODEL)
        net = NetworkClass(n_input_channels=NR_OF_GRADIENTS, n_classes=self.HP.NR_OF_CLASSES, n_filt=self.HP.UNET_NR_FILT,
                   batchnorm=self.HP.BATCH_NORM, dropout=self.HP.USE_DROPOUT)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)

        # if self.HP.TRAIN:
        #     ExpUtils.print_and_save(self.HP, str(net), only_log=True)

        if self.HP.OPTIMIZER == "Adamax":
            optimizer = Adamax(net.parameters(), lr=self.HP.LEARNING_RATE)
        elif self.HP.OPTIMIZER == "Adam":
            optimizer = Adam(net.parameters(), lr=self.HP.LEARNING_RATE)
            # optimizer = Adam(net.parameters(), lr=self.HP.LEARNING_RATE, weight_decay=self.HP.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")

        if self.HP.LOAD_WEIGHTS:
            ExpUtils.print_verbose(self.HP, "Loading weights ... ({})".format(join(self.HP.EXP_PATH, self.HP.WEIGHTS_PATH)))
            load_model(join(self.HP.EXP_PATH, self.HP.WEIGHTS_PATH))

        if self.HP.RESET_LAST_LAYER:
            net.conv_5 = conv2d(self.HP.UNET_NR_FILT, self.HP.NR_OF_CLASSES, kernel_size=1, stride=1, padding=0, bias=True).to(device)

        self.train = train
        self.predict = test
        self.get_probs = predict
        self.save_model = save_model
        self.load_model = load_model
        self.print_current_lr = print_current_lr
        # self.scheduler = scheduler