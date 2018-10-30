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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from os.path import join
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

from tractseg.libs import pytorch_utils
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils


class BaseModel:
    def __init__(self, Config):
        self.Config = Config

        # torch.backends.cudnn.benchmark = True     #not faster
        if self.Config.NR_CPUS > 0:
            torch.set_num_threads(self.Config.NR_CPUS)

        if self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "single_direction":
            NR_OF_GRADIENTS = self.Config.NR_OF_GRADIENTS
            # NR_OF_GRADIENTS = 9 * 5    # 5 slices
        elif self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "combined":
            self.Config.NR_OF_GRADIENTS = 3 * self.Config.NR_OF_CLASSES
        else:
            self.Config.NR_OF_GRADIENTS = 33

        if self.Config.LOSS_FUNCTION == "soft_sample_dice":
            self.criterion = pytorch_utils.soft_sample_dice
        elif self.Config.LOSS_FUNCTION == "soft_batch_dice":
            self.criterion = pytorch_utils.soft_batch_dice
        elif self.Config.EXPERIMENT_TYPE == "peak_regression":
            self.criterion = pytorch_utils.angle_length_loss
        else:
            # weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES, self.Config.INPUT_DIM[0], self.Config.INPUT_DIM[1])).cuda()
            # weights[:, 5, :, :] *= 10     #CA
            # weights[:, 21, :, :] *= 10    #FX_left
            # weights[:, 22, :, :] *= 10    #FX_right
            # self.criterion = nn.BCEWithLogitsLoss(weight=weights)
            self.criterion = nn.BCEWithLogitsLoss()

        NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()), self.Config.MODEL)
        self.net = NetworkClass(n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES, n_filt=self.Config.UNET_NR_FILT,
                           batchnorm=self.Config.BATCH_NORM, dropout=self.Config.USE_DROPOUT)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net.to(self.device)

        # if self.Config.TRAIN:
        #     exp_utils.print_and_save(self.Config, str(net), only_log=True)    # print network

        if self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(net.parameters(), lr=self.Config.LEARNING_RATE)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE)
            # self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE, weight_decay=self.Config.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")

        if self.Config.LR_SCHEDULE:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
            # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max")

        if self.Config.LOAD_WEIGHTS:
            exp_utils.print_verbose(self.Config, "Loading weights ... ({})".format(join(self.Config.EXP_PATH, self.Config.WEIGHTS_PATH)))
            self.load_model(join(self.Config.EXP_PATH, self.Config.WEIGHTS_PATH))

        if self.Config.RESET_LAST_LAYER:
            self.net.conv_5 = nn.Conv2d(self.Config.UNET_NR_FILT, self.Config.NR_OF_CLASSES, kernel_size=1, stride=1, padding=0, bias=True).to(self.device)


    def train(self, X, y, weight_factor=10):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # (bs, features, x, y)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)  # (bs, classes, x, y)

        self.optimizer.zero_grad()
        self.net.train()
        outputs, outputs_sigmoid = self.net(X)  # forward; outputs: (bs, classes, x, y)

        if weight_factor > 1:
            weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES, y.shape[2], y.shape[3])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  # 10
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(outputs_sigmoid, y)
                # loss = criterion(outputs_sigmoid, y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)

        loss.backward()  # calc gradients
        self.optimizer.step()

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > 0.5, outputs.detach(), per_class=True)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), outputs_sigmoid.detach(), per_class=True, threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            # probs = outputs_sigmoid.detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = outputs_sigmoid
        else:
            probs = None    #faster

        return loss.item(), probs, f1


    def test(self, X, y, weight_factor=10):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs, outputs_sigmoid = self.net(X)  # forward

        if weight_factor > 1:
            weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES, y.shape[2], y.shape[3])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(outputs_sigmoid, y)
                # loss = criterion(outputs_sigmoid, y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR, max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > 0.5, outputs.detach(), per_class=True)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), outputs_sigmoid.detach(), per_class=True, threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            # probs = outputs_sigmoid.detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = outputs_sigmoid
        else:
            probs = None  # faster

        return loss.item(), probs, f1


    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs, outputs_sigmoid = self.net(X)  # forward
        if self.Config.EXPERIMENT_TYPE == "peak_regression" or self.Config.EXPERIMENT_TYPE == "dm_regression":
            probs = outputs.detach().cpu().numpy().transpose(0,2,3,1)  # (bs, x, y, classes)
        else:
            probs = outputs_sigmoid.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (bs, x, y, classes)
        return probs


    def save_models(self, metrics, epoch_nr):
        max_f1_idx = np.argmax(metrics["f1_macro_validate"])
        max_f1 = np.max(metrics["f1_macro_validate"])
        if epoch_nr == max_f1_idx and max_f1 > 0.01:  # saving to network drives takes 5s (to local only 0.5s) -> do not save so often
            print("  Saving weights...")
            for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep*")):  # remove weights from previous epochs
                os.remove(fl)
            try:
                #Actually is a pkl not a npz
                pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"), unet=self.net)
            except IOError:
                print("\nERROR: Could not save weights because of IO Error\n")
            self.Config.BEST_EPOCH = epoch_nr


    def load_model(self, path):
        pytorch_utils.load_checkpoint(path, unet=self.net)


    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            exp_utils.print_and_save(self.Config, "current learning rate: {}".format(param_group['lr']))

