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
import torch.nn.functional as F
from apex import amp

from tractseg.libs import pytorch_utils
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils

class BaseModel:
    def __init__(self, Config, inference=False):
        self.Config = Config

        if not inference:
            torch.backends.cudnn.benchmark = True

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
            if self.Config.LOSS_FUNCTION == "angle_length_loss":
                self.criterion = pytorch_utils.angle_length_loss
            elif self.Config.LOSS_FUNCTION == "angle_loss":
                self.criterion = pytorch_utils.angle_loss
            elif self.Config.LOSS_FUNCTION == "l2_loss":
                self.criterion = pytorch_utils.l2_loss
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            # self.criterion = nn.MSELoss()   # aggregate by mean
            self.criterion = nn.MSELoss(size_average=False, reduce=True)   # aggregate by sum
        else:
            # weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
            #                       self.Config.INPUT_DIM[0], self.Config.INPUT_DIM[1])).cuda()
            # weights[:, 5, :, :] *= 10     #CA
            # weights[:, 21, :, :] *= 10    #FX_left
            # weights[:, 22, :, :] *= 10    #FX_right
            # self.criterion = nn.BCEWithLogitsLoss(weight=weights)
            self.criterion = nn.BCEWithLogitsLoss()

        NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                               self.Config.MODEL)
        self.net = NetworkClass(n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES,
                                n_filt=self.Config.UNET_NR_FILT, batchnorm=self.Config.BATCH_NORM,
                                dropout=self.Config.USE_DROPOUT, upsample=self.Config.UPSAMPLE_TYPE)

        # Somehow not really faster (max 10% speedup): GPU utility low -> why? (CPU also low)
        # (with bigger batch_size even worse)
        # - GPU slow connection? (but maybe same problem as before pin_memory)
        # - Wrong setup with pin_memory, async, ...? -> should be correct
        # - load from npy instead of nii -> will not solve entire problem
        # nr_gpus = torch.cuda.device_count()
        # exp_utils.print_and_save(self.Config, "nr of gpus: {}".format(nr_gpus))
        # self.net = nn.DataParallel(self.net)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net.to(self.device)

        # if self.Config.TRAIN:
        #     exp_utils.print_and_save(self.Config, str(net), only_log=True)    # print network

        if self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(net.parameters(), lr=self.Config.LEARNING_RATE)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE)
            # self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE,
            #                       weight_decay=self.Config.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")

        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")

        if self.Config.LR_SCHEDULE:
            # Slightly better results could be archived if training for 500ep without reduction of LR
            # -> but takes too long -> using reudceOnPlateau gives benefits if only training for 200ep
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode=self.Config.LR_SCHEDULE_MODE,
                                                            patience=self.Config.LR_SCHEDULE_PATIENCE)

        if self.Config.LOAD_WEIGHTS:
            exp_utils.print_verbose(self.Config, "Loading weights ... ({})".format(join(self.Config.EXP_PATH,
                                                                                        self.Config.WEIGHTS_PATH)))
            self.load_model(join(self.Config.EXP_PATH, self.Config.WEIGHTS_PATH))

        if self.Config.RESET_LAST_LAYER:
            self.net.conv_5 = nn.Conv2d(self.Config.UNET_NR_FILT, self.Config.NR_OF_CLASSES, kernel_size=1,
                                        stride=1, padding=0, bias=True).to(self.device)


    def train(self, X, y, weight_factor=None):
        X = X.contiguous().cuda(non_blocking=True)  # (bs, features, x, y)
        y = y.contiguous().cuda(non_blocking=True)  # (bs, classes, x, y)

        self.optimizer.zero_grad()
        self.net.train()
        outputs = self.net(X)  # forward; outputs: (bs, classes, x, y)
        angle_err = None

        if weight_factor is not None:
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  # 10

            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
                # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            # probs = F.sigmoid(outputs).detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = F.sigmoid(outputs)
        else:
            probs = None    #faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def test(self, X, y, weight_factor=None):
        """

        Args:
            X: float torch tensor
            y: float torch tensor
            weight_factor:

        Returns:

        """
        with torch.no_grad():
            X = X.contiguous().cuda(non_blocking=True)
            y = y.contiguous().cuda(non_blocking=True)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs = self.net(X)  # forward
        angle_err = None

        if weight_factor is not None:
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
                # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            # probs = F.sigmoid(outputs).detach().cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).contiguous().to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs = self.net(X)  # forward
        if self.Config.EXPERIMENT_TYPE == "peak_regression" or self.Config.EXPERIMENT_TYPE == "dm_regression":
            probs = outputs.detach().cpu().numpy()
        else:
            probs = F.sigmoid(outputs).detach().cpu().numpy()

        if self.Config.DIM == "2D":
            probs = probs.transpose(0, 2, 3, 1)  # (bs, x, y, classes)
        else:
            probs = probs.transpose(0, 2, 3, 4, 1)  # (bs, x, y, z, classes)
        return probs


    def save_model(self, metrics, epoch_nr, mode="f1"):
        if mode == "f1":
            max_f1_idx = np.argmax(metrics["f1_macro_validate"])
            max_f1 = np.max(metrics["f1_macro_validate"])
            do_save = epoch_nr == max_f1_idx and max_f1 > 0.01
        else:
            min_loss_idx = np.argmin(metrics["loss_validate"])
            # min_loss = np.min(metrics["loss_validate"])
            do_save = epoch_nr == min_loss_idx

        # saving to network drives takes 5s (to local only 0.5s) -> do not save so often
        if do_save:
            print("  Saving weights...")
            for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep*")):  # remove weights from previous epochs
                os.remove(fl)
            try:
                #Actually is a pkl not a npz
                pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                                              unet=self.net)
            except IOError:
                print("\nERROR: Could not save weights because of IO Error\n")
            self.Config.BEST_EPOCH = epoch_nr


    def load_model(self, path):
        pytorch_utils.load_checkpoint(path, unet=self.net)


    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            exp_utils.print_and_save(self.Config, "current learning rate: {}".format(param_group['lr']))

