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

import os
import glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.models.BaseModel import BaseModel
from tractseg.libs.MetricUtils import MetricUtils

# nonlinearity = nn.ReLU()
nonlinearity = nn.LeakyReLU()

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer


class UNet(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64):
        super(UNet, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes

        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        # self.deconv_1 = nn.Upsample(scale_factor=2)     #does only upscale width and height  #Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)  # no activation function, because is in LossFunction (...WithLogits)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        pool_4 = self.pool_4(contr_4_2)

        pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        deconv_2 = self.deconv_2(expand_1_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)
        return conv_5


class UNet_Pytorch_Regression(BaseModel):
    def create_network(self):
        # torch.backends.cudnn.benchmark = True     #not faster

        def train(X, y, weight_factor=10):
            X = torch.from_numpy(X.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))
            if torch.cuda.is_available():
                X, y = Variable(X.cuda()), Variable(y.cuda())  # X: (bs, features, x, y)   y: (bs, classes, x, y)
            else:
                X, y = Variable(X), Variable(y)
            optimizer.zero_grad()
            net.train()
            outputs = net(X)  # forward     # outputs: (bs, classes, x, y)

            weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, self.HP.INPUT_DIM[0], self.HP.INPUT_DIM[1])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  #10

            loss = criterion(outputs, y, Variable(weights))
            # loss = criterion1(outputs, y, Variable(weights)) + criterion2(outputs, y, Variable(weights))

            loss.backward()  # backward
            optimizer.step()  # optimise

            if self.HP.CALC_F1:
                # f1 = PytorchUtils.f1_score_macro(y.data, outputs.data, per_class=True)
                # f1_a = MetricUtils.calc_peak_dice_pytorch(self.HP, outputs.data, y.data, max_angle_error=self.HP.PEAK_DICE_THR)
                f1 = MetricUtils.calc_peak_length_dice_pytorch(self.HP, outputs.data, y.data,
                                                               max_angle_error=self.HP.PEAK_DICE_THR, max_length_error=self.HP.PEAK_DICE_LEN_THR)
                # f1 = (f1_a, f1_b)
            else:
                f1 = np.ones(outputs.shape[3])

            if self.HP.USE_VISLOGGER:
                probs = outputs.data.cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            else:
                # probs = outputs.data.cpu().numpy().transpose(0,2,3,1)  # (bs, x, y, classes)
                probs = None    #faster

            return loss.data[0], probs, f1

        def test(X, y, weight_factor=10):
            X = torch.from_numpy(X.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))
            if torch.cuda.is_available():
                X, y = Variable(X.cuda(), volatile=True), Variable(y.cuda(), volatile=True)
            else:
                X, y = Variable(X, volatile=True), Variable(y, volatile=True)
            net.train(False)
            outputs = net(X)  # forward

            weights = torch.ones((self.HP.BATCH_SIZE, self.HP.NR_OF_CLASSES, self.HP.INPUT_DIM[0], self.HP.INPUT_DIM[1])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  #10

            loss = criterion(outputs, y, Variable(weights))
            # loss = criterion1(outputs, y, Variable(weights)) + criterion2(outputs, y, Variable(weights))

            if self.HP.CALC_F1:
                # f1 = PytorchUtils.f1_score_macro(y.data, outputs.data, per_class=True)
                # f1_a = MetricUtils.calc_peak_dice_pytorch(self.HP, outputs.data, y.data, max_angle_error=self.HP.PEAK_DICE_THR)
                f1 = MetricUtils.calc_peak_length_dice_pytorch(self.HP, outputs.data, y.data,
                                                               max_angle_error=self.HP.PEAK_DICE_THR, max_length_error=self.HP.PEAK_DICE_LEN_THR)
                # f1 = (f1_a, f1_b)
            else:
                f1 = np.ones(outputs.shape[3])

            # probs = outputs.data.cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
            probs = None  # faster
            return loss.data[0], probs, f1

        def predict(X):
            X = torch.from_numpy(X.astype(np.float32))
            if torch.cuda.is_available():
                X = Variable(X.cuda(), volatile=True)
            else:
                X = Variable(X, volatile=True)
            net.train(False)
            outputs = net(X)  # forward
            probs = outputs.data.cpu().numpy().transpose(0,2,3,1)   # (bs, x, y, classes)
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

            #Saving Last Epoch:
            # print("  Saving weights last epoch...")
            # for fl in glob.glob(join(self.HP.EXP_PATH, "weights_ep*")):  # remove weights from previous epochs
            #     os.remove(fl)
            # try:
            #     # Actually is a pkl not a npz
            #     PytorchUtils.save_checkpoint(join(self.HP.EXP_PATH, "weights_ep" + str(epoch_nr) + ".npz"), unet=net)
            # except IOError:
            #     print("\nERROR: Could not save weights because of IO Error\n")

            self.HP.BEST_EPOCH = epoch_nr


        def load_model(path):
            PytorchUtils.load_checkpoint(path, unet=net)

        def print_current_lr():
            for param_group in optimizer.param_groups:
                ExpUtils.print_and_save(self.HP, "current learning rate: {}".format(param_group['lr']))

        if self.HP.SEG_INPUT == "Peaks" and self.HP.TYPE == "single_direction":
            NR_OF_GRADIENTS = 9
            # NR_OF_GRADIENTS = 9 * 5
            # NR_OF_GRADIENTS = 9 * 9
            # NR_OF_GRADIENTS = 33
        elif self.HP.SEG_INPUT == "Peaks" and self.HP.TYPE == "combined":
            NR_OF_GRADIENTS = 3*self.HP.NR_OF_CLASSES
        else:
            NR_OF_GRADIENTS = 33

        if torch.cuda.is_available():
            net = UNet(n_input_channels=NR_OF_GRADIENTS, n_classes=self.HP.NR_OF_CLASSES, n_filt=self.HP.UNET_NR_FILT).cuda()
        else:
            net = UNet(n_input_channels=NR_OF_GRADIENTS, n_classes=self.HP.NR_OF_CLASSES, n_filt=self.HP.UNET_NR_FILT)

        # if self.HP.TRAIN:
        #     ExpUtils.print_and_save(self.HP, str(net), only_log=True)


        # criterion1 = PytorchUtils.MSE_weighted
        # criterion2 = PytorchUtils.angle_loss

        # criterion = PytorchUtils.MSE_weighted
        # criterion = PytorchUtils.angle_loss
        criterion = PytorchUtils.angle_length_loss


        optimizer = Adamax(net.parameters(), lr=self.HP.LEARNING_RATE)

        if self.HP.LOAD_WEIGHTS:
            ExpUtils.print_verbose(self.HP, "Loading weights ... ({})".format(join(self.HP.EXP_PATH, self.HP.WEIGHTS_PATH)))
            load_model(join(self.HP.EXP_PATH, self.HP.WEIGHTS_PATH))

        self.train = train
        self.predict = test
        self.get_probs = predict
        self.save_model = save_model
        self.load_model = load_model
        self.print_current_lr = print_current_lr
        # self.scheduler = scheduler