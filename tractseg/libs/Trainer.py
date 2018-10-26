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

from os.path import join
import time
import pickle
import numpy as np
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from tractseg.libs.DatasetUtils import DatasetUtils
from tractseg.libs import plot_utils
import socket
from tqdm import tqdm
import datetime

class Trainer:

    def __init__(self, model, dataManager):
        self.model = model
        self.dataManager = dataManager

    def train(self, HP):

        if HP.USE_VISLOGGER:
            try:
                from trixi.logger.visdom import PytorchVisdomLogger
            except ImportError:
                pass
            trixi = PytorchVisdomLogger(port=8080, auto_start=True)

        exp_utils.print_and_save(HP, socket.gethostname())

        epoch_times = []
        nr_of_updates = 0

        metrics = {}
        for type in ["train", "test", "validate"]:
            metrics_new = {
                "loss_" + type: [0],
                "f1_macro_" + type: [0],
            }
            metrics = dict(list(metrics.items()) + list(metrics_new.items()))

        for epoch_nr in range(HP.NUM_EPOCHS):
            start_time = time.time()
            # current_lr = HP.LEARNING_RATE * (HP.LR_DECAY ** epoch_nr)
            # current_lr = HP.LEARNING_RATE

            batch_gen_time = 0
            data_preparation_time = 0
            network_time = 0
            metrics_time = 0
            saving_time = 0
            plotting_time = 0

            batch_nr = {
                "train": 0,
                "test": 0,
                "validate": 0
            }

            if HP.LOSS_WEIGHT_LEN == -1:
                weight_factor = float(HP.LOSS_WEIGHT)
            else:
                if epoch_nr < HP.LOSS_WEIGHT_LEN:
                    # weight_factor = -(9./100.) * epoch_nr + 10.   #ep0: 10 -> linear decrease -> ep100: 1
                    weight_factor = -((HP.LOSS_WEIGHT-1)/float(HP.LOSS_WEIGHT_LEN)) * epoch_nr + float(HP.LOSS_WEIGHT)
                    # weight_factor = -((HP.LOSS_WEIGHT-5)/float(HP.LOSS_WEIGHT_LEN)) * epoch_nr + float(HP.LOSS_WEIGHT)
                else:
                    weight_factor = 1.
                    # weight_factor = 5.

            for type in ["train", "test", "validate"]:
                print_loss = []
                start_time_batch_gen = time.time()

                batch_generator = self.dataManager.get_batches(batch_size=HP.BATCH_SIZE,
                                                               type=type, subjects=getattr(HP, type.upper() + "_SUBJECTS"))
                batch_gen_time = time.time() - start_time_batch_gen
                # print("batch_gen_time: {}s".format(batch_gen_time))

                nr_of_samples = len(getattr(HP, type.upper() + "_SUBJECTS")) * HP.INPUT_DIM[0]
                nr_batches = int(nr_of_samples / HP.BATCH_SIZE)

                print("Start looping batches...")
                start_time_batch_part = time.time()
                for i in range(nr_batches):
                    batch = next(batch_generator)

                    start_time_data_preparation = time.time()
                    batch_nr[type] += 1

                    x = batch["data"] # (bs, nr_of_channels, x, y)
                    y = batch["seg"]  # (bs, nr_of_classes, x, y)
                    # since using new BatchGenerator y is not int anymore but float -> would be good for Pytorch but not Lasagne
                    # y = y.astype(HP.LABELS_TYPE)  #for bundle_peaks regression: is already float -> saves 0.2s/batch if left out

                    data_preparation_time += time.time() - start_time_data_preparation
                    # self.model.learning_rate.set_value(np.float32(current_lr))
                    start_time_network = time.time()
                    if type == "train":
                        nr_of_updates += 1
                        loss, probs, f1 = self.model.train(x, y, weight_factor=weight_factor)    # probs: # (bs, x, y, nrClasses)
                        # loss, probs, f1, intermediate = self.model.train(x, y)
                    elif type == "validate":
                        loss, probs, f1 = self.model.predict(x, y, weight_factor=weight_factor)
                    elif type == "test":
                        loss, probs, f1 = self.model.predict(x, y, weight_factor=weight_factor)
                    network_time += time.time() - start_time_network

                    start_time_metrics = time.time()

                    if HP.CALC_F1:
                        if HP.EXPERIMENT_TYPE == "peak_regression":
                            #Following two lines increase metrics_time by 30s (without < 1s); time per batch increases by 1.5s by these lines
                            # y_flat = y.transpose(0, 2, 3, 1)  # (bs, x, y, nr_of_classes)
                            # y_flat = np.reshape(y_flat, (-1, y_flat.shape[-1]))  # (bs*x*y, nr_of_classes)
                            # metrics = MetricUtils.calculate_metrics(metrics, y_flat, probs, loss, f1=np.mean(f1), type=type, threshold=HP.THRESHOLD,
                            #                                         f1_per_bundle={"CA": f1[5], "FX_left": f1[23], "FX_right": f1[24]})

                            #Numpy
                            # y_right_order = y.transpose(0, 2, 3, 1)  # (bs, x, y, nr_of_classes)
                            # peak_f1 = MetricUtils.calc_peak_dice(HP, probs, y_right_order)
                            # peak_f1_mean = np.array([s for s in peak_f1.values()]).mean()

                            #Pytorch
                            peak_f1_mean = np.array([s for s in list(f1.values())]).mean()  #if f1 for multiple bundles
                            metrics = metric_utils.calculate_metrics(metrics, None, None, loss, f1=peak_f1_mean, type=type, threshold=HP.THRESHOLD)

                            #Pytorch 2 F1
                            # peak_f1_mean_a = np.array([s for s in f1[0].values()]).mean()
                            # peak_f1_mean_b = np.array([s for s in f1[1].values()]).mean()
                            # metrics = MetricUtils.calculate_metrics(metrics, None, None, loss, f1=peak_f1_mean_a, type=type, threshold=HP.THRESHOLD,
                            #                                         f1_per_bundle={"LenF1": peak_f1_mean_b})

                            #Single Bundle
                            # metrics = MetricUtils.calculate_metrics(metrics, None, None, loss, f1=f1["CST_right"][0], type=type, threshold=HP.THRESHOLD,
                            #                                         f1_per_bundle={"Thr1": f1["CST_right"][1], "Thr2": f1["CST_right"][2]})
                            # metrics = MetricUtils.calculate_metrics(metrics, None, None, loss, f1=f1["CST_right"], type=type, threshold=HP.THRESHOLD)
                        else:
                            metrics = metric_utils.calculate_metrics(metrics, None, None, loss, f1=np.mean(f1), type=type, threshold=HP.THRESHOLD)

                    else:
                        metrics = metric_utils.calculate_metrics_onlyLoss(metrics, loss, type=type)

                    metrics_time += time.time() - start_time_metrics

                    print_loss.append(loss)
                    if batch_nr[type] % HP.PRINT_FREQ == 0:
                        time_batch_part = time.time() - start_time_batch_part
                        start_time_batch_part = time.time()
                        exp_utils.print_and_save(HP, "{} Ep {}, Sp {}, loss {}, t print {}s, t batch {}s".format(type, epoch_nr,
                                                                                                                 batch_nr[type] * HP.BATCH_SIZE,
                                                                                                                 round(np.array(print_loss).mean(), 6), round(time_batch_part, 3),
                                                                                                                 round(time_batch_part / HP.PRINT_FREQ, 3)))
                        print_loss = []

                    if HP.USE_VISLOGGER:
                        plot_utils.plot_result_trixi(trixi, x, y, probs, loss, f1, epoch_nr)


            ###################################
            # Post Training tasks (each epoch)
            ###################################

            #Adapt LR
            if HP.LR_SCHEDULE:
                self.model.scheduler.step()
                # self.model.scheduler.step(np.mean(f1))
                self.model.print_current_lr()

            # Average loss per batch over entire epoch
            metrics = metric_utils.normalize_last_element(metrics, batch_nr["train"], type="train")
            metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")
            metrics = metric_utils.normalize_last_element(metrics, batch_nr["test"], type="test")

            print("  Epoch {}, Average Epoch loss = {}".format(epoch_nr, metrics["loss_train"][-1]))
            print("  Epoch {}, nr_of_updates {}".format(epoch_nr, nr_of_updates))

            # Save Weights
            start_time_saving = time.time()
            if HP.SAVE_WEIGHTS:
                self.model.save_model(metrics, epoch_nr)
            saving_time += time.time() - start_time_saving

            # Create Plots
            start_time_plotting = time.time()
            pickle.dump(metrics, open(join(HP.EXP_PATH, "metrics.pkl"), "wb")) # wb -> write (override) and binary (binary only needed on windows, on unix also works without) # for loading: pickle.load(open("metrics.pkl", "rb"))
            plot_utils.create_exp_plot(metrics, HP.EXP_PATH, HP.EXP_NAME)
            plot_utils.create_exp_plot(metrics, HP.EXP_PATH, HP.EXP_NAME, without_first_epochs=True)
            plotting_time += time.time() - start_time_plotting

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            exp_utils.print_and_save(HP, "  Epoch {}, time total {}s".format(epoch_nr, epoch_time))
            exp_utils.print_and_save(HP, "  Epoch {}, time UNet: {}s".format(epoch_nr, network_time))
            exp_utils.print_and_save(HP, "  Epoch {}, time metrics: {}s".format(epoch_nr, metrics_time))
            exp_utils.print_and_save(HP, "  Epoch {}, time saving files: {}s".format(epoch_nr, saving_time))
            exp_utils.print_and_save(HP, str(datetime.datetime.now()))

            # Adding next Epoch
            if epoch_nr < HP.NUM_EPOCHS-1:
                metrics = metric_utils.add_empty_element(metrics)


        ####################################
        # After all epochs
        ###################################
        with open(join(HP.EXP_PATH, "Hyperparameters.txt"), "a") as f:  # a for append
            f.write("\n\n")
            f.write("Average Epoch time: {}s".format(sum(epoch_times) / float(len(epoch_times))))

        return metrics


    def get_seg_single_img(self, HP, probs=False, scale_to_world_shape=True, only_prediction=False):
        '''
        Returns layers for one image (batch manager is only allowed to return batches for one image)

        :param HP:
        :return: ([146, 174, 146, nrClasses], [146, 174, 146, nrClasses])    (Prediction, Groundtruth)
        '''

        def finalize_data(layers):
            layers = np.array(layers)

            # Get in right order (x,y,z) and
            if HP.SLICE_DIRECTION == "x":
                layers = layers.transpose(0, 1, 2, 3)

            elif HP.SLICE_DIRECTION == "y":
                layers = layers.transpose(1, 0, 2, 3)

            elif HP.SLICE_DIRECTION == "z":
                layers = layers.transpose(1, 2, 0, 3)

            if scale_to_world_shape:
                layers = DatasetUtils.scale_input_to_world_shape(layers, HP.DATASET, HP.RESOLUTION)

            return layers.astype(np.float32)


        #Test Time DAug
        for i in range(1):
            # segs = []
            # ys = []

            layers_seg = []
            layers_y = []
            batch_generator = self.dataManager.get_batches(batch_size=1)
            batch_generator = list(batch_generator)
            for batch in tqdm(batch_generator):
                x = batch["data"]   # (bs, nr_of_channels, x, y)
                y = batch["seg"]    # (bs, x, y, nr_of_classes)
                y = y.astype(HP.LABELS_TYPE)
                y = np.squeeze(y)   # remove bs dimension which is only 1 -> (x, y, nrClasses)

                if HP.DROPOUT_SAMPLING:
                    #For Dropout Sampling (must set Deterministic=False in model)
                    NR_SAMPLING = 30
                    samples = []
                    for i in range(NR_SAMPLING):
                        layer_probs = self.model.get_probs(x)  # (bs, x, y, nrClasses)
                        samples.append(layer_probs)

                    samples = np.array(samples)  # (NR_SAMPLING, bs, x, y, nrClasses)
                    samples = np.squeeze(samples) # (NR_SAMPLING, x, y, nrClasses)
                    # layer_probs = np.mean(samples, axis=0)
                    layer_probs = np.std(samples, axis=0)    # (x,y,nrClasses)
                else:
                    # For normal prediction
                    layer_probs = self.model.get_probs(x)  # (bs, x, y, nrClasses)
                    layer_probs = np.squeeze(layer_probs)  # remove bs dimension which is only 1 -> (x, y, nrClasses)

                if probs:
                    seg = layer_probs   # (x, y, nrClasses)
                else:
                    seg = layer_probs
                    seg[seg >= HP.THRESHOLD] = 1
                    seg[seg < HP.THRESHOLD] = 0
                    seg = seg.astype(np.int16)

                layers_seg.append(seg)
                if not only_prediction:
                    layers_y.append(y)

        layers_seg = finalize_data(layers_seg)
        if not only_prediction:
            layers_y = finalize_data(layers_y)

        return layers_seg, layers_y   # (Prediction, Groundtruth)
