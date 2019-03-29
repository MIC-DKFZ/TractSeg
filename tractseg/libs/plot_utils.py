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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import math
import numpy as np

from tractseg.libs import exp_utils

import matplotlib
matplotlib.use('Agg') #Solves error with ssh and plotting

#https://www.quora.com/If-a-Python-program-already-has-numerous-matplotlib-plot-functions-what-is-the-quickest-way-to-convert-them-all-to-a-way-where-all-the-plots-can-be-produced-as-hard-images-with-minimal-modification-of-code
import matplotlib.pyplot as plt

#Might fix problems with matplotlib over ssh (failing after connection is open for longer)
#   (http://stackoverflow.com/questions/2443702/problem-running-python-matplotlib-in-background-after-ending-ssh-session)
plt.ioff()


def plot_mask(renderer, mask_data, affine, x_current, y_current,
              orientation="axial", smoothing=10, brain_mask=None):
    from tractseg.libs import vtk_utils

    if brain_mask is not None:
        brain_mask = brain_mask.transpose(0, 2, 1)
        brain_mask = brain_mask[::-1, :, :]
        if orientation == "sagittal":
            brain_mask = brain_mask.transpose(2, 1, 0)
            brain_mask = brain_mask[::-1, :, :]
        cont_actor = vtk_utils.contour_from_roi_smooth(brain_mask, affine=affine,
                                                       color=[.9, .9, .9], opacity=.1, smoothing=30)
        cont_actor.SetPosition(x_current, y_current, 0)
        renderer.add(cont_actor)

    # 3D Bundle
    mask = mask_data
    mask = mask.transpose(0, 2, 1)
    mask = mask[::-1, :, :]
    if orientation == "sagittal":
        mask = mask.transpose(2, 1, 0)
        mask = mask[::-1, :, :]
    color = [1, .27, .18]  # red

    cont_actor = vtk_utils.contour_from_roi_smooth(mask, affine=affine,
                                                   color=color, opacity=1, smoothing=smoothing)
    cont_actor.SetPosition(x_current, y_current, 0)
    renderer.add(cont_actor)


def plot_tracts(classes, bundle_segmentations, affine, out_dir, brain_mask=None):
    '''
    By default this does not work on a remote server connection (ssh -X) because -X does not support OpenGL.
    On the remote Server you can do 'export DISPLAY=":0"' .
    (you should set the value you get if you do 'echo $DISPLAY' if you
    login locally on the remote server). Then all graphics will get rendered locally and not via -X.
    (important: graphical session needs to be running on remote server (e.g. via login locally))
    (important: login needed, not just stay at login screen)

    If running on a headless server without Display using Xvfb might help:
    https://stackoverflow.com/questions/6281998/can-i-run-glu-opengl-on-a-headless-server
    '''
    from dipy.viz import window
    from tractseg.libs import vtk_utils

    SMOOTHING = 10
    WINDOW_SIZE = (800, 800)
    bundles = ["CST_right", "CA", "IFO_right"]

    renderer = window.Renderer()
    renderer.projection('parallel')

    rows = len(bundles)
    X, Y, Z = bundle_segmentations.shape[:3]
    for j, bundle in enumerate(bundles):
        i = 0  #only one method

        bundle_idx = exp_utils.get_bundle_names(classes)[1:].index(bundle)
        mask_data = bundle_segmentations[:,:,:,bundle_idx]

        if bundle == "CST_right":
            orientation = "axial"
        elif bundle == "CA":
            orientation = "axial"
        elif bundle == "IFO_right":
            orientation = "sagittal"
        else:
            orientation = "axial"

        #bigger: more border
        if orientation == "axial":
            border_y = -100  #-60
        else:
            border_y = -100

        x_current = X * i  # column (width)
        y_current = rows * (Y * 2 + border_y) - (Y * 2 + border_y) * j  # row (height)  (starts from bottom?)

        plot_mask(renderer, mask_data, affine, x_current, y_current,
                            orientation=orientation, smoothing=SMOOTHING, brain_mask=brain_mask)

        #Bundle label
        text_offset_top = -50  # 60
        text_offset_side = -100 # -30
        position = (0 - int(X) + text_offset_side, y_current + text_offset_top, 50)
        text_actor = vtk_utils.label(text=bundle, pos=position, scale=(6, 6, 6), color=(1, 1, 1))
        renderer.add(text_actor)

    renderer.reset_camera()
    window.record(renderer, out_path=join(out_dir, "preview.png"),
                  size=(WINDOW_SIZE[0], WINDOW_SIZE[1]), reset_camera=False, magnification=2)


def plot_tracts_matplotlib(classes, bundle_segmentations, background_img, out_dir):

    def plot_single_tract(bg, data, orientation, bundle):
        if orientation == "coronal":
            data = data.transpose(2,0,1)[::-1,:,:]
            bg = bg.transpose(2,0,1)[::-1,:,:]
        elif orientation == "sagittal":
            data = data.transpose(2,1,0)[::-1,:,:]
            bg = bg.transpose(2,1,0)[::-1,:,:]
        else:  # axial
            pass

        mask_voxel_coords = np.where(data != 0)
        minidx = int(np.min(mask_voxel_coords[2]))
        maxidx = int(np.max(mask_voxel_coords[2])) + 1
        mean_slice = int(np.mean([minidx, maxidx]))
        bg = bg[:, :, mean_slice]
        # bg = matplotlib.colors.Normalize()(bg)

        # project 3D to 2D image
        if aggregation == "mean":
            data = data.mean(axis=2)
        else:
            data = data.max(axis=2)

        plt.imshow(bg, cmap="gray")
        data = np.ma.masked_where(data < 0.0001, data)
        plt.imshow(data, cmap="autumn")
        plt.title(bundle, fontsize=7)

    if classes.startswith("AutoPTX"):
        bundles = ["cst_r", "cst_s_r", "ifo_r", "fx_l", "fx_r", "or_l", "fma"]
    else:
        bundles = ["CST_right", "CST_s_right", "CA", "IFO_right", "FX_left", "FX_right", "OR_left", "CC_1"]

    aggregation = "max"
    cols = 4
    rows = math.ceil(len(bundles) / cols)

    background_img = background_img[...,0]

    for j, bundle in enumerate(bundles):
        bun = bundle.lower()
        if bun.startswith("ca") or bun.startswith("fx_") or bun.startswith("or_") or \
                bun.startswith("cc_1") or bun.startswith("fma"):
            orientation = "axial"
        elif bun.startswith("ifo_") or bun.startswith("icp_") or bun.startswith("cst_s_"):
            bundle = bundle.replace("_s", "")
            orientation = "sagittal"
        elif bun.startswith("cst_"):
            orientation = "coronal"
        else:
            raise ValueError("invalid bundle")

        bundle_idx = exp_utils.get_bundle_names(classes)[1:].index(bundle)
        mask_data = bundle_segmentations[:, :, :, bundle_idx]

        plt.subplot(rows, cols, j+1)
        plt.axis("off")
        plot_single_tract(background_img, mask_data, orientation, bundle)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(join(out_dir, "preview.png"), bbox_inches='tight', dpi=300)


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
        plt1, = ax.plot(list(range(5, len(metrics["loss_train"]))),
                        metrics["loss_train"][5:], "r:", label='loss train')
        plt2, = ax.plot(list(range(5, len(metrics["loss_validate"]))),
                        metrics["loss_validate"][5:], "r", label='loss val')
        plt3, = ax.plot(list(range(5, len(metrics["loss_test"]))),
                        metrics["loss_test"][5:], "r--", label='loss test')

        plt4, = ax2.plot(list(range(5, len(metrics["f1_macro_train"]))),
                         metrics["f1_macro_train"][5:], "g:", label='f1_macro_train')
        plt5, = ax2.plot(list(range(5, len(metrics["f1_macro_validate"]))),
                         metrics["f1_macro_validate"][5:], "g", label='f1_macro_val')
        plt6, = ax2.plot(list(range(5, len(metrics["f1_macro_test"]))),
                         metrics["f1_macro_test"][5:], "g--", label='f1_macro_test')

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
        plt8, = ax2.plot(metrics["f1_macro_validate"], "g", label='f1_macro_val')
        plt9, = ax2.plot(metrics["f1_macro_test"], "g--", label='f1_macro_test')

        # #tmp
        # plt10, = ax2.plot(metrics["f1_LenF1_train"], "b:", label='f1_LenF1_train')
        # plt11, = ax2.plot(metrics["f1_LenF1_validate"], "b", label='f1_LenF1_val')
        # plt12, = ax2.plot(metrics["f1_LenF1_test"], "b--", label='f1_LenF1_test')
        #
        # #tmp
        # plt13, = ax2.plot(metrics["f1_Thr2_train"], "m:", label='f1_Thr2_train')
        # plt14, = ax2.plot(metrics["f1_Thr2_validate"], "m", label='f1_Thr2_val')
        # plt15, = ax2.plot(metrics["f1_Thr2_test"], "m--", label='f1_Thr2_test')

        plt.legend(handles=[plt1, plt2, plt3, plt7, plt8, plt9],
        # plt.legend(handles=[plt1, plt2, plt3, plt7, plt8, plt9, plt10, plt11, plt12],
        # plt.legend(handles=[plt1, plt2, plt3, plt7, plt8, plt9, plt10, plt11, plt12, plt13, plt14, plt15],
                   loc=2,
                   borderaxespad=0.,
                   bbox_to_anchor=(1.03, 1))

        fig_name = "metrics_all.png"


    fig.text(0.12, 0.95, exp_name, size=12, weight="bold")
    fig.text(0.12, 0.02, description)
    fig.savefig(join(path, fig_name), dpi=100)
    plt.close()


def plot_result_trixi(trixi, x, y, probs, loss, f1, epoch_nr):
    import torch
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-7)  # for proper plotting
    trixi.show_image_grid(torch.tensor(x_norm).float()[:5, 0:1, :, :], name="input batch",
                          title="Input batch")  # all channels of one batch

    probs_shaped = probs[:, 15:16, :, :]  # (bs, 1, x, y)
    probs_shaped_bin = (probs_shaped > 0.5).int()
    trixi.show_image_grid(probs_shaped[:5], name="predictions", title="Predictions Probmap")
    # nvl.show_images(probs_shaped_bin[:5], name="predictions_binary", title="Predictions Binary")

    # Show GT and Prediction in one image  (bundle: CST); GREEN: GT; RED: prediction (FP); YELLOW: prediction (TP)
    combined = torch.zeros((y.shape[0], 3, y.shape[2], y.shape[3]))
    combined[:, 0:1, :, :] = probs_shaped_bin  # Red
    combined[:, 1:2, :, :] = torch.tensor(y)[:, 15:16, :, :]  # Green
    trixi.show_image_grid(combined[:5], name="predictions_combined", title="Combined")

    # #Show feature activations
    # contr_1_2 = intermediate[2].data.cpu().numpy()   # (bs, nr_feature_channels=64, x, y)
    # contr_1_2 = contr_1_2[0:1,:,:,:].transpose((1,0,2,3)) # (nr_feature_channels=64, 1, x, y)
    # contr_1_2 = (contr_1_2 - contr_1_2.min()) / (contr_1_2.max() - contr_1_2.min())
    # nvl.show_images(contr_1_2, name="contr_1_2", title="contr_1_2")
    #
    # # Show feature activations
    # contr_3_2 = intermediate[1].data.cpu().numpy()  # (bs, nr_feature_channels=64, x, y)
    # contr_3_2 = contr_3_2[0:1, :, :, :].transpose((1, 0, 2, 3))  # (nr_feature_channels=64, 1, x, y)
    # contr_3_2 = (contr_3_2 - contr_3_2.min()) / (contr_3_2.max() - contr_3_2.min())
    # nvl.show_images(contr_3_2, name="contr_3_2", title="contr_3_2")
    #
    # # Show feature activations
    # deconv_2 = intermediate[0].data.cpu().numpy()  # (bs, nr_feature_channels=64, x, y)
    # deconv_2 = deconv_2[0:1, :, :, :].transpose((1, 0, 2, 3))  # (nr_feature_channels=64, 1, x, y)
    # deconv_2 = (deconv_2 - deconv_2.min()) / (deconv_2.max() - deconv_2.min())
    # nvl.show_images(deconv_2, name="deconv_2", title="deconv_2")

    trixi.show_value(value=float(loss), counter=epoch_nr, name="loss", tag="loss")
    trixi.show_value(value=float(np.mean(f1)), counter=epoch_nr, name="f1", tag="f1")