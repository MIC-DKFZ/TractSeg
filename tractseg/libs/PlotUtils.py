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
from tractseg.libs.ExpUtils import ExpUtils

class PlotUtils:

    @staticmethod
    def plot_mask(renderer, mask_data, affine, x_current, y_current,
                  orientation="axial", smoothing=10, brain_mask=None):
        from tractseg.libs.VtkUtils import VtkUtils

        if brain_mask is not None:
            brain_mask = brain_mask.transpose(0, 2, 1)
            brain_mask = brain_mask[::-1, :, :]
            if orientation == "sagittal":
                brain_mask = brain_mask.transpose(2, 1, 0)
                brain_mask = brain_mask[::-1, :, :]
            cont_actor = VtkUtils.contour_from_roi_smooth(brain_mask, affine=affine,
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

        cont_actor = VtkUtils.contour_from_roi_smooth(mask, affine=affine,
                                                      color=color, opacity=1, smoothing=smoothing)
        cont_actor.SetPosition(x_current, y_current, 0)
        renderer.add(cont_actor)


    @staticmethod
    def plot_tracts(HP, bundle_segmentations, affine, out_dir, brain_mask=None):
        '''
        By default this does not work on a remote server connection (ssh -X) because -X does not support OpenGL.
        On the remote Server you can do 'export DISPLAY=":0"' (you should set the value you get if you do 'echo $DISPLAY' if you
        login locally on the remote server). Then all graphics will get rendered locally and not via -X.
        (important: graphical session needs to be running on remote server (e.g. via login locally))
        '''
        from dipy.viz import window
        from tractseg.libs.VtkUtils import VtkUtils

        SMOOTHING = 10
        WINDOW_SIZE = (800, 800)
        bundles = ["CST_right", "CA", "IFO_right"]

        renderer = window.Renderer()
        renderer.projection('parallel')

        rows = len(bundles)
        X, Y, Z = bundle_segmentations.shape[:3]
        for j, bundle in enumerate(bundles):
            i = 0  #only one method

            bundle_idx = ExpUtils.get_bundle_names(HP.CLASSES)[1:].index(bundle)
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

            PlotUtils.plot_mask(renderer, mask_data, affine, x_current, y_current,
                                orientation=orientation, smoothing=SMOOTHING, brain_mask=brain_mask)

            #Bundle label
            text_offset_top = -50  # 60
            text_offset_side = -100 # -30
            position = (0 - int(X) + text_offset_side, y_current + text_offset_top, 50)
            text_actor = VtkUtils.label(text=bundle, pos=position, scale=(6, 6, 6), color=(1, 1, 1))
            renderer.add(text_actor)

        renderer.reset_camera()
        window.record(renderer, out_path=join(out_dir, "preview.png"),
                      size=(WINDOW_SIZE[0], WINDOW_SIZE[1]), reset_camera=False, magnification=2)
