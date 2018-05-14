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

import os
from glob import glob
from os.path import join
from tractseg.libs.Config import Config as C
from tractseg.libs.ExpUtils import ExpUtils
import string
import random
import shutil

class ClusterUtils:
    
    def __init__(self):
        return None

    @staticmethod
    def copy_training_files_to_ssd(HP, data_path):

        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))

        target_data_path = join("/ssd/", "tmp_" + id_generator(), HP.DATASET_FOLDER)
        ExpUtils.make_dir(join(target_data_path))

        #get all folders in data_path directory
        subjects = [os.path.basename(os.path.normpath(d)) for d in glob(data_path + "/*/")]

        for subject in subjects:
            src = join(data_path, subject, HP.FEATURES_FILENAME)
            target = join(target_data_path, subject, HP.FEATURES_FILENAME)
            print("cp: {} -> {}".format(src, target))
            # shutil.copyfile(src, target)

        return target_data_path

    @staticmethod
    def remove_training_file_from_ssd(target_data_path):
        print("Removing tmp files on SSD: {}".format(target_data_path))
        shutil.rmtree(target_data_path)


