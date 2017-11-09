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
from os.path import join

class Mrtrix():

    @staticmethod
    def create_brain_mask(input_file, output_dir):
        print("Creating brain mask...")
        input_dir = os.path.dirname(input_file)
        input_file_without_ending = os.path.basename(input_file).split(".")[0]
        os.chdir(output_dir)
        # os.system("pwd")
        os.system('bet ' + join(input_dir, input_file_without_ending) + ' nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m')
        os.system('rm nodif_brain_mask.nii.gz')           #masked brain
        os.system('mv nodif_brain_mask_mask.nii.gz nodif_brain_mask.nii.gz')

    @staticmethod
    def create_fods(input_file, output_dir, bvals, bvecs, brain_mask, csd_resolution):
        os.chdir(output_dir)

        if csd_resolution == "HIGH":
            # MSMT 5TT
            t1_file = join(os.path.dirname(input_file), "T1w_acpc_dc_restore_brain.nii.gz") # todo: Add default T1 name to Doku
            os.system("5ttgen fsl " + t1_file + " 5TT.mif -premasked")
            os.system("dwi2response msmt_5tt " + input_file + " 5TT.mif RF_WM.txt RF_GM.txt RF_CSF.txt -voxels RF_voxels.mif -fslgrad " + bvecs + " " + bvals)         # multi-shell, multi-tissue
            os.system("dwi2fod msmt_csd " + input_file + " RF_WM.txt WM_FODs.mif RF_GM.txt GM.mif RF_CSF.txt CSF.mif -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals)       # multi-shell, multi-tissue

            # MSMT DHollander    (only works with msmt_csd, not with csd)
            ##dhollander does not need a T1 image to estimate the response function (more recent (2016) than tournier (2013))
            # os.system("dwi2response dhollander -mask " + brain_mask + " " + input_file + " RF_WM_DHol.txt RF_GM_DHol.txt RF_CSF_DHol.txt -fslgrad " + bvecs + " " + bvals)
            ##dwi2fod csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs_csd.mif RF_GM_DHol.txt GM_FODs_csd.mif RF_CSF_DHol.txt CSF_FODs_csd.mif -mask " + brain_mask + " -fslgrad Diffusion.bvecs Diffusion.bvals
            # os.system("dwi2fod msmt_csd " + input_file + " RF_WM_DHol.txt WM_FODs.mif -fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask + "")

            os.system("sh2peaks WM_FODs.mif peaks.nii.gz")
        else:   #LOW
            # CSD Tournier
            print("Creating peaks (1 of 3)...")
            os.system("dwi2response tournier " + input_file + " response.txt -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals + " -quiet")
            print("Creating peaks (2 of 3)...")
            os.system("dwi2fod csd " + input_file + " response.txt WM_FODs.mif -shell 1000 -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals + " -quiet")
            print("Creating peaks (3 of 3)...")
            os.system("sh2peaks WM_FODs.mif peaks.nii.gz -quiet")

    @staticmethod
    def clean_up(HP):
        if not HP.KEEP_INTERMEDIATE_FILES:
            os.chdir(HP.PREDICT_IMG_OUTPUT)

            os.system("rm -f nodif_brain_mask.nii.gz")
            os.system("rm -f WM_FODs.mif")
            os.system("rm -f peaks.nii.gz")

            if HP.CSD_RESOLUTION == "HIGH":
                os.system("rm -f 5TT.mif")
                os.system("rm -f RF_WM.txt")
                os.system("rm -f RF_GM.txt")
                os.system("rm -f RF_CSF.txt")
                os.system("rm -f RF_voxels.mif")
                os.system("rm -f CSF.mif")
                os.system("rm -f GM.mif")
            else:
                os.system("rm -f response.txt")
