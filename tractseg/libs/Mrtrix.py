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
from tractseg.libs.FiberUtils import FiberUtils
import nibabel as nib
import tempfile
from tractseg.libs.ImgUtils import ImgUtils
import shutil
from pkg_resources import resource_filename
import numpy as np

class Mrtrix():

    @staticmethod
    def move_to_MNI_space(input_file, bvals, bvecs, brain_mask, output_dir):
        print("Moving input to MNI space...")

        os.system("calc_FA -i " + input_file + " -o " + output_dir + "/FA.nii.gz --bvals " + bvals +
                  " --bvecs " + bvecs + " --brain_mask " + brain_mask)

        dwi_spacing = ImgUtils.get_image_spacing(input_file)

        template_path = resource_filename('resources', 'MNI_FA_template.nii.gz')

        os.system("flirt -ref " + template_path + " -in " + output_dir + "/FA.nii.gz -out " + output_dir +
                  "/FA_MNI.nii.gz -omat " + output_dir + "/FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo")

        os.system("flirt -ref " + template_path + " -in " + input_file + " -out " + output_dir +
                  "/Diffusion_MNI.nii.gz -applyisoxfm " + dwi_spacing + " -init " + output_dir + "/FA_2_MNI.mat -dof 6")
        os.system("cp " + bvals + " " + output_dir + "/Diffusion_MNI.bvals")
        os.system("cp " + bvecs + " " + output_dir + "/Diffusion_MNI.bvecs")

        new_input_file = join(output_dir, "Diffusion_MNI.nii.gz")
        bvecs = join(output_dir, "Diffusion_MNI.bvecs")
        bvals = join(output_dir, "Diffusion_MNI.bvals")

        brain_mask = Mrtrix.create_brain_mask(new_input_file, output_dir)

        return new_input_file, bvals, bvecs, brain_mask

    @staticmethod
    def move_to_subject_space(output_dir):
        print("Moving input to subject space...")

        file_path_in = output_dir + "/bundle_segmentations.nii.gz"
        file_path_out = output_dir + "/bundle_segmentations_subjectSpace.nii.gz"
        dwi_spacing = ImgUtils.get_image_spacing(file_path_in)
        os.system("convert_xfm -omat " + output_dir + "/MNI_2_FA.mat -inverse " + output_dir + "/FA_2_MNI.mat")
        os.system("flirt -ref " + output_dir + "/FA.nii.gz -in " + file_path_in + " -out " + file_path_out +
                  " -applyisoxfm " + dwi_spacing + " -init " + output_dir + "/MNI_2_FA.mat -dof 6")
        os.system("fslmaths " + file_path_out + " -thr 0.5 -bin " + file_path_out)


    @staticmethod
    def create_brain_mask(input_file, output_dir):
        print("Creating brain mask...")

        # os.system("export FSLDIR=/usr/local/fsl")
        # os.system("export PATH=$FSLDIR/bin:$PATH")
        os.system("export PATH=/usr/local/fsl/bin:$PATH")

        input_dir = os.path.dirname(input_file)
        input_file_without_ending = os.path.basename(input_file).split(".")[0]
        os.system("bet " + join(input_dir, input_file_without_ending) + " " + output_dir + "/nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m")
        os.system("rm " + output_dir + "/nodif_brain_mask.nii.gz")           #masked brain
        os.system("mv " + output_dir + "/nodif_brain_mask_mask.nii.gz " + output_dir + "/nodif_brain_mask.nii.gz")
        return join(output_dir, "nodif_brain_mask.nii.gz")

    @staticmethod
    def create_fods(input_file, output_dir, bvals, bvecs, brain_mask, csd_type):
        os.system("export PATH=/code/mrtrix3/bin:$PATH")

        if csd_type == "csd_msmt_5tt":
            # MSMT 5TT
            print("Creating peaks (1 of 4)...")
            t1_file = join(os.path.dirname(input_file), "T1w_acpc_dc_restore_brain.nii.gz")
            os.system("5ttgen fsl " + t1_file + " " + output_dir + "/5TT.mif -premasked")
            print("Creating peaks (2 of 4)...")
            os.system("dwi2response msmt_5tt " + input_file + " " + output_dir + "/5TT.mif " + output_dir + "/RF_WM.txt " +
                      output_dir + "/RF_GM.txt " + output_dir + "/RF_CSF.txt -voxels " + output_dir + "/RF_voxels.mif -fslgrad " +
                      bvecs + " " + bvals + " -mask " + brain_mask)         # multi-shell, multi-tissue
            print("Creating peaks (3 of 4)...")
            os.system("dwi2fod msmt_csd " + input_file + " " + output_dir + "/RF_WM.txt " + output_dir +
                      "/WM_FODs.mif " + output_dir + "/RF_GM.txt " + output_dir + "/GM.mif " + output_dir +
                      "/RF_CSF.txt " + output_dir + "/CSF.mif -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals)       # multi-shell, multi-tissue
            print("Creating peaks (4 of 4)...")
            os.system("sh2peaks " + output_dir + "/WM_FODs.mif " + output_dir + "/peaks.nii.gz -quiet")
        elif csd_type == "csd_msmt":
            # MSMT DHollander    (only works with msmt_csd, not with csd)
            # dhollander does not need a T1 image to estimate the response function (more recent (2016) than tournier (2013))
            print("Creating peaks (1 of 3)...")
            os.system("dwi2response dhollander -mask " + brain_mask + " " + input_file + " " + output_dir + "/RF_WM.txt " +
                      output_dir + "/RF_GM.txt " + output_dir + "/RF_CSF.txt -fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask)
            print("Creating peaks (2 of 3)...")
            os.system("dwi2fod msmt_csd " + input_file + " " + output_dir + "/RF_WM.txt " + output_dir +
                      "/WM_FODs.mif -fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask + "")
            print("Creating peaks (3 of 3)...")
            os.system("sh2peaks " + output_dir + "/WM_FODs.mif " + output_dir + "/peaks.nii.gz -quiet")
        elif csd_type == "csd":
            # CSD Tournier
            print("Creating peaks (1 of 3)...")
            os.system("dwi2response tournier " + input_file + " " + output_dir + "/response.txt -mask " + brain_mask +
                      " -fslgrad " + bvecs + " " + bvals + " -quiet")
            print("Creating peaks (2 of 3)...")
            os.system("dwi2fod csd " + input_file + " " + output_dir + "/response.txt " + output_dir +
                      "/WM_FODs.mif -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals + " -quiet")
            print("Creating peaks (3 of 3)...")
            os.system("sh2peaks " + output_dir + "/WM_FODs.mif " + output_dir + "/peaks.nii.gz -quiet")
        else:
            raise ValueError("'csd_type' contains invalid String")

    @staticmethod
    def track(bundle, peaks, output_dir, filter_by_endpoints=False, output_format="trk"):
        '''

        :param bundle:   Bundle name
        :param peaks:
        :param output_dir:
        :param filter_by_endpoints:     use results of endings_segmentation to filter out all fibers not endings in those regions
        :return:
        '''
        tracking_folder = "TOM_trackings"
        # tracking_folder = "TOM_trackings_FiltEP6_FiltMask3"
        smooth = None       # None / 10
        TOM_folder = "TOM"      # TOM / TOM_thr1

        tmp_dir = tempfile.mkdtemp()
        os.system("export PATH=/code/mrtrix3/bin:$PATH")
        os.system("mkdir -p " + output_dir + "/" + tracking_folder)

        if filter_by_endpoints:
            bundle_mask_ok = nib.load(output_dir + "/bundle_segmentations/" + bundle + ".nii.gz").get_data().max() > 0
            beginnings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz").get_data().max() > 0
            endings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz").get_data().max() > 0

            if not bundle_mask_ok:
                print("WARNING: tract mask of {} empty. Falling back to tracking without filtering by endpoints.".format(bundle))

            if not beginnings_mask_ok:
                print("WARNING: tract beginnings mask of {} empty. Falling back to tracking without filtering by endpoints.".format(bundle))

            if not endings_mask_ok:
                print("WARNING: tract endings mask of {} empty. Falling back to tracking without filtering by endpoints.".format(bundle))

        if filter_by_endpoints and bundle_mask_ok and beginnings_mask_ok and endings_mask_ok:
            # dilation has to be quite high, because endings sometimes almost completely missing
            ImgUtils.dilate_binary_mask(output_dir + "/bundle_segmentations/" + bundle + ".nii.gz",
                                        tmp_dir + "/" + bundle + ".nii.gz", dilation=3)
            ImgUtils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz",
                                        tmp_dir + "/" + bundle + "_e.nii.gz", dilation=6)
            ImgUtils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz",
                                        tmp_dir + "/" + bundle + "_b.nii.gz", dilation=6)

            os.system("tckgen -algorithm FACT " +
                      output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                      output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                      " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                      " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                      " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                      " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                      " -minlength 40 -select 2000 -force -quiet")

            # #Probabilistic Tracking without TOM
            # os.system("tckgen -algorithm iFOD2 " +
            #           peaks + " " +
            #           output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
            #           " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
            #           " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
            #           " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
            #           " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
            #           " -minlength 40 -seeds 200000 -select 2000 -force")
        else:
            ImgUtils.peak_image_to_binary_mask_path(peaks, tmp_dir + "/peak_mask.nii.gz", peak_length_threshold=0.01)
            os.system("tckgen -algorithm FACT " +
                      output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                      output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                      " -seed_image " + tmp_dir + "/peak_mask.nii.gz" +
                      " -minlength 40 -select 2000 -force -quiet")

        if output_format == "trk":
            ref_img = nib.load(peaks)
            reference_affine = ref_img.get_affine()
            reference_shape = ref_img.get_data().shape[:3]
            FiberUtils.convert_tck_to_trk(output_dir + "/" + tracking_folder + "/" + bundle + ".tck",
                                          output_dir + "/" + tracking_folder + "/" + bundle + ".trk",
                                          reference_affine, reference_shape, compress_err_thr=0.1, smooth=smooth)
            os.system("rm -f " + output_dir + "/" + tracking_folder + "/" + bundle + ".tck")
        shutil.rmtree(tmp_dir)

    @staticmethod
    def clean_up(HP):
        if not HP.KEEP_INTERMEDIATE_FILES:
            os.chdir(HP.PREDICT_IMG_OUTPUT)

            # os.system("rm -f nodif_brain_mask.nii.gz")
            # os.system("rm -f peaks.nii.gz")
            os.system("rm -f WM_FODs.mif")

            if HP.CSD_TYPE == "csd_msmt" or HP.CSD_TYPE == "csd_msmt_5tt":
                os.system("rm -f 5TT.mif")
                os.system("rm -f RF_WM.txt")
                os.system("rm -f RF_GM.txt")
                os.system("rm -f RF_CSF.txt")
                os.system("rm -f RF_voxels.mif")
                os.system("rm -f CSF.mif")
                os.system("rm -f GM.mif")
            else:
                os.system("rm -f response.txt")
