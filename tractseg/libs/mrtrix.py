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
from os.path import join
import tempfile
import shutil
from pkg_resources import resource_filename
import subprocess
import nibabel as nib

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils
from tractseg.libs import tracking

def move_to_MNI_space(input_file, bvals, bvecs, brain_mask, output_dir):
    print("Moving input to MNI space...")

    os.system("calc_FA -i " + input_file + " -o " + output_dir + "/FA.nii.gz --bvals " + bvals +
              " --bvecs " + bvecs + " --brain_mask " + brain_mask)

    dwi_spacing = img_utils.get_image_spacing(input_file)

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

    brain_mask = create_brain_mask(new_input_file, output_dir)

    return new_input_file, bvals, bvecs, brain_mask


def move_to_subject_space(output_dir):
    print("Moving input to subject space...")

    file_path_in = output_dir + "/bundle_segmentations.nii.gz"
    file_path_out = output_dir + "/bundle_segmentations_subjectSpace.nii.gz"
    dwi_spacing = img_utils.get_image_spacing(file_path_in)
    os.system("convert_xfm -omat " + output_dir + "/MNI_2_FA.mat -inverse " + output_dir + "/FA_2_MNI.mat")
    os.system("flirt -ref " + output_dir + "/FA.nii.gz -in " + file_path_in + " -out " + file_path_out +
              " -applyisoxfm " + dwi_spacing + " -init " + output_dir + "/MNI_2_FA.mat -dof 6")
    os.system("fslmaths " + file_path_out + " -thr 0.5 -bin " + file_path_out)


def create_brain_mask(input_file, output_dir):
    print("Creating brain mask...")

    # os.system("export FSLDIR=/usr/local/fsl")
    # os.system("export PATH=$FSLDIR/bin:$PATH")
    os.system("export PATH=/usr/local/fsl/bin:$PATH")

    input_dir = os.path.dirname(input_file)
    input_file_without_ending = os.path.basename(input_file).split(".")[0]
    os.system("bet " + join(input_dir, input_file_without_ending) + " " +
              output_dir + "/nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m")
    os.system("rm " + output_dir + "/nodif_brain_mask.nii.gz")           #masked brain
    os.system("mv " + output_dir + "/nodif_brain_mask_mask.nii.gz " + output_dir + "/nodif_brain_mask.nii.gz")
    return join(output_dir, "nodif_brain_mask.nii.gz")


def create_fods(input_file, output_dir, bvals, bvecs, brain_mask, csd_type, nr_cpus=-1):
    os.system("export PATH=/code/mrtrix3/bin:$PATH")

    if nr_cpus > 0:
        nthreads = " -nthreads " + str(nr_cpus)
    else:
        nthreads = ""

    if csd_type == "csd_msmt_5tt":
        # MSMT 5TT
        print("Creating peaks (1 of 4)...")
        t1_file = join(os.path.dirname(input_file), "T1w_acpc_dc_restore_brain.nii.gz")
        os.system("5ttgen fsl " + t1_file + " " + output_dir + "/5TT.nii.gz -premasked" + nthreads)
        print("Creating peaks (2 of 4)...")
        os.system("dwi2response msmt_5tt " + input_file + " " + output_dir + "/5TT.nii.gz " + output_dir +
                  "/RF_WM.txt " + output_dir + "/RF_GM.txt " + output_dir + "/RF_CSF.txt -voxels " + output_dir +
                  "/RF_voxels.nii.gz -fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask + nthreads)
        print("Creating peaks (3 of 4)...")
        os.system("dwi2fod msmt_csd " + input_file + " " + output_dir + "/RF_WM.txt " + output_dir +
                  "/WM_FODs.nii.gz " + output_dir + "/RF_GM.txt " + output_dir + "/GM.nii.gz " + output_dir +
                  "/RF_CSF.txt " + output_dir + "/CSF.nii.gz -mask " + brain_mask +
                  " -fslgrad " + bvecs + " " + bvals + nthreads)       # multi-shell, multi-tissue
        print("Creating peaks (4 of 4)...")
        os.system("sh2peaks " + output_dir + "/WM_FODs.nii.gz " + output_dir + "/peaks.nii.gz -quiet" + nthreads)
    elif csd_type == "csd_msmt":
        # MSMT DHollander    (only works with msmt_csd, not with csd)
        # dhollander does not need a T1 image to estimate the response function)
        print("Creating peaks (1 of 3)...")
        os.system("dwi2response dhollander -mask " + brain_mask + " " + input_file + " " + output_dir + "/RF_WM.txt " +
                  output_dir + "/RF_GM.txt " + output_dir + "/RF_CSF.txt -fslgrad " + bvecs + " " + bvals +
                  " -mask " + brain_mask + nthreads)
        print("Creating peaks (2 of 3)...")
        os.system("dwi2fod msmt_csd " + input_file + " " +
                  output_dir + "/RF_WM.txt " + output_dir + "/WM_FODs.nii.gz " +
                  output_dir + "/RF_GM.txt " + output_dir + "/GM_FODs.nii.gz " +
                  output_dir + "/RF_CSF.txt " + output_dir + "/CSF_FODs.nii.gz " +
                  "-fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask + nthreads)
        print("Creating peaks (3 of 3)...")
        os.system("sh2peaks " + output_dir + "/WM_FODs.nii.gz " + output_dir + "/peaks.nii.gz -quiet" + nthreads)
    elif csd_type == "csd":
        # CSD Tournier
        print("Creating peaks (1 of 3)...")
        os.system("dwi2response tournier " + input_file + " " + output_dir + "/response.txt -mask " + brain_mask +
                  " -fslgrad " + bvecs + " " + bvals + " -quiet" + nthreads)
        print("Creating peaks (2 of 3)...")
        os.system("dwi2fod csd " + input_file + " " + output_dir + "/response.txt " + output_dir +
                  "/WM_FODs.nii.gz -mask " + brain_mask + " -fslgrad " + bvecs + " " + bvals + " -quiet" + nthreads)
        print("Creating peaks (3 of 3)...")
        os.system("sh2peaks " + output_dir + "/WM_FODs.nii.gz " + output_dir + "/peaks.nii.gz -quiet" + nthreads)
    else:
        raise ValueError("'csd_type' contains invalid String")


def track(bundle, peaks, output_dir, filter_by_endpoints=True, output_format="trk", nr_fibers=2000, nr_cpus=-1,
          peak_prob_tracking=True, tracking_on_FODs="False", tracking_folder="auto", dilation=1,
          use_best_original_peaks=False, dir_postfix="", use_as_prior=False, TOM_mrtrix_algorithm="FACT"):
    """

    Args:
        bundle: bundle name
        peaks: path to original peaks
        output_dir: output directory
        filter_by_endpoints: use results of endings_segmentation to filter out all fibers not endings in those regions
        output_format:
        nr_fibers:
        nr_cpus:
        peak_prob_tracking: If doing filter_by_endpoint, use own probabilistic peak tracking instead of mtrix tracking
        tracking_on_FODs: Runs iFOD2 tracking on original FODs (have to be provided to -i without setting
            --raw_diffusion_input) instead of running FACT tracking on TOMs.
            options: False | FACT | iFOD2
        tracking_folder:
        dilation:
        use_best_original_peaks:
        dir_postfix:
        use_as_prior:
        TOM_mrtrix_algorithm:

    Returns:
        Void
    """

    def mrtrix_tck_to_trk():
        ref_img = nib.load(output_dir + "/bundle_segmentations"  + dir_postfix + "/" + bundle + ".nii.gz")
        reference_affine = ref_img.get_affine()
        reference_shape = ref_img.get_data().shape[:3]
        fiber_utils.convert_tck_to_trk(output_dir + "/" + tracking_folder + "/" + bundle + ".tck",
                                       output_dir + "/" + tracking_folder + "/" + bundle + ".trk",
                                       reference_affine, reference_shape, compress_err_thr=0.1, smooth=None,
                                       nr_cpus=nr_cpus, tracking_format=output_format)
        subprocess.call("rm -f " + output_dir + "/" + tracking_folder + "/" + bundle + ".tck", shell=True)

    def get_tracking_folder_name():
        if tracking_on_FODs == "FACT":
            tracking_folder = "Peaks_FACT_trackings"
        elif tracking_on_FODs == "SD_STREAM":
            tracking_folder = "FOD_SD_STREAM_trackings"
        elif tracking_on_FODs == "iFOD2":
            tracking_folder = "FOD_iFOD2_trackings"
        elif use_best_original_peaks:
            tracking_folder = "BestOrig_trackings"
        else:
            tracking_folder = "TOM_trackings"
        return tracking_folder


    ################### Preparing ###################

    # Auto set tracking folder name
    if tracking_folder == "auto":
        tracking_folder = get_tracking_folder_name()
    TOM_folder = "TOM" + dir_postfix

    # Set nr threads for MRtrix
    if nr_cpus > 0:
        nthreads = " -nthreads " + str(nr_cpus)
    else:
        nthreads = ""

    # Misc
    subprocess.call("export PATH=/code/mrtrix3/bin:$PATH", shell=True)
    subprocess.call("mkdir -p " + output_dir + "/" + tracking_folder, shell=True)
    tmp_dir = tempfile.mkdtemp()
    if tracking_on_FODs != "False":
        peak_prob_tracking = False

    # Check if bundle masks are valid
    if filter_by_endpoints:
        bundle_mask_ok = nib.load(output_dir + "/bundle_segmentations" + dir_postfix
                                  + "/" + bundle + ".nii.gz").get_data().max() > 0
        beginnings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz").get_data().max() > 0
        endings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz").get_data().max() > 0

        if not bundle_mask_ok:
            print("WARNING: tract mask of {} empty. Falling back to "
                  "tracking without filtering by endpoints.".format(bundle))

        if not beginnings_mask_ok:
            print("WARNING: tract beginnings mask of {} empty. Falling "
                  "back to tracking without filtering by endpoints.".format(bundle))

        if not endings_mask_ok:
            print("WARNING: tract endings mask of {} empty. Falling back "
                  "to tracking without filtering by endpoints.".format(bundle))


    ################### Tracking ###################

    # No filtering
    if not filter_by_endpoints:
        img_utils.peak_image_to_binary_mask_path(peaks, tmp_dir + "/peak_mask.nii.gz", peak_length_threshold=0.01)

        # FACT Tracking on TOMs
        subprocess.call("tckgen -algorithm FACT " +
                        output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                        output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                        " -seed_image " + tmp_dir + "/peak_mask.nii.gz" +
                        " -minlength 40 -select " + str(nr_fibers) + " -force -quiet" + nthreads, shell=True)

        if output_format == "trk" or output_format == "trk_legacy":
            mrtrix_tck_to_trk()

    # Filtering
    if filter_by_endpoints and bundle_mask_ok and beginnings_mask_ok and endings_mask_ok:

        # Mrtrix Tracking
        if tracking_on_FODs != "False" or not peak_prob_tracking:

            # Prepare files
            img_utils.dilate_binary_mask(output_dir + "/bundle_segmentations" + dir_postfix + "/" + bundle + ".nii.gz",
                                         tmp_dir + "/" + bundle + ".nii.gz", dilation=dilation)
            img_utils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz",
                                         tmp_dir + "/" + bundle + "_e.nii.gz", dilation=dilation + 1)
            img_utils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz",
                                         tmp_dir + "/" + bundle + "_b.nii.gz", dilation=dilation + 1)

            # Mrtrix tracking on original FODs (have to be provided to -i)
            if tracking_on_FODs != "False":
                algorithm = tracking_on_FODs
                if algorithm == "FACT" or algorithm == "SD_STREAM":
                    seeds = 1000000
                else:
                    seeds = 200000
                # Quite slow
                subprocess.call("tckgen -algorithm " + algorithm + " " +
                                peaks + " " +
                                output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                " -minlength 40 -seeds " + str(seeds) + " -select " +
                                str(nr_fibers) + " -force" + nthreads,
                                shell=True)
                if output_format == "trk" or output_format == "trk_legacy":
                    mrtrix_tck_to_trk()

            # FACT tracking on TOMs
            elif tracking_on_FODs == "False" and not peak_prob_tracking and TOM_mrtrix_algorithm == "FACT":
                # Takes around 2.5min for 1 subject (2mm resolution)
                subprocess.call("tckgen -algorithm FACT " +
                                output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                                output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                " -minlength 40 -select " + str(nr_fibers) + " -force -quiet" + nthreads,
                                shell=True)
                if output_format == "trk" or output_format == "trk_legacy":
                    mrtrix_tck_to_trk()

            # iFOD2 tracking on TOMs
            elif tracking_on_FODs == "False" and not peak_prob_tracking and TOM_mrtrix_algorithm == "iFOD2":
                # Takes around 12min for 1 subject (2mm resolution)
                img_utils.peaks2fixel(output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz", tmp_dir + "/fixel")
                subprocess.call("fixel2sh " + tmp_dir + "/fixel/amplitudes.nii.gz " +
                                tmp_dir + "/fixel/sh.nii.gz -quiet", shell=True)
                subprocess.call("tckgen -algorithm iFOD2 " +
                                tmp_dir + "/fixel/sh.nii.gz " +
                                output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                " -minlength 40 -select " + str(nr_fibers) + " -force -quiet" + nthreads,
                                shell=True)
                if output_format == "trk" or output_format == "trk_legacy":
                    mrtrix_tck_to_trk()


        # TractSeg probabilistic tracking
        elif tracking_on_FODs == "False" and peak_prob_tracking:

            # Prepare files
            bundle_mask = nib.load(output_dir + "/bundle_segmentations" + dir_postfix + "/"
                                   + bundle + ".nii.gz").get_data()
            beginnings = nib.load(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz").get_data()
            endings = nib.load(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz").get_data()
            seed_img = nib.load(output_dir + "/bundle_segmentations" + dir_postfix + "/" +
                                bundle + ".nii.gz")
            tom_peaks = nib.load(output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz").get_data()

            #Get best original peaks
            if use_best_original_peaks:
                orig_peaks = nib.load(peaks)
                best_orig_peaks = fiber_utils.get_best_original_peaks(tom_peaks, orig_peaks.get_data())
                nib.save(nib.Nifti1Image(best_orig_peaks, orig_peaks.get_affine()),
                         output_dir + "/" + tracking_folder + "/" + bundle + ".nii.gz")
                tom_peaks = best_orig_peaks

            #Get weighted mean between best original peaks and TOMs
            if use_as_prior:
                orig_peaks = nib.load(peaks)
                best_orig_peaks = fiber_utils.get_best_original_peaks(tom_peaks, orig_peaks.get_data())
                weighted_peaks = fiber_utils.get_weighted_mean_of_peaks(best_orig_peaks, tom_peaks, weight=0.5)
                nib.save(nib.Nifti1Image(weighted_peaks, orig_peaks.get_affine()),
                         output_dir + "/" + tracking_folder + "/" + bundle + "_weighted.nii.gz")
                tom_peaks = weighted_peaks

            # Takes around 6min for 1 subject (2mm resolution)
            streamlines = tracking.track(tom_peaks, seed_img, max_nr_fibers=nr_fibers, smooth=10, compress=0.1,
                                         bundle_mask=bundle_mask, start_mask=beginnings, end_mask=endings,
                                         dilation=dilation, nr_cpus=nr_cpus, verbose=False)

            if output_format == "trk_legacy":
                fiber_utils.save_streamlines_as_trk_legacy(output_dir + "/" + tracking_folder + "/" + bundle + ".trk",
                                                           streamlines, seed_img.get_affine(),
                                                           seed_img.get_data().shape)
            else:  # tck or trk (determined by file ending)
                fiber_utils.save_streamlines(
                    output_dir + "/" + tracking_folder + "/" + bundle + "." + output_format,
                    streamlines, seed_img.get_affine(),
                    seed_img.get_data().shape)


    shutil.rmtree(tmp_dir)


def clean_up(Config, preprocessing_done=False):
    if not Config.KEEP_INTERMEDIATE_FILES:
        os.chdir(Config.PREDICT_IMG_OUTPUT)

        # os.system("rm -f nodif_brain_mask.nii.gz")
        # os.system("rm -f peaks.nii.gz")
        os.system("rm -f WM_FODs.nii.gz")

        if Config.CSD_TYPE == "csd_msmt" or Config.CSD_TYPE == "csd_msmt_5tt":
            os.system("rm -f 5TT.nii.gz")
            os.system("rm -f RF_WM.txt")
            os.system("rm -f RF_GM.txt")
            os.system("rm -f RF_CSF.txt")
            os.system("rm -f RF_voxels.nii.gz")
            os.system("rm -f CSF.nii.gz")
            os.system("rm -f GM.nii.gz")
            os.system("rm -f CSF_FODs.nii.gz")
            os.system("rm -f GM_FODs.nii.gz")
        else:
            os.system("rm -f response.txt")

    if preprocessing_done:
        os.system("rm -f FA.nii.gz")
        os.system("rm -f FA_MNI.nii.gz")
        os.system("rm -f FA_2_MNI.mat")