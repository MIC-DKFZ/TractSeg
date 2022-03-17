
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
from pkg_resources import resource_filename
from tqdm import tqdm

from tractseg.libs import img_utils


def reorient_to_std_space(input_file, bvals, bvecs, brain_mask, output_dir):
    print("Reorienting input to MNI space...")

    # Only working with FSL 6
    # os.system("fslreorient2std -m " + output_dir + "/reorient2std.mat " + input_file +
    #           " " + output_dir + "/Diffusion_MNI.nii.gz")

    # Working with FSL 5 and 6
    os.system("fslreorient2std " + input_file + " > " + output_dir + "/reorient2std.mat")
    os.system("fslreorient2std " + input_file + " " + output_dir + "/Diffusion_MNI.nii.gz")

    os.system("cp " + bvals + " " + output_dir + "/Diffusion_MNI.bvals")
    os.system("rotate_bvecs -i " + bvecs + " -t " + output_dir + "/reorient2std.mat -o " +
              output_dir + "/Diffusion_MNI.bvecs")

    os.system("flirt -ref " + output_dir + "/Diffusion_MNI.nii.gz -in " + brain_mask +
              " -out " + output_dir + "/nodif_brain_mask_MNI.nii.gz -applyxfm -init " +
              output_dir + "/reorient2std.mat -dof 6")

    new_input_file = join(output_dir, "Diffusion_MNI.nii.gz")
    bvecs = join(output_dir, "Diffusion_MNI.bvecs")
    bvals = join(output_dir, "Diffusion_MNI.bvals")
    brain_mask = join(output_dir, "nodif_brain_mask_MNI.nii.gz")

    return new_input_file, bvals, bvecs, brain_mask


def move_to_MNI_space(input_file, bvals, bvecs, brain_mask, output_dir):
    print("Moving input to MNI space...")

    os.system("calc_FA -i " + input_file + " -o " + output_dir + "/FA.nii.gz --bvals " + bvals +
              " --bvecs " + bvecs + " --brain_mask " + brain_mask)

    dwi_spacing = img_utils.get_image_spacing(input_file)

    template_path = resource_filename('tractseg.resources', 'MNI_FA_template.nii.gz')

    os.system("flirt -ref " + template_path + " -in " + output_dir + "/FA.nii.gz -out " + output_dir +
              "/FA_MNI.nii.gz -omat " + output_dir + "/FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo")

    os.system("flirt -ref " + template_path + " -in " + input_file + " -out " + output_dir +
              "/Diffusion_MNI.nii.gz -applyisoxfm " + dwi_spacing + " -init " + output_dir +
              "/FA_2_MNI.mat -dof 6 -interp trilinear")
    os.system("cp " + bvals + " " + output_dir + "/Diffusion_MNI.bvals")
    os.system("rotate_bvecs -i " + bvecs + " -t " + output_dir + "/FA_2_MNI.mat" +
              " -o " + output_dir + "/Diffusion_MNI.bvecs")

    os.system("flirt -ref " + template_path + " -in " + brain_mask +
              " -out " + output_dir + "/nodif_brain_mask_MNI.nii.gz -applyisoxfm " + dwi_spacing + " -init " +
              output_dir + "/FA_2_MNI.mat -dof 6 -interp nearestneighbour")

    new_input_file = join(output_dir, "Diffusion_MNI.nii.gz")
    bvecs = join(output_dir, "Diffusion_MNI.bvecs")
    bvals = join(output_dir, "Diffusion_MNI.bvals")
    brain_mask = join(output_dir, "nodif_brain_mask_MNI.nii.gz")

    return new_input_file, bvals, bvecs, brain_mask


def move_to_subject_space_single_file(output_dir, experiment_type, output_subdir, output_float=False):
    print("Moving output to subject space...")

    os.system("mv " + output_dir + "/" + output_subdir + ".nii.gz " + output_dir + "/" + output_subdir + "_MNI.nii.gz")

    file_path_in = output_dir + "/" + output_subdir + "_MNI.nii.gz"
    file_path_out = output_dir + "/" + output_subdir + ".nii.gz"
    dwi_spacing = img_utils.get_image_spacing(file_path_in)
    os.system("convert_xfm -omat " + output_dir + "/MNI_2_FA.mat -inverse " + output_dir + "/FA_2_MNI.mat")
    os.system("flirt -ref " + output_dir + "/FA.nii.gz -in " + file_path_in + " -out " + file_path_out +
              " -applyisoxfm " + dwi_spacing + " -init " + output_dir + "/MNI_2_FA.mat -dof 6" +
              " -interp trilinear")
    if not output_float:
        os.system("fslmaths " + file_path_out + " -thr 0.5 -bin " + file_path_out)


def move_to_subject_space(output_dir, bundles, experiment_type, output_subdir, output_float=False):
    print("Moving output to subject space...")

    os.system("mkdir -p " + output_dir + "/" + output_subdir + "_MNI")
    os.system("mv " + output_dir + "/" + output_subdir + "/* " + output_dir + "/" + output_subdir + "_MNI")
    os.system("convert_xfm -omat " + output_dir + "/MNI_2_FA.mat -inverse " + output_dir + "/FA_2_MNI.mat")

    for bundle in tqdm(bundles):
        file_path_in = output_dir + "/" + output_subdir + "_MNI/" + bundle + ".nii.gz"
        file_path_out = output_dir + "/" + output_subdir + "/" + bundle + ".nii.gz"
        dwi_spacing = img_utils.get_image_spacing(file_path_in)
        if experiment_type == "peak_regression":
            os.system("flip_peaks -i " + file_path_in + " -o " + file_path_in[:-7] + "_flip.nii.gz -a x")  # flip to fsl format
            os.system("vecreg -i " + file_path_in[:-7] + "_flip.nii.gz -o " + file_path_out +
                      " -r " + output_dir + "/FA.nii.gz -t " + output_dir + "/MNI_2_FA.mat")  # Use vecreg to transform peaks
            os.system("flip_peaks -i " + file_path_out + " -o " + file_path_out + " -a x")  # flip back to mrtrix format
            os.system("rm " + file_path_in[:-7] + "_flip.nii.gz")  # remove flipped tmp file
        else:
            # do not use spline interpolation because makes a lot of holes into masks
            os.system("flirt -ref " + output_dir + "/FA.nii.gz -in " + file_path_in + " -out " + file_path_out +
                      " -applyisoxfm " + dwi_spacing + " -init " + output_dir + "/MNI_2_FA.mat -dof 6" +
                      " -interp trilinear")
        if not output_float:
            os.system("fslmaths " + file_path_out + " -thr 0.5 -bin " + file_path_out)


def create_brain_mask(input_file, output_dir):
    print("Creating brain mask...")

    input_dir = os.path.dirname(input_file)
    input_file_without_ending = os.path.basename(input_file).split(".")[0]
    os.system("bet " + join(input_dir, input_file_without_ending) + " " +
              output_dir + "/nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m")
    os.system("rm " + output_dir + "/nodif_brain_mask.nii.gz")  # masked brain
    os.system("mv " + output_dir + "/nodif_brain_mask_mask.nii.gz " + output_dir + "/nodif_brain_mask.nii.gz")
    # For newer fsl versions bet will create 4D brainmask. Causes error in mrtrix. Only keep the first 3D volume.
    os.system("fslroi " + output_dir + "/nodif_brain_mask.nii.gz " + output_dir + "/nodif_brain_mask.nii.gz " + "0 " + "1" )
    return join(output_dir, "nodif_brain_mask.nii.gz")


def create_fods(input_file, output_dir, bvals, bvecs, brain_mask, csd_type, nr_cpus=-1):

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
                  " -fslgrad " + bvecs + " " + bvals + nthreads)  # multi-shell, multi-tissue
        print("Creating peaks (4 of 4)...")
        os.system("sh2peaks " + output_dir + "/WM_FODs.nii.gz " + output_dir + "/peaks.nii.gz -quiet" + nthreads)
    elif csd_type == "csd_msmt":
        # MSMT DHollander    (only works with msmt_csd, not with csd)
        # (Dhollander does not need a T1 image to estimate the response function)
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


def clean_up(keep_intermediate_files, predict_img_output, csd_type, preprocessing_done=False):
    if not keep_intermediate_files:
        os.chdir(predict_img_output)

        # os.system("rm -f nodif_brain_mask.nii.gz")
        # os.system("rm -f peaks.nii.gz")
        os.system("rm -f WM_FODs.nii.gz")

        if csd_type == "csd_msmt" or csd_type == "csd_msmt_5tt":
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
