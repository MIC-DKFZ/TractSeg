#!/bin/bash
set -e  # stop on error

#######################################################################################################################
#
# This is an example script for basic preprocessing of DWI images. It does the following steps:
#
# 1. Denoising
# 2. Remove Gibbs Ringing Artifacts
# 3. Remove Eddy Currents and head motion
# 4. Perform B1 field inhomogeneity correction
# 5. Create brain mask
# 6. Rigidly align to MNI space
# 7. Extract FODs and peaks
# 8. Run TractSeg
#
# This might not be the best preprocessing but it should be a pretty solid baseline.
#
# (If you run TractSeg with the options `--raw_diffusion_input --preprocess` then TractSeg will automatically
# perform steps 5 to 8 internally. You will still have to do steps 1 to 4 manually.)
#
# IMPORTANT: Look for all "todo" comments and adapt those parts according to your data.
#
#######################################################################################################################


function eddy_correct_extract_mask_denoise() {
    cd $data_path/$1
    cd preproc

    echo "Denoising..."
    dwidenoise Diffusion.nii.gz Diffusion_denoise.nii.gz -noise noise.nii.gz
    cp -v Diffusion.bvals Diffusion_denoise.bvals
    cp -v Diffusion.bvecs Diffusion_denoise.bvecs
    mrcalc Diffusion.nii.gz Diffusion_denoise.nii.gz -subtract noise_residual.nii.gz

    echo "Unringing..."
    mrdegibbs Diffusion_denoise.nii.gz Diffusion_denoise_unr.nii.gz -axes 0,1 -force
    cp -v Diffusion_denoise.bvals Diffusion_denoise_unr.bvals
    cp -v Diffusion_denoise.bvecs Diffusion_denoise_unr.bvecs

    echo "Eddy..."
    #todo: adapt -pe_dir parameter according to the way your data was acquired
    # (see https://mrtrix.readthedocs.io/en/3.0_rc1/reference/scripts/dwipreproc.html for details)
    dwipreproc Diffusion_denoise_unr.nii.gz Diffusion_denoise_unr_eddy.nii.gz -rpe_none -pe_dir j- \
    -fslgrad Diffusion_denoise_unr.bvecs Diffusion_denoise_unr.bvals \
    -eddyqc_text eddyqc
    cp -v Diffusion_denoise_unr.bvals Diffusion_denoise_unr_eddy.bvals
    cp -v Diffusion_denoise_unr.bvecs Diffusion_denoise_unr_eddy.bvecs

    echo "Bias correcting..."  # ants strongly recommended over fsl
    dwibiascorrect Diffusion_denoise_unr_eddy.nii.gz Diffusion_denoise_unr_eddy_bias.nii.gz -ants -bias bias_field.nii.gz \
    -fslgrad Diffusion_denoise_unr_eddy.bvecs Diffusion_denoise_unr_eddy.bvals -force
    cp -v Diffusion_denoise_unr_eddy.bvals Diffusion_denoise_unr_eddy_bias.bvals
    cp -v Diffusion_denoise_unr_eddy.bvecs Diffusion_denoise_unr_eddy_bias.bvecs

    echo "Brain masking..."
    #todo: if brain masks are too big or too small: adjust -f parameter (range: 0.1-0.5)
    bet Diffusion_denoise_unr_eddy_bias nodif_brain_mask.nii.gz -f 0.3 -g 0 -m
    rm nodif_brain_mask.nii.gz      #only b0 not all gradients
    mv nodif_brain_mask_mask.nii.gz nodif_brain_mask.nii.gz

    python ~/dev/bsp/scripts/misc/apply_brain_mask.py Diffusion_denoise_unr_eddy_bias.nii.gz \
    Diffusion_denoise_unr_eddy_bias_brain.nii.gz nodif_brain_mask.nii.gz
    cp -v Diffusion_denoise_unr_eddy_bias.bvals Diffusion_denoise_unr_eddy_bias_brain.bvals
    cp -v Diffusion_denoise_unr_eddy_bias.bvecs Diffusion_denoise_unr_eddy_bias_brain.bvecs
}

function  register_DWI_to_MNI() {
    echo "Registering to MNI..."
    cd $data_path/$1
    cd preproc

    calc_FA Diffusion_denoise_unr_eddy_bias_brain nodif_brain_mask.nii.gz  # calc_FA is part of TractSeg
    dwi_spacing=$(get_image_spacing Diffusion.nii.gz)  # get_image_spacing is part of TractSeg
    atlas=$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz

    #B0 2mm to MNI - mutualinfo working better
    flirt -ref $atlas -in FA.nii.gz \
    -out FA_MNI.nii.gz -omat FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo -interp spline

    #Register DWI to MNI
    flirt -ref $atlas \
    -in Diffusion_denoise_unr_eddy_bias_brain.nii.gz -out Diffusion_denoise_unr_eddy_bias_brain_MNI.nii.gz \
    -applyisoxfm "$dwi_spacing" -init FA_2_MNI.mat -dof 6 -interp spline
    cp -v Diffusion_denoise_unr_eddy_bias_brain.bvals Diffusion_denoise_unr_eddy_bias_brain_MNI.bvals
    rotate_bvecs Diffusion_denoise_unr_eddy_bias_brain.bvecs FA_2_MNI.mat \
    Diffusion_denoise_unr_eddy_bias_brain_MNI.bvecs

    #Transform brain mask to MNI with T1 2mm transform
    flirt -ref $atlas -in nodif_brain_mask.nii.gz \
    -out nodif_brain_mask_MNI.nii.gz -applyisoxfm "$dwi_spacing" -init FA_2_MNI.mat -dof 6
    fslmaths nodif_brain_mask_MNI.nii.gz -thr 0.5 -bin nodif_brain_mask_MNI.nii.gz

    #Remove negative values (introduced by spline interpolation)
    remove_negative_values Diffusion_denoise_unr_eddy_bias_brain_MNI.nii.gz \
    Diffusion_denoise_unr_eddy_bias_brain_MNI.nii.gz  # remove_negative_values is part of TractSeg

    #Get final files
    cp -v Diffusion_denoise_unr_eddy_bias_brain_MNI.bvals ../Diffusion.bvals
    cp -v Diffusion_denoise_unr_eddy_bias_brain_MNI.bvecs ../Diffusion.bvecs
    cp -v Diffusion_denoise_unr_eddy_bias_brain_MNI.nii.gz ../Diffusion.nii.gz
    cp -v nodif_brain_mask_MNI.nii.gz ../nodif_brain_mask.nii.gz

}

function create_fods() {
    echo "Creating FODs..."
    cd $data_path/$1

    #todo: select if you have single or multishell data
    shells="SINGLE"    # SINGLE / MULTI

    dwi2response dhollander Diffusion.nii.gz RF_WM_DHol.txt RF_GM_DHol.txt RF_CSF_DHol.txt \
    -fslgrad Diffusion.bvecs Diffusion.bvals -mask nodif_brain_mask.nii.gz

    if [ "$shells" = "MULTI" ]; then
        dwi2fod msmt_csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs.nii.gz \
        RF_GM_DHol.txt GM_FODs.nii.gz RF_CSF_DHol.txt CSF_FODs.nii.gz \
        -fslgrad Diffusion.bvecs Diffusion.bvals -mask nodif_brain_mask.nii.gz
    elif [ "$shells" = "SINGLE" ]; then
        dwi2fod msmt_csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs.nii.gz RF_CSF_DHol.txt CSF_FODs.nii.gz \
        -fslgrad Diffusion.bvecs Diffusion.bvals -mask nodif_brain_mask.nii.gz
    else
        echo "unrecognized option"
    fi

    sh2peaks WM_FODs.nii.gz peaks.nii.gz -num 3
}

function run_TractSeg() {
    echo "Running TractSeg..."
    cd $data_path/$1
    fn=peaks

    TractSeg -i "$fn".nii.gz
    TractSeg -i "$fn".nii.gz --output_type endings_segmentation
    TractSeg -i "$fn".nii.gz --output_type TOM

    Tracking -i "$fn".nii.gz --tracking_dir TOM_trackings_filt
}


function preprocessing_pipeline() {
    eddy_correct_extract_mask_denoise $1
    register_DWI_to_MNI $1
    create_fods $1
    run_TractSeg $1
}

#todo: install the following
# - FSL
# - Ants (needed for dwibiascorrect)
# - mrtrix
# - tractseg

#todo: your data needs to be in the following folder structure:
# For each subject you need a folder "<data_path>/<subject_id>/preproc" containing the following files:
#   Diffusion.nii.gz
#   Diffusion.bvals
#   Diffusion.bvecs

#todo: set path to your data (the folder you specify here must contain one folder for each subject)
data_path="/home/ubuntu/data/my_dwi_project"

#todo: list of all subject IDs
subjects=( subject001 subject002 subject003 )
for i in "${subjects[@]}"
do
    echo "processing" $i
    preprocessing_pipeline $i
done

