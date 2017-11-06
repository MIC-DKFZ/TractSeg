#!/bin/bash
set -e  # stop on error
#
# Parameters:
#   $1: Path to Input DWI (as nifti)
#


function create_brain_mask() {
    echo "Creating brain mask..."
    DIR=$(dirname "${1}")
    cd $DIR
    /usr/local/fsl/bin/bet Diffusion nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m
    rm nodif_brain_mask.nii.gz      #masked brain
    mv nodif_brain_mask_mask.nii.gz nodif_brain_mask.nii.gz
}

function create_fods() {
    echo "Creating Peaks..."
    DIR=$(dirname "${1}")
    cd $DIR

    #dwi2mask Diffusion.nii.gz mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals   #if we do not have a mask yet

    if [ "$resolution" = "HIGH" ]; then
        5ttgen fsl T1w_acpc_dc_restore_brain.nii.gz 5TT.mif -premasked  #in LOWRES: Needed for constrained Tracking
    fi

    if [ "$resolution" = "HIGH" ]; then
        # MSMT 5TT
        dwi2response msmt_5tt Diffusion.nii.gz 5TT.mif RF_WM.txt RF_GM.txt RF_CSF.txt -voxels RF_voxels.mif -fslgrad Diffusion.bvecs Diffusion.bvals         # multi-shell, multi-tissue
        dwi2fod msmt_csd Diffusion.nii.gz RF_WM.txt WM_FODs.mif RF_GM.txt GM.mif RF_CSF.txt CSF.mif -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals       # multi-shell, multi-tissue

        # MSMT DHollander    (only works with msmt_csd, not with csd)
        #dhollander does not need a T1 image to estimate the response function (more recent (2016) than tournier (2013))
#        dwi2response dhollander -mask nodif_brain_mask.nii.gz Diffusion.nii.gz RF_WM_DHol.txt RF_GM_DHol.txt RF_CSF_DHol.txt -fslgrad Diffusion.bvecs Diffusion.bvals
#        #dwi2fod csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs_csd.mif RF_GM_DHol.txt GM_FODs_csd.mif RF_CSF_DHol.txt CSF_FODs_csd.mif -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals
#        dwi2fod msmt_csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs.mif -fslgrad Diffusion.bvecs Diffusion.bvals -mask nodif_brain_mask.nii.gz

        sh2peaks WM_FODs.mif peaks.nii.gz
    else    #LOW
        # CSD Tournier
        dwi2response tournier Diffusion.nii.gz response.txt -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals -force
        dwi2fod csd Diffusion.nii.gz response.txt WM_FODs_csd_b1000.mif -shell 1000 -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals -force
        sh2peaks WM_FODs_csd_b1000.mif peaks.nii.gz -force
    fi

}

function segment_bundles() {
    echo "Segmenting bundles..."
    DIR=$(dirname "${1}")
    cd $DIR

    #--en HCP_fold0 / HCP_normAfter  -> second one is with Rotation and Mirroring
    python ~/dev/dl-tracking/ExpRunner.py --train=False --test=False --lw=True \
    --en=HCP_fold0\
    --predict_img=peaks.nii.gz \
    --predict_img_out=bundle_segmentation.nii.gz
}

#type=$(python ~/dev/dl-tracking/ShellScripts/helpers/get_HOST_TYPE.py)
resolution="LOW"    # LOW / HIGH

echo "Processing" $1
#create_brain_mask $1
#create_fods $1
segment_bundles $1


#Commands:
# if needed:    python ~/dev/dl-tracking/scripts/preprocessing/flip_gradients.py Diffusion flipGrad_x/Diffusion
# locally:      sh segment_new_subject.sh /Volumes/E130-Personal/Wasserthal/data/DTI_Challenge_2015/patient3/32g_25mm/flipGrad_x/Diffusion.nii.gz
# gpu node:     ./segment_new_subject.sh /mnt/jakob/E130-Personal/Wasserthal/data/VISIS/s01/243g_25mm/flipGrad_x/Diffusion.nii.gz
