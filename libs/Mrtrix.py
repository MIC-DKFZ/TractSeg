import os

class Mrtrix():

    @staticmethod
    def create_brain_mask(file_path):
        print("Creating brain mask...")
        dir = os.path.dirname(file_path)
        file_without_ending = os.path.basename(file_path).split(".")[0]
        os.system('cd ' + dir)
        #todo: make fsl path flexibel
        os.system('/usr/local/fsl/bin/bet ' + file_without_ending + ' nodif_brain_mask.nii.gz  -f 0.3 -g 0 -m')
        os.system('rm nodif_brain_mask.nii.gz')           #masked brain
        os.system('mv nodif_brain_mask_mask.nii.gz nodif_brain_mask.nii.gz')

    @staticmethod
    def create_fods(file_path):
        print("Creating peaks...")
        dir = os.path.dirname(file_path)
        file= os.path.basename(file_path)
        os.system('cd ' + dir)

        csd_resolution= "HIGH"

        if csd_resolution == "HIGH":
            # MSMT 5TT
            # todo: Add default T1 name to Doku
            os.system("5ttgen fsl T1w_acpc_dc_restore_brain.nii.gz 5TT.mif -premasked")
            #todo: make bvals/vecs flexible
            os.system("dwi2response msmt_5tt " + file + " 5TT.mif RF_WM.txt RF_GM.txt RF_CSF.txt -voxels RF_voxels.mif -fslgrad Diffusion.bvecs Diffusion.bvals")         # multi-shell, multi-tissue
            os.system("dwi2fod msmt_csd " + file + " RF_WM.txt WM_FODs.mif RF_GM.txt GM.mif RF_CSF.txt CSF.mif -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals")       # multi-shell, multi-tissue

            # MSMT DHollander    (only works with msmt_csd, not with csd)
            #dhollander does not need a T1 image to estimate the response function (more recent (2016) than tournier (2013))
            #        dwi2response dhollander -mask nodif_brain_mask.nii.gz Diffusion.nii.gz RF_WM_DHol.txt RF_GM_DHol.txt RF_CSF_DHol.txt -fslgrad Diffusion.bvecs Diffusion.bvals
            #        #dwi2fod csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs_csd.mif RF_GM_DHol.txt GM_FODs_csd.mif RF_CSF_DHol.txt CSF_FODs_csd.mif -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals
            #        dwi2fod msmt_csd Diffusion.nii.gz RF_WM_DHol.txt WM_FODs.mif -fslgrad Diffusion.bvecs Diffusion.bvals -mask nodif_brain_mask.nii.gz

            os.system("sh2peaks WM_FODs.mif peaks.nii.gz")
        else:   #LOW
            # CSD Tournier
            # todo: make bvals/vecs flexible
            os.path("dwi2response tournier " + file + " response.txt -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals -force")
            os.path("dwi2fod csd " + file + " response.txt WM_FODs.mif -shell 1000 -mask nodif_brain_mask.nii.gz -fslgrad Diffusion.bvecs Diffusion.bvals -force")
            os.path("sh2peaks WM_FODs.mif peaks.nii.gz -force")