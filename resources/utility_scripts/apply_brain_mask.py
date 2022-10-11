import os, sys
import nibabel as nib
import numpy as np

file_in = sys.argv[1]  # T1 or DWI
file_out = sys.argv[2]  # out
mask_in = sys.argv[3]  # brain_mask

img = nib.load(file_in)
img_data = img.get_fdata()

mask = nib.load(mask_in).get_fdata() < 0.5
img_data[mask] = 0

nib.save(nib.Nifti1Image(img_data.astype(np.int32), img.affine), file_out)
