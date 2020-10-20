'''
Assuming spacing is isotropic
'''
import os, sys, inspect
import nibabel as nib

file_in = sys.argv[1]   #T1 or DWI

img = nib.load(file_in)
affine = img.affine

print(str(abs(round(affine[0,0],2))))
