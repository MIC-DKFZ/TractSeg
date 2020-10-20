import sys
import nibabel as nib

file_in = sys.argv[1]
file_out = sys.argv[2]

img = nib.load(file_in)
img_data = img.get_data()

img_data[img_data < 0] = 0  # better to use very small postive number?

img = nib.Nifti1Image(img_data, img.affine)
nib.save(img, file_out)

