#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import nibabel as nib
import numpy as np
import dipy.reconst.dti as dti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy


def main():
    parser = argparse.ArgumentParser(description="Generate FA image.",
                                        epilog="Written by Jakob Wasserthal. Please reference 'Wasserthal et al. "
                                               "TractSeg - Fast and accurate white matter tract segmentation. "
                                               "https://doi.org/10.1016/j.neuroimage.2018.07.070)'")
    parser.add_argument("-i", metavar="file_in", dest="file_in", help="input file (4D nifti diffusion image)",
                        required=True)
    parser.add_argument("-o", metavar="file_out", dest="file_out", help="FA image", required=True)
    parser.add_argument("--bvals", metavar="bvals", dest="bvals", help="bvals file", required=True)
    parser.add_argument("--bvecs", metavar="bvecs", dest="bvecs", help="bvecs file", required=True)
    parser.add_argument("--brain_mask", metavar="brain_mask", dest="brain_mask", help="a brain mask", required=True)
    args = parser.parse_args()

    img = nib.load(args.file_in)
    data = img.get_fdata()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    gtab = gradient_table(bvals, bvecs)

    mask = nib.load(args.brain_mask).get_fdata()
    masked_brain = data
    masked_brain[mask < 0.5] = 0

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(masked_brain)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0

    fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
    nib.save(fa_img, args.file_out)


if __name__ == '__main__':
    main()