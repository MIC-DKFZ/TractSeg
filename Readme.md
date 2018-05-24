# TractSeg
 
![Alt text](examples/Pipeline_img_for_readme.png)

Tool for fast and accurate white matter bundle segmentation from Diffusion MRI. It can create 
bundle segmentations, segmentations of the endregions of bundles and Tract Orientation Maps (TOMs).

The tool works very good for HCP style data. For other MRI datasets it also works but results 
will have lower quality due to the domain gap.

TractSeg is the code for the paper [TractSeg - Fast and accurate white matter tract segmentation](https://arxiv.org/abs/1805.07103). 
Please cite the paper if you use it. 

[![Build Status](https://travis-ci.org/MIC-DKFZ/TractSeg.svg?branch=master)](https://travis-ci.org/MIC-DKFZ/TractSeg)

## Install
TractSeg only runs on Linux and OSX. It works with Python 2 and Python 3.

#### Install Prerequisites
* [Pytorch](http://pytorch.org/) (if you do not have a GPU, install Pytorch via conda as this is fastest on CPU)
* [Mrtrix 3](http://mrtrix.readthedocs.io/en/latest/installation/linux_install.html)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) (if you already have a brain mask this is not needed)
* BatchGenerators: `pip install https://github.com/MIC-DKFZ/batchgenerators/archive/tractseg_stable.zip`

#### Install TractSeg
Latest stable version:
```
pip install https://github.com/MIC-DKFZ/TractSeg/archive/v1.2.zip
```

## Usage

#### Simple example:
To segment the bundles on a Diffusion Nifti image run the following command. 
You can use the example image provided in this repository under `examples`.  
```
TractSeg -i Diffusion.nii.gz    # expects Diffusion.bvals and Diffusion.bvecs to be in the same directory
```
This will create a folder `tractseg_ouput` inside of the same directory as your input file. 
This folder contains `bundle_segmentations.nii.gz` which is a 4D Nifti image (`[x,y,z,bundle]`). 
The fourth dimension contains the binary bundle segmentations. 

#### Custom input and output path:
```
TractSeg -i my/path/my_diffusion_image.nii.gz
         -o my/output/directory
         --bvals my/other/path/my.bvals
         --bvecs yet/another/path/my.bvecs
         --output_multiple_files
```

#### Use existing peaks
```
TractSeg -i my/path/my_mrtrix_csd_peaks.nii.gz --skip_peak_extraction
```

#### Create Tract Orientation Maps (TOMs)
TOM only works for HCP data so far.
```
TractSeg -i Diffusion.nii.gz --output_type TOM --output_multiple_files
```

#### Segment bundle start and end regions
```
TractSeg -i Diffusion.nii.gz --output_type endings_segmentation --output_multiple_files
```

#### Bundle names
The following list shows the index of 
each extracted bundle in the output file.
```
0: AF_left         (Arcuate fascicle)
1: AF_right
2: ATR_left        (Anterior Thalamic Radiation)
3: ATR_right
4: CA              (Commissure Anterior)
5: CC_1            (Rostrum)
6: CC_2            (Genu)
7: CC_3            (Rostral body (Premotor))
8: CC_4            (Anterior midbody (Primary Motor))
9: CC_5            (Posterior midbody (Primary Somatosensory))
10: CC_6           (Isthmus)
11: CC_7           (Splenium)
12: CG_left        (Cingulum left)
13: CG_right   
14: CST_left       (Corticospinal tract)
15: CST_right 
16: MLF_left       (Middle longitudinal fascicle)
17: MLF_right
18: FPT_left       (Fronto-pontine tract)
19: FPT_right 
20: FX_left        (Fornix)
21: FX_right
22: ICP_left       (Inferior cerebellar peduncle)
23: ICP_right 
24: IFO_left       (Inferior occipito-frontal fascicle) 
25: IFO_right
26: ILF_left       (Inferior longitudinal fascicle) 
27: ILF_right 
28: MCP            (Middle cerebellar peduncle)
29: OR_left        (Optic radiation) 
30: OR_right
31: POPT_left      (Parieto‐occipital pontine)
32: POPT_right 
33: SCP_left       (Superior cerebellar peduncle)
34: SCP_right 
35: SLF_I_left     (Superior longitudinal fascicle I)
36: SLF_I_right 
37: SLF_II_left    (Superior longitudinal fascicle II)
38: SLF_II_right
39: SLF_III_left   (Superior longitudinal fascicle III)
40: SLF_III_right 
41: STR_left       (Superior Thalamic Radiation)
42: STR_right 
43: UF_left        (Uncinate fascicle) 
44: UF_right 
45: CC             (Corpus Callosum - all)
46: T_PREF_left    (Thalamo-prefrontal)
47: T_PREF_right 
48: T_PREM_left    (Thalamo-premotor)
49: T_PREM_right 
50: T_PREC_left    (Thalamo-precentral)
51: T_PREC_right 
52: T_POSTC_left   (Thalamo-postcentral)
53: T_POSTC_right 
54: T_PAR_left     (Thalamo-parietal)
55: T_PAR_right 
56: T_OCC_left     (Thalamo-occipital)
57: T_OCC_right 
58: ST_FO_left     (Striato-fronto-orbital)
59: ST_FO_right 
60: ST_PREF_left   (Striato-prefrontal)
61: ST_PREF_right 
62: ST_PREM_left   (Striato-premotor)
63: ST_PREM_right 
64: ST_PREC_left   (Striato-precentral)
65: ST_PREC_right 
66: ST_POSTC_left  (Striato-postcentral)
67: ST_POSTC_right
68: ST_PAR_left    (Striato-parietal)
69: ST_PAR_right 
70: ST_OCC_left    (Striato-occipital)
71: ST_OCC_right
```


#### Advanced Options
Run `TractSeg --help` for more advanced options. For example you can specify your own `brain_mask`,
`bvals` and `bvecs`.

If you have multi-shell data and you do not need fast runtime use `--csd_type csd_msmt_5tt` for slightly better results.


## FAQ
**My output segmentation does not look like any bundle at all!**

The input image must have the same "orientation" as the Human Connectome Project data (LEFT must be 
on the same side as LEFT of the HCP data). If the image 
orientation and the gradient orientation of your data is the same as in `examples/Diffusion.nii.gz`
you are fine.

**Did I install the prerequisites correctly?**

You can check if you installed Mrtrix correctly if you can run the following command on your terminal:
`dwi2response -help`

You can check if you installed FSL correctly if you can run the following command on your terminal: 
`bet -help`

TractSeg uses these commands so they have to be available.

**My image does not contain any b=1000mm/s^2 values.**

Use `--csd_type csd_msmt` or `--csd_type csd_msmt_5tt`. Those work for any b-value.


## Train your own model
TractSeg uses a pretrained model. However, you can also train your own model on your own data.
But be aware: This is more complicated than just running with the pretrained model. The following 
guide is quite short and you might have problems following every step. Contact the author if
you need help training your own model.

1. The folder structure of your training data should be the following:
```
custom_path/HCP/subject_01/
      '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
      '-> bundle_masks.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
custom_path/HCP/subject_02/
      ...
```
2. Adapt the file tractseg/config/custom/My_custom_experiment.py.
3. Create a file `~/.tractseg/config.txt`. This contains the path to your data directory, e.g.
`working_dir=custom_path`.
4. Adapt `tractseg.libs.DatasetUtils.scale_input_to_unet_shape()` to scale your input data to the 
UNet input size of `144x144`. This is not very convenient. Contact the author if you need help.
5. Adapt `tractseg.libs.ExpUtils.get_bundle_names()` with the bundles you use in your reference data.
6. Adapt `tractseg.libs.ExpUtils.get_labels_filename()` with the names of your label files.
7. Adapt `tractseg.libs.Subjects` with the list of your subject IDs.
8. Run `ExpRunner --config My_custom_experiment` 
9. `custom_path/hcp_exp/My_custom_experiment` contains the results

