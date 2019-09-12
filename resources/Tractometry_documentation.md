# Tractometry

> **NOTE**: This code was not used in any paper yet. We did not evaluate if this approach of doing Tractometry is
 the best compared to other approaches. Therefore use with care.  
 
> **Warning**: Use TractSeg `master` because earlier versions contain small bug in Tractometry script (streamlines 
incorrectly shifted by 0.5 voxels).  
`pip install https://github.com/MIC-DKFZ/TractSeg/archive/master.zip`

![Tractometry concept figure](Tractometry_concept1.png)

Measuring the FA (or MD or other values) along tracts can provide valuable insights (e.g. [Yeatman et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049790)).
In our implementation of Tractometry we do the following:
1. Resample all streamlines to an equal number of segments/points
2. Find the centroid of all streamlines
3. For each streamline assign each segment to the closest centroid segment
4. Evaluate the FA (or any other metric) at each segment of each streamline
5. For each centroid segment average the FA for all assigned streamline segments  

This approach is based on the Bundle Analytics paper from 
[Chandio et al](https://ww5.aievolution.com/hbm1901/index.cfm?do=abs.viewAbs&abs=1914). Please cite their work if you
use this.

An easier approach (which we used previously in TractSeg) would be to evaluate the FA along 20 equality distant 
points along each streamline. Then take the mean for each of those 20 points over all streamlines. However, this 
leads to more blurring of the segments as can be seen in the following figure:

![Tractometry methods comparison figure](Compare_tractometry_methods.png)

Run the following steps:
1. Go to the folder where you have your `Diffusion.nii.gz`, `Diffusion.bvals`, `Diffusion.bvecs` and `FA.nii.gz` files. 
They should rigidly be aligned to [MNI space](https://github.com/MIC-DKFZ/TractSeg#aligning-image-to-mni-space) and 
already be preprocessed (motion and distortion correction, ...).
2. Create segmentation of bundles:  
`TractSeg -i Diffusion.nii.gz -o tractseg_output --raw_diffusion_input --output_type tract_segmentation` (runtime on 
GPU: 2min ~14s)  
(**Note**: if you already have the MRtrix CSD peaks you can also pass those as input and remove the option `--raw_diffusion_input`)
3. Create segmentation of start and end regions of bundles:  
`TractSeg -i tractseg_output/peaks.nii.gz -o tractseg_output --output_type endings_segmentation` (runtime on GPU: ~42s)
4. Create Tract Orientation Maps and use them to do bundle-specific tracking:  
`TractSeg -i tractseg_output/peaks.nii.gz -o tractseg_output --output_type TOM` (runtime on GPU: ~1min 30s)  
`Tracking -i tractseg_output/peaks.nii.gz -o tractseg_output --nr_fibers 10000` (runtime on CPU: ~23min)  
 **Note**: As the streamline seeding is random, results will be slightly different everytime you run it. 
 A high number of streamlines like 10000 will keep this variation low. It is not recommendable to use a lower number.
5. Run tractometry:  
`cd tractseg_output`  
`Tractometry -i TOM_trackings/ -o Tractometry_subject1.csv -e endings_segmentations/ -s ../FA.nii.gz` (runtime on CPU: ~20s)  
6. Repeat step 1-4 for every subject (use a shell script for that)
7. Plot the results with [this python code](../examples/plot_tractometry_results.ipynb)

### Further options   
Instead of analysing the FA along the tracts you can also analyze the peak length along the tracts. 
This has one major advantage: The peaks are generated using Constrained Spherical Deconvolution (CSD) 
which can handle crossing fibers in contrast to FA which can not. How does TractSeg analyze the peak 
length along a certain tract:
CSD gives us up to three peaks per voxel (all further ones are discarded). 
Now TractSeg selects the peak which is most similar to the direction of the bundle at that voxel 
(= peak with the lowest angular error to the peak from the Tract Orientation Map for that bundle). 
The length of that peak is then analyzed.  
> **IMPORTANT NOTE**: The peak length depends on the response function which will be different for each subject.
Therefore results will not be comparable between subjects. To solve this you have to use the same response function
for all subjects when calculating the CSD. Moreover, you should use bias field correction and intensity normalisation. 
Those steps are documented in detail 
[here](https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/st_fibre_density_cross-section.html).
You have to make yourself familiar with this before you use it. (some more information can also be found 
[here](https://github.com/MIC-DKFZ/TractSeg/issues/42))

`Tractometry -i TOM_trackings/ -o Tractometry_subject1.csv -e endings_segmentations/ -s peaks.nii.gz --TOM TOM --peak_length`