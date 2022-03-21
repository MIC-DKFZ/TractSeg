## Master
* Update documentation
* fixed bug with newer fsl bet (thanks to @CarolynMcNabb)


## Release 2.5
* Move cython tracking to `cython_tracking` branch. It is not part of the main branch and pip package anymore because building the cython binaries for all possible operating system is a lot of pain. If people want to use it they can install from the `cython_tracking` branch directly which will automatically compile the cython code on their sytem.


## Release 2.4
* Update documentation
* Bugfix when tracking in non-MNI space
* Use Cython for Tracking. This will speed up tracking by roughly 2x and only require 1 CPU core instead of all CPU cores. Thanks to @StavrAspa and @eliaskoromilas for this contribution.


## Release 2.3
* Update documentation
* Update 3d tract visualization in `plot_tractometry_results` to use fury
* Updated pytorch to 1.8.1


## Release 2.2

* 3D plot of streamlines with coloring according to tractometry FA
* Add pretrained weights for XTRACT tract definitions
* Set range of y-axis in `plot_tractometry_results` (thanks to @elder-mama)
* Make `--preprocess` also work for endings_segmentation and TOM
* Simplified dockerfile
* Minor improvements


## Release 2.1.1

* Available on pypi
* Manually specify which bundles to track
* Minor improvements


## Release 2.1

* **Interface change**: The option `--bundle_specific_threshold` was removed. TractSeg checks itself now if CA or FX 
are incomplete and then applies a lower threshold.
* **Interface change**: Postprocessing is activated by default now. If you want to deactivate is use `--no_postprocess`.
* minor improvements & Bugfixes
* FP16 training (increased training speed)
* Tractometry more testing and bugfix
* Tractometry now uses a far more advanced option to sample e.g. the FA along the tracts.
* Statistical analysis for tractometry data
* Python 2 not actively supported anymore (because dipy 1.0.0 does not support python 2 anymore)
* Add rotation (now peaks are also properly rotated) to data augmentation.
* Applies signs of affine to data if array not oriented like MNI data (needed to properly work with `fslreorient2std`)
* '--preprocess' will move output back to subject space.
* Updated weights for tract segmentation, endings segmentation and density regression (slightly increased 
performance; now also trained with rotation during data augmentation; TOM not trained with rotation yet)
* Pretrained model also works with bedpostX peaks (instead of CSD peaks) (segmentation accuracy is the same)


## Release 2.0

* Increase training speed roughly by factor of 2 by using pin_memory and non_blocking for pytorch and by 
cropping all non-brain area from the input images (requires preprocessing of the training data using
`tractseg/data/preprocessing.py`).
* Works with newer version of batchgenerators (Note: DataAugmentation slightly changed)
* Support bedpostX input
* Support aPTX tract definitions (but no pretrained model yet)
* Refactor `--preview`. Works without vtk now.
* Add plateau LR schedule to training
* Add API for mrtrix FACT tracking on TOMs
* Fix bug in rotation of bvecs when using option `--preprocess`.
* minor improvements
* Update TOM model and pretrained weights (Only angle in loss instead of angle and length. Gives slightly better 
peak orientations.). Improved peak orientations allows for slightly less sensitive probabilistic tracking: lowering
stddev from 0.2 to 0.15.


## Release 1.9

* Tracking on best original peaks or on weighted mean of best original peaks and TOMs (non-public interface).
* **Interface change**: All tracking related commands (whenever you used `--track`) are not part of `TractSeg` anymore
 but now are combined under `Tracking`. Moreover the option `--filter_tracking_by_endpoints` is now activated per
 default. If you want to deactivate is use `--no_filtering_by_endpoints`.
 So the following command 
```
TractSeg -i peaks.nii.gz --output_type TOM --track --filter_tracking_by_endpoints
``` 
becomes 
```
TractSeg -i peaks.nii.gz --output_type TOM 
Tracking -i peaks.nii.gz
```
* Works with pytorch 1.0 now
* Bugfixes and minor improvements 


## Release 1.8

* "Probabilistic" tracking on TOM output
* New trk format (nibabel.streamlines API). Use `--tracking_format trk` to use it.
* Option to do Mrtrix iFOD2 tracking on original FODs but filter by TractSeg masks.
* Added 3D U-Net, but not used
* Minor improvements & Bugfixes


## Release 1.7.1

* **Interface change**: TractSeg does not automatically flip the peaks anymore if it detects that they probably have
the wrong orientation (The peak check is only correct in 98% of the cases. In the remaining 2% it would incorrectly flip
the peaks and the user would wonder why the results of TractSeg are so bad. Therefore now the user is informed if
TractSeg thinks that flipping is needed, but he has to do it on his own and manually verify the result.) Therefore
the command line option `-deactivate_peak_check` is not needed anymore and removed.
* Bugfixes


## Release 1.7

* Reduced memory consumption of TOM (downside: `--single_output_file` for TOM not working anymore)
* Added more testing
* Number of fibers can be set as parameter
* Brain mask not needed anymore for tracking
* Make location where to store pretrained weights customizable
* Bugfixes and minor improvements


## Release 1.6

* TOM (Tract Orientation Mapping) now supports all 72 bundles instead of only 20. Downside: needs 4x more RAM (roughly 22GB).
* Code for performing [tractometry](resources/Tractometry_documentation.md)
* Added more documentation: [Best pratices for standard usecases](resources/Tutorial.md)
* Removed batchgenerators dependency. Now it might even work on windows (not tested yet!).
* **Breaking Change**: Improved interface:
    * `-i` expects a peak image by default now. If you provide a Diffusion image you have to set `--raw_diffusion_input` to make
    TractSeg run CSD and extract peaks
    * `--output_multiple_files` is default now. If you only want one output file set `--single_output_file`
* When using a Diffusion image as input and setting `--raw_diffusion_input` the resulting peak image `peaks.nii.gz` will
not be deleted after TractSeg is finished, but stays there. Good if you want to run TractSeg again with other 
output type.
* Minor improvements and bugfixes


## Release 1.5

* Super-resolution
* Uncertainty estimation
* Automatic preprocessing (rigid registration to MNI space + automatic check for correct peak orientation and flip if needed) 
* Minor improvements and bugfixes
* Automatic tracking on Tract Orientation Maps
* Postprocessing and bundle specific threshold for improved results on small bundles


## Release 1.4

* Updated to pytorch 0.4
* Added bundle specific threshold
* Endings segmentations for all 72 classes
* More testing
* Bugfixes and minor improvements


## Release 1.3

* Add dropout sampling
* Add density map regression
* Bugfixes and minor improvements
