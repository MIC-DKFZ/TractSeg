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
* minor improvements and bugfixes


## Release 1.5

* Super-resolution
* Uncertainty estimation
* Automatic preprocessing (rigid registration to MNI space + automatic check for correct peak orientation and flip if needed) 
* Minor improvements and bugfixes
* Automatic tracking on Tract Orientation Maps
* Postprocessing and bundle specific threshold for improved results on small bundles