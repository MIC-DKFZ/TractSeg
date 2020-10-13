set -e

#Bundle specific postprocessing
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/BS_PP/tractseg_output --single_orientation --test
pytest -v tests/test_end_to_end.py::test_end_to_end::test_bundle_specific_postprocessing

#Bundle specific postprocessing
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/no_PP/tractseg_output --single_orientation --no_postprocess --test
pytest -v tests/test_end_to_end.py::test_end_to_end::test_postprocessing

#Get probabilities
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/Probs/tractseg_output --single_orientation --test --get_probabilities
pytest -v tests/test_end_to_end.py::test_end_to_end::test_get_probabilities

#Get uncertainties
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/Uncert/tractseg_output --single_orientation --test --uncertainty
pytest -v tests/test_end_to_end.py::test_end_to_end::test_uncertainty

#Density map regression
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/DM/tractseg_output --single_orientation --test --output_type dm_regression
pytest -v tests/test_end_to_end.py::test_end_to_end::test_density_regression

#Bundles
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/tractseg_output --single_orientation
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tractseg_output

#Endings
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/tractseg_output --output_type endings_segmentation --single_orientation --nr_cpus 1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_endingsseg_output

#TOM
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/tractseg_output --output_type TOM --nr_cpus 1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_peakreg_output

#Track
Tracking -i tests/reference_files/peaks.nii.gz -o examples/tractseg_output --nr_fibers 10000 --test 3

#Tractometry
Tractometry -i examples/tractseg_output/TOM_trackings -o examples/Tractometry.csv \
-e examples/tractseg_output/endings_segmentations/ -s tests/reference_files/FA.nii.gz --test 3
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tractometry

#Tractometry toy example
mkdir -p tractometry_toy_example
python tests/reference_files/create_toy_streamlines.py tractometry_toy_example
Tractometry -i tractometry_toy_example/TOM_trackings -o tractometry_toy_example/Tractometry.csv \
-e tractometry_toy_example/endings_segmentations/ -s tractometry_toy_example/toy_FA.nii.gz --test 2 \
--nr_points 10 --tracking_format trk_legacy
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tractometry_toy_example

#Statistical analysis of tractometry
plot_tractometry_results -i tests/reference_files/subjects_group.txt \
-o examples/tractometry_result_group.png --mc --save_csv
pytest -v tests/test_end_to_end.py::test_end_to_end::test_statistical_analysis_group
plot_tractometry_results -i tests/reference_files/subjects_correlation.txt \
-o examples/tractometry_result_correlation.png --mc --save_csv
pytest -v tests/test_end_to_end.py::test_end_to_end::test_statistical_analysis_correlation

#Bundles SR noPP
TractSeg -i tests/reference_files/peaks.nii.gz -o examples/SR_noPP/tractseg_output --single_orientation --super_resolution --no_postprocess
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tractseg_output_SR_noPP
