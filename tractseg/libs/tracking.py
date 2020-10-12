
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import shutil
import subprocess

import nibabel as nib
import numpy as np

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils
from tractseg.libs import tractseg_prob_tracking
from tractseg.libs import peak_utils


def _mrtrix_tck_to_trk(output_dir, tracking_folder, dir_postfix, bundle, output_format, nr_cpus):
    ref_img = nib.load(output_dir + "/bundle_segmentations" + dir_postfix + "/" + bundle + ".nii.gz")
    reference_affine = ref_img.affine
    reference_shape = ref_img.get_fdata().shape[:3]
    fiber_utils.convert_tck_to_trk(output_dir + "/" + tracking_folder + "/" + bundle + ".tck",
                                   output_dir + "/" + tracking_folder + "/" + bundle + ".trk",
                                   reference_affine, reference_shape, compress_err_thr=0.1, smooth=None,
                                   nr_cpus=nr_cpus, tracking_format=output_format)
    subprocess.call("rm -f " + output_dir + "/" + tracking_folder + "/" + bundle + ".tck", shell=True)


def get_tracking_folder_name(tracking_algorithm, use_best_original_peaks):
    if tracking_algorithm == "FACT":
        tracking_folder = "Peaks_FACT_trackings"
    elif tracking_algorithm == "SD_STREAM":
        tracking_folder = "FOD_SD_STREAM_trackings"
    elif tracking_algorithm == "iFOD2":
        tracking_folder = "FOD_iFOD2_trackings"
    elif use_best_original_peaks:
        tracking_folder = "BestOrig_trackings"
    else:
        tracking_folder = "TOM_trackings"
    return tracking_folder


def track(bundle, peaks, output_dir, tracking_on_FODs, tracking_software, tracking_algorithm,
          use_best_original_peaks=False, use_as_prior=False, filter_by_endpoints=True,
          tracking_folder="auto", dir_postfix="", dilation=1,
          next_step_displacement_std=0.15,
          output_format="trk", nr_fibers=2000, nr_cpus=-1):

    ################### Preparing ###################

    # Auto set tracking folder name
    if tracking_folder == "auto":
        tracking_folder = get_tracking_folder_name(tracking_algorithm, use_best_original_peaks)
    TOM_folder = "TOM" + dir_postfix

    # Set nr threads for MRtrix
    if nr_cpus > 0:
        nthreads = " -nthreads " + str(nr_cpus)
    else:
        nthreads = ""

    # Misc
    subprocess.call("export PATH=/code/mrtrix3/bin:$PATH", shell=True)
    subprocess.call("mkdir -p " + output_dir + "/" + tracking_folder, shell=True)
    tmp_dir = tempfile.mkdtemp()

    # Check if bundle masks are valid
    if filter_by_endpoints:
        bundle_mask_ok = nib.load(output_dir + "/bundle_segmentations" + dir_postfix
                                  + "/" + bundle + ".nii.gz").get_fdata().max() > 0
        beginnings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz").get_fdata().max() > 0
        endings_mask_ok = nib.load(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz").get_fdata().max() > 0

        if not bundle_mask_ok:
            print("WARNING: tract mask of {} empty. Creating empty tractogram.".format(bundle))

        if not beginnings_mask_ok:
            print("WARNING: tract beginnings mask of {} empty. Creating empty tractogram.".format(bundle))

        if not endings_mask_ok:
            print("WARNING: tract endings mask of {} empty. Creating empty tractogram.".format(bundle))


    ################### Tracking ###################

    if not bundle_mask_ok or not beginnings_mask_ok or not endings_mask_ok:
        fiber_utils.create_empty_tractogram(output_dir + "/" + tracking_folder + "/" +
                                            bundle + "." + output_format,
                                            output_dir + "/bundle_segmentations" + dir_postfix + "/" +
                                            bundle + ".nii.gz",
                                            tracking_format=output_format)
    else:
        # Filtering
        if filter_by_endpoints:

            # Mrtrix Tracking
            if tracking_software == "mrtrix":

                # Prepare files
                img_utils.dilate_binary_mask(output_dir + "/bundle_segmentations" + dir_postfix + "/" + bundle + ".nii.gz",
                                             tmp_dir + "/" + bundle + ".nii.gz", dilation=dilation)
                img_utils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz",
                                             tmp_dir + "/" + bundle + "_e.nii.gz", dilation=dilation + 1)
                img_utils.dilate_binary_mask(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz",
                                             tmp_dir + "/" + bundle + "_b.nii.gz", dilation=dilation + 1)

                # Mrtrix tracking on original FODs (have to be provided to -i)
                if tracking_on_FODs:
                    if tracking_algorithm == "FACT" or tracking_algorithm == "SD_STREAM":
                        seeds = 1000000
                    else:
                        seeds = 200000
                    # Quite slow
                    # cutoff 0.1 gives more sensitive results than 0.05 (default) (tested for HCP msmt)
                    # - better for CA & FX (less oversegmentation)
                    # - worse for CST (missing lateral projections)
                    subprocess.call("tckgen -algorithm " + tracking_algorithm + " " +
                                    peaks + " " +
                                    output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                    " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                    " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                    " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                    " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                    " -minlength 40 -maxlength 250 -seeds " + str(seeds) +
                                    " -select " + str(nr_fibers) + " -cutoff 0.05 -force" + nthreads,
                                    shell=True)
                    if output_format == "trk" or output_format == "trk_legacy":
                        _mrtrix_tck_to_trk(output_dir, tracking_folder, dir_postfix, bundle, output_format, nr_cpus)

                else:
                    # FACT tracking on TOMs
                    if tracking_algorithm == "FACT":
                        # Takes around 2.5min for 1 subject (2mm resolution)
                        subprocess.call("tckgen -algorithm FACT " +
                                        output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                                        output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                        " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                        " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                        " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                        " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                        " -minlength 40 -maxlength 250 -select " + str(nr_fibers) +
                                        " -force -quiet" + nthreads,
                                        shell=True)
                        if output_format == "trk" or output_format == "trk_legacy":
                            _mrtrix_tck_to_trk(output_dir, tracking_folder, dir_postfix, bundle, output_format, nr_cpus)

                    # iFOD2 tracking on TOMs
                    elif tracking_algorithm == "iFOD2":
                        # Takes around 12min for 1 subject (2mm resolution)
                        img_utils.peaks2fixel(output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz", tmp_dir + "/fixel")
                        subprocess.call("fixel2sh " + tmp_dir + "/fixel/amplitudes.nii.gz " +
                                        tmp_dir + "/fixel/sh.nii.gz -quiet", shell=True)
                        subprocess.call("tckgen -algorithm iFOD2 " +
                                        tmp_dir + "/fixel/sh.nii.gz " +
                                        output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                                        " -seed_image " + tmp_dir + "/" + bundle + ".nii.gz" +
                                        " -mask " + tmp_dir + "/" + bundle + ".nii.gz" +
                                        " -include " + tmp_dir + "/" + bundle + "_b.nii.gz" +
                                        " -include " + tmp_dir + "/" + bundle + "_e.nii.gz" +
                                        " -minlength 40 -maxlength 250 -select " + str(nr_fibers) +
                                        " -force -quiet" + nthreads,
                                        shell=True)
                        if output_format == "trk" or output_format == "trk_legacy":
                            _mrtrix_tck_to_trk(output_dir, tracking_folder, dir_postfix, bundle, output_format, nr_cpus)

                    else:
                        raise ValueError("Unknown tracking algorithm: {}".format(tracking_algorithm))


            # TractSeg probabilistic tracking
            else:

                # Prepare files
                bundle_mask_img = nib.load(output_dir + "/bundle_segmentations" + dir_postfix + "/"
                                           + bundle + ".nii.gz")
                beginnings_img = nib.load(output_dir + "/endings_segmentations/" + bundle + "_b.nii.gz")
                endings_img = nib.load(output_dir + "/endings_segmentations/" + bundle + "_e.nii.gz")
                tom_peaks_img = nib.load(output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz")

                # Ensure same orientation as MNI space
                bundle_mask, flip_axis = img_utils.flip_axis_to_match_MNI_space(bundle_mask_img.get_fdata().astype(np.uint8),
                                                                                bundle_mask_img.affine)
                beginnings, flip_axis = img_utils.flip_axis_to_match_MNI_space(beginnings_img.get_fdata().astype(np.uint8),
                                                                                beginnings_img.affine)
                endings, flip_axis = img_utils.flip_axis_to_match_MNI_space(endings_img.get_fdata().astype(np.uint8),
                                                                                endings_img.affine)
                tom_peaks, flip_axis = img_utils.flip_axis_to_match_MNI_space(tom_peaks_img.get_fdata(),
                                                                                  tom_peaks_img.affine)

                # tracking_uncertainties = nib.load(output_dir + "/tracking_uncertainties/" + bundle + ".nii.gz").get_fdata()
                tracking_uncertainties = None

                #Get best original peaks
                if use_best_original_peaks:
                    orig_peaks_img = nib.load(peaks)
                    orig_peaks, flip_axis = img_utils.flip_axis_to_match_MNI_space(orig_peaks_img.get_fdata(),
                                                                                   orig_peaks_img.affine)
                    best_orig_peaks = fiber_utils.get_best_original_peaks(tom_peaks, orig_peaks)
                    for axis in flip_axis:
                        best_orig_peaks = img_utils.flip_axis(best_orig_peaks, axis)
                    nib.save(nib.Nifti1Image(best_orig_peaks, orig_peaks_img.affine),
                             output_dir + "/" + tracking_folder + "/" + bundle + ".nii.gz")
                    tom_peaks = best_orig_peaks

                #Get weighted mean between best original peaks and TOMs
                if use_as_prior:
                    orig_peaks_img = nib.load(peaks)
                    orig_peaks, flip_axis = img_utils.flip_axis_to_match_MNI_space(orig_peaks_img.get_fdata(),
                                                                                   orig_peaks_img.affine)
                    best_orig_peaks = fiber_utils.get_best_original_peaks(tom_peaks, orig_peaks)
                    weighted_peaks = fiber_utils.get_weighted_mean_of_peaks(best_orig_peaks, tom_peaks, weight=0.5)
                    for axis in flip_axis:
                        weighted_peaks = img_utils.flip_axis(weighted_peaks, axis)
                    nib.save(nib.Nifti1Image(weighted_peaks, orig_peaks_img.affine),
                             output_dir + "/" + tracking_folder + "/" + bundle + "_weighted.nii.gz")
                    tom_peaks = weighted_peaks

                # Takes around 6min for 1 subject (2mm resolution)
                streamlines = tractseg_prob_tracking.track(tom_peaks, max_nr_fibers=nr_fibers, smooth=5,
                                                           compress=0.1, bundle_mask=bundle_mask, start_mask=beginnings,
                                                           end_mask=endings,
                                                           tracking_uncertainties=tracking_uncertainties,
                                                           dilation=dilation,
                                                           next_step_displacement_std=next_step_displacement_std,
                                                           nr_cpus=nr_cpus, affine=bundle_mask_img.affine,
                                                           spacing=bundle_mask_img.header.get_zooms()[0],
                                                           verbose=False)

                if output_format == "trk_legacy":
                    fiber_utils.save_streamlines_as_trk_legacy(output_dir + "/" + tracking_folder + "/" + bundle + ".trk",
                                                               streamlines, bundle_mask_img.affine,
                                                               bundle_mask_img.get_fdata().shape)
                else:  # tck or trk (determined by file ending)
                    fiber_utils.save_streamlines(
                        output_dir + "/" + tracking_folder + "/" + bundle + "." + output_format,
                        streamlines, bundle_mask_img.affine,
                        bundle_mask_img.get_fdata().shape)


        # No streamline filtering
        else:

            peak_utils.peak_image_to_binary_mask_path(peaks, tmp_dir + "/peak_mask.nii.gz",
                                                      peak_length_threshold=0.01)

            # FACT Tracking on TOMs
            subprocess.call("tckgen -algorithm FACT " +
                            output_dir + "/" + TOM_folder + "/" + bundle + ".nii.gz " +
                            output_dir + "/" + tracking_folder + "/" + bundle + ".tck" +
                            " -seed_image " + tmp_dir + "/peak_mask.nii.gz" +
                            " -minlength 40 -maxlength 250 -select " + str(nr_fibers) +
                            " -force -quiet" + nthreads, shell=True)

            if output_format == "trk" or output_format == "trk_legacy":
                _mrtrix_tck_to_trk(output_dir, tracking_folder, dir_postfix, bundle, output_format, nr_cpus)


    shutil.rmtree(tmp_dir)
