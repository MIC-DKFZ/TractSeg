"""
Random code for testing purposes
"""
from pathlib import Path
import subprocess as sp
from p_tqdm import p_map
import numpy as np


def run_tractseg(subject_id):
    # dir = base / subject_id
    dir = base / subject_id / "session_1"
    # sp.call(f"TractSeg -i {dir}/peaks.nii.gz --preview", shell=True)
    # sp.call(f"TractSeg -i {dir}/peaks.nii.gz --output_type endings_segmentation --preview", shell=True)
    # sp.call(f"TractSeg -i {dir}/peaks.nii.gz --output_type TOM --preview", shell=True)
    sp.call(f"Tracking -i {dir}/peaks.nii.gz --tracking_format tck --algorithm prob --test 3", shell=True)
    # sp.call(f"Tractometry -i {dir}/tractseg_output/TOM_trackings " +
    #         f"-o {dir}/tractseg_output/Tractometry.csv " +
    #         f"-e {dir}/tractseg_output/endings_segmentations -s {dir}/FA.nii.gz --tracking_format tck", 
    #         shell=True)


if __name__ == '__main__':
    # base = Path("/mnt/nvme/data/dwi/tractometry_test_subjectSpace")
    base = Path("/mnt/nvme/data/dwi/tractseg_example")
    # base = Path("/mnt/nvme/data/dwi/rotation_test")
    # subjects = ["s01", "s02", "s03", "s04"]
    subjects = ["s01"]
    # subjects = ["UZB"]

    def process_subject(subject_id):
        run_tractseg(subject_id)

    p_map(process_subject, subjects, num_cpus=1, disable=False)

    # Run Tractometry statistics
    # cd /mnt/nvme/data/dwi/tractometry_test
    # plot_tractometry_results -i subjects.txt -o tractometry_result_group.png --mc --save_csv --plot3D metric
