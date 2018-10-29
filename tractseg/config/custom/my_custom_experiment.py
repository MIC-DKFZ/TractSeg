import os
from tractseg.config.tract_seg import HP as TractSegHP

class HP(TractSegHP):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET_FOLDER = "HCP"              # name of folder that contains all the subjects (each subject has its own folder with the name of the subjectID)
    FEATURES_FILENAME = "mrtrix_peaks"  # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
