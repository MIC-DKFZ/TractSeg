from tractseg.config.BaseHP import HP as BaseHP

class HP(BaseHP):

    DAUG_SCALE = True
    DAUG_ELASTIC_DEFORM = False
    DAUG_ROTATE = False
    DAUG_RESAMPLE = True
    DAUG_NOISE = True

    DATASET = "HCP"  # HCP / HCP_32g
    RESOLUTION = "1.25mm"  # 1.25mm (/ 2.5mm)
    FEATURES_FILENAME = "270g_125mm_peaks"  # 12g90g270g / 270g_125mm_xyz / 270g_125mm_peaks / 90g_125mm_peaks / 32g_25mm_peaks / 32g_25mm_xyz
    LABELS_FILENAME = "bundle_peaks/CA"  # IMPORTANT: Adapt BatchGen if 808080              # bundle_masks / bundle_masks_72 / bundle_masks_dm / bundle_peaks      #Only used when using DataManagerNifti
    CLASSES = "CA"
