"""Exercise the cleanup call without invoking MRtrix or removing a file."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


def test_tck_cleanup_preserves_the_complete_filename():
    source = Path(__file__).parents[1] / "tractseg/libs/tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_under_test", source)
    tracking = importlib.util.module_from_spec(spec)
    modules = {"tractseg.libs." + name: Mock() for name in
               ("fiber_utils", "img_utils", "tractseg_prob_tracking", "peak_utils")}
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(tracking)
    image = Mock(affine=np.eye(4))
    image.get_fdata.return_value = np.zeros((2, 2, 2))
    folder = "/synthetic output/space ; literal"
    bundle = "bundle $(literal)"
    filename = folder + "/trackings/" + bundle + ".tck"
    with patch.object(tracking.nib, "load", return_value=image), \
            patch.object(tracking.subprocess, "call") as cleanup, \
            patch.object(tracking.fiber_utils, "convert_tck_to_trk") as converter:
        tracking._mrtrix_tck_to_trk(folder, "trackings", "", bundle, "trk", 1)
    cleanup.assert_called_once_with(["rm", "-f", "--", filename])
    assert converter.call_args.args[0] == filename
