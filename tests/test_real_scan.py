"""Integration test for the 128-cube MRI preprocessing pipeline."""

import os
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

from preprocessing.config import PipelineConfig
from preprocessing.pipeline import preprocess_mri_scan

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_real_scan() -> Path | None:
    """Prefer the checked-in pilot scans, with an environment override."""
    configured = os.environ.get("DGTCAV_TEST_SCAN")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"DGTCAV_TEST_SCAN does not exist: {candidate}")
        return candidate

    candidates = (
        REPO_ROOT / "data/raw_scans/003_S_4152/Sag_IR-SPGR.nii.gz",
        REPO_ROOT / "data/raw_scans/941_S_6422/Accelerated_Sagittal_MPRAGE.nii.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raw_scans = REPO_ROOT / "data/raw_scans"
    return next(iter(sorted(raw_scans.rglob("*.nii.gz"))), None)


class TestRealScanIntegration(unittest.TestCase):
    def setUp(self):
        self.input_scan = find_real_scan()
        if self.input_scan is None:
            self.skipTest(
                "No real scan found. Set DGTCAV_TEST_SCAN or add a NIfTI under data/raw_scans/."
            )

    def test_end_to_end_real_scan(self):
        """Verify output grid, spacing, and mask for a real T1 scan."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "preprocessed.nii.gz"
            mask_file = Path(tmp_dir) / "preprocessed_mask.nii.gz"

            config = PipelineConfig(
                template_path=REPO_ROOT / "templates/mni152.nii.gz",
                target_shape=(128, 128, 128),
            )

            result = preprocess_mri_scan(
                str(self.input_scan),
                output_path=str(out_file),
                mask_output_path=str(mask_file),
                config=config,
            )

            # 1. Output files exist
            self.assertTrue(out_file.is_file())
            self.assertTrue(mask_file.is_file())

            # 2. Dimensions and physical extent based voxel spacing
            img = nib.load(str(out_file))
            mask = nib.load(str(mask_file))
            self.assertEqual(img.shape, (128, 128, 128))
            self.assertEqual(mask.shape, (128, 128, 128))
            template = nib.load(str(config.template_path))
            expected_spacing = np.asarray(template.header.get_zooms()[:3]) * np.asarray(
                template.shape[:3]
            ) / 128.0
            self.assertTrue(
                np.allclose(img.header.get_zooms()[:3], expected_spacing, atol=0.01)
            )

            # 3. Completion and inspection mask
            mask_arr = np.asarray(mask.dataobj) > 0
            img_arr = np.asarray(img.dataobj, dtype=np.float32)
            self.assertTrue(mask_arr.any())
            self.assertTrue(np.isfinite(img_arr).all())


if __name__ == "__main__":
    unittest.main()
