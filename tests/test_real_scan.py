"""Integration test running end-to-end preprocessing on real ADNI scan 002_S_2010."""

import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

from preprocessing.config import PipelineConfig
from preprocessing.pipeline import preprocess_mri_scan


class TestRealScanIntegration(unittest.TestCase):
    def setUp(self):
        self.input_scan = Path("data/test/002_S_2010/MPRAGE.nii.gz")
        if not self.input_scan.is_file():
            self.skipTest(f"Sample scan not found: {self.input_scan}")

    def test_end_to_end_002_s_2010(self):
        """Verify full pipeline produces an anatomically plausible 2mm MNI volume."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "preprocessed.nii.gz"
            mask_file = Path(tmp_dir) / "preprocessed_mask.nii.gz"

            config = PipelineConfig(
                template_path=Path("templates/mni152_t1_2mm.nii.gz"),
                brain_extractor="hd-bet",
                device="cpu",
                disable_tta=True,
                registration="affine",
                normalization="zscore",
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

            # 2. Dimensions and voxel grid
            img = nib.load(str(out_file))
            mask = nib.load(str(mask_file))
            self.assertEqual(img.shape, (91, 109, 91))
            self.assertEqual(mask.shape, (91, 109, 91))
            self.assertTrue(np.allclose(img.header.get_zooms()[:3], (2.0, 2.0, 2.0), atol=0.01))

            # 3. Biological Brain Volume QA Criteria
            # Human adult brain is 1.1L - 1.5L; must NOT explode to 4.6L
            self.assertTrue(result.qa_passed, f"Biological QA failed: {result.qa_message}")
            self.assertTrue(
                1.0 <= result.brain_volume_liters <= 1.8,
                f"Implausible brain volume: {result.brain_volume_liters:.2f}L",
            )
            self.assertTrue(
                125_000 <= result.brain_voxels <= 225_000,
                f"Implausible voxel count: {result.brain_voxels}",
            )

            # 4. Strict Background Zeroing
            mask_arr = np.asarray(mask.dataobj) > 0
            img_arr = np.asarray(img.dataobj, dtype=np.float32)
            self.assertTrue((img_arr[~mask_arr] == 0.0).all())

            # 5. Foreground Z-Score Normalization
            brain_vals = img_arr[mask_arr]
            self.assertTrue(np.isclose(brain_vals.mean(), 0.0, atol=1e-3))
            self.assertTrue(np.isclose(brain_vals.std(), 1.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
