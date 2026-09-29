"""Unit tests for preprocessing pipeline using synthetic 3D phantoms.
Zero external test framework dependencies (uses standard library unittest).
"""

import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

from preprocessing.normalization import normalize_brain_volume
from preprocessing.orientation import load_canonical_nifti
from preprocessing.registration import register_to_mni
from preprocessing.pipeline import validate_brain_mask


class TestPipelineSynthetic(unittest.TestCase):
    def test_normalization_zscore_and_minmax(self):
        """Verify foreground normalization and strict background zeroing."""
        data = np.ones((20, 20, 20), dtype=np.float32) * 50.0
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        rng = np.random.default_rng(42)
        data[mask] = rng.normal(loc=100.0, scale=15.0, size=mask.sum())

        # Z-score test
        z_norm, mean, std = normalize_brain_volume(data, mask, mode="zscore")
        self.assertTrue(np.isclose(z_norm[mask].mean(), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(z_norm[mask].std(), 1.0, atol=1e-5))
        self.assertTrue((z_norm[~mask] == 0.0).all())

        # Min-Max test
        mm_norm, vmin, vrange = normalize_brain_volume(data, mask, mode="minmax")
        self.assertTrue(np.isclose(mm_norm[mask].min(), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(mm_norm[mask].max(), 1.0, atol=1e-5))
        self.assertTrue((mm_norm[~mask] == 0.0).all())

    def test_orientation_canonical_ras(self):
        """Verify canonical RAS reorientation and header verification."""
        shape = (40, 50, 40)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        data = np.ones(shape, dtype=np.float32) * 100.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_volume.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), str(file_path))
            canonical = load_canonical_nifti(file_path)
            self.assertEqual(len(canonical.shape), 3)
            self.assertTrue(np.isfinite(canonical.get_fdata()).all())
            self.assertEqual(canonical.header.get_slope_inter(), (1.0, 0.0))

    def test_biological_qa_gate(self):
        """Verify that oversized or boundary-touching masks are strictly rejected."""
        shape = (91, 109, 91)
        # 1. Valid mask (~1.2 Liters)
        valid_mask = np.zeros(shape, dtype=bool)
        valid_mask[20:70, 25:85, 20:70] = True  # 50 * 60 * 50 = 150,000 voxels = 1.20L
        passed, msg, vol, vox, _ = validate_brain_mask(valid_mask)
        self.assertTrue(passed)
        self.assertEqual(msg, "VALID")
        self.assertTrue(1.0 <= vol <= 1.5)

        # 2. Oversized mask (4.6 Liters -> mask explosion failure)
        oversized_mask = np.ones(shape, dtype=bool)
        passed, msg, vol, vox, _ = validate_brain_mask(oversized_mask)
        self.assertFalse(passed)
        self.assertIn("OVERSIZED_MASK", msg)

        # 3. Boundary touching mask
        border_mask = np.zeros(shape, dtype=bool)
        border_mask[0:50, 25:85, 20:70] = True  # touches x=0
        passed, msg, vol, vox, _ = validate_brain_mask(border_mask)
        self.assertFalse(passed)
        self.assertIn("BOUNDARY_COLLISION", msg)

    def test_registration_synthetic_alignment(self):
        """Verify registration converges without volume explosion on a synthetic target."""
        shape = (91, 109, 91)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        z, y, x = np.ogrid[:shape[2], :shape[1], :shape[0]]

        fixed_mask = (((x - 45) / 25) ** 2 + ((y - 54) / 30) ** 2 + ((z - 45) / 25) ** 2) <= 1.0
        fixed_data = np.zeros(shape, dtype=np.float32)
        fixed_data[fixed_mask] = 100.0

        moving_mask = (((x - 48) / 25) ** 2 + ((y - 51) / 30) ** 2 + ((z - 43) / 25) ** 2) <= 1.0
        moving_data = np.zeros(shape, dtype=np.float32)
        moving_data[moving_mask] = 100.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixed_path = tmp_path / "fixed_brain.nii.gz"
            nib.save(nib.Nifti1Image(fixed_data, affine), str(fixed_path))
            mask_path = tmp_path / "fixed_mask.nii.gz"
            nib.save(nib.Nifti1Image(fixed_mask.astype(np.uint8), affine), str(mask_path))
            moving_path = tmp_path / "moving_brain.nii.gz"
            nib.save(nib.Nifti1Image(moving_data, affine), str(moving_path))

            registered, reg_mask, _ = register_to_mni(
                moving_path,
                moving_mask,
                fixed_path,
                reference_mask_path=mask_path,
                registration="affine",
                iterations=50,
            )

            self.assertEqual(registered.shape, shape)
            reg_voxels = np.count_nonzero(reg_mask)
            fixed_voxels = np.count_nonzero(fixed_mask)
            ratio = reg_voxels / fixed_voxels
            self.assertTrue(0.80 <= ratio <= 1.20, f"Registration ratio={ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
