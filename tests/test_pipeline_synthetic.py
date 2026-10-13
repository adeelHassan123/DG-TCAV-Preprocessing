"""Fast synthetic checks for MRI image operations."""

import unittest

import numpy as np
import SimpleITK as sitk

from preprocessing.image_ops import normalize_zscore, resize_image


class TestImageOperations(unittest.TestCase):
    def test_normalization_uses_complete_volume(self):
        values = np.arange(20 * 18 * 16, dtype=np.float32).reshape(16, 18, 20)
        image = sitk.GetImageFromArray(values)
        normalized = sitk.GetArrayFromImage(normalize_zscore(image))
        self.assertTrue(np.isclose(float(normalized.mean()), 0.0, atol=1e-6))
        # SimpleITK and NumPy can use slightly different variance conventions;
        # keep this a check for unit-scale normalization, not bit-level agreement.
        self.assertTrue(np.isclose(float(normalized.std()), 1.0, atol=1e-4))

    def test_resize_preserves_physical_extent(self):
        image = sitk.Image((20, 24, 18), sitk.sitkFloat32)
        image.SetSpacing((1.1, 1.3, 1.5))
        image.SetOrigin((-12.0, 8.0, 4.0))
        resized = resize_image(image, (128, 128, 128))
        expected_spacing = tuple(
            old_size * old_spacing / 128.0
            for old_size, old_spacing in zip(image.GetSize(), image.GetSpacing())
        )
        self.assertEqual(resized.GetSize(), (128, 128, 128))
        self.assertTrue(np.allclose(resized.GetSpacing(), expected_spacing))
        self.assertEqual(resized.GetOrigin(), image.GetOrigin())
        self.assertEqual(resized.GetDirection(), image.GetDirection())


if __name__ == "__main__":
    unittest.main()
