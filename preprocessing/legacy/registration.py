"""Archived wrapper for affine MNI registration."""

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from ..image_ops import registration as _register_image, sitk_to_xyz


def register_to_mni(
    moving_path: str | Path,
    moving_mask: np.ndarray,
    template_path: str | Path,
    template_mask_path: Optional[str | Path] = None,
    registration: str = "affine",
    **_unused_options,
):
    """Register a scan to the supplied MNI template.

    The optional mask and optimizer arguments from the former 2 mm pipeline
    are accepted for call compatibility; they do not affect the image fit.
    """
    del template_mask_path, _unused_options
    if registration != "affine":
        raise ValueError("This archived wrapper supports affine registration only")

    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    aligned, transform, fixed = _register_image(moving, template_path)
    mask_array = np.asarray(moving_mask, dtype=np.uint8)
    expected_shape = tuple(int(v) for v in moving.GetSize())
    if mask_array.shape != expected_shape:
        raise ValueError(
            f"Moving mask shape {mask_array.shape} does not match image {expected_shape}"
        )
    mask_image = sitk.GetImageFromArray(mask_array.transpose(2, 1, 0))
    mask_image.CopyInformation(moving)
    aligned_mask = sitk.Resample(
        mask_image,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk_to_xyz(aligned), sitk_to_xyz(aligned_mask) > 0, fixed
