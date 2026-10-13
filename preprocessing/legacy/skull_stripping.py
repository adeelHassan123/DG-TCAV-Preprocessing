"""Archived wrapper for the DeepBrain extractor."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ..image_ops import sitk_to_xyz, skull_strip


def extract_brain(
    input_nifti: str | Path,
    work_dir: str | Path | None = None,
    extractor: str = "deepbrain",
    device: str = "auto",
    disable_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a NIfTI and return the extracted brain image and binary mask.

    ``work_dir``, ``device`` and ``disable_tta`` remain accepted for older
    callers, but this implementation does not use them.
    """
    del work_dir, device, disable_tta
    if extractor.lower() != "deepbrain":
        raise ValueError("This archived wrapper supports DeepBrain only")
    image = sitk.ReadImage(str(input_nifti), sitk.sitkFloat32)
    stripped, mask = skull_strip(image)
    return sitk_to_xyz(stripped), sitk_to_xyz(mask) > 0
