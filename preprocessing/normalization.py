"""Intensity Normalization Module for Structural T1 MRI.
======================================================
This module provides validated, deterministic intensity normalization functions
for 3D MRI brain volumes. In accordance with neuroimaging standards, normalization
statistics (mean, standard deviation, min, max) are computed strictly over
voxels belonging to the intracranial brain mask, while background non-brain
voxels are anchored strictly to 0.0.
"""

from typing import Tuple
import numpy as np


def normalize_brain_volume(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = "zscore"
) -> Tuple[np.ndarray, float, float]:
    """Normalize a 3D brain image within the provided binary brain mask.

    Parameters
    ----------
    image : np.ndarray
        3D float32 image volume.
    mask : np.ndarray
        3D boolean or binary integer mask indicating brain parenchyma (1=brain, 0=background).
    mode : str, default="zscore"
        Normalization strategy:
          - "zscore": (I - mean) / std strictly across the brain foreground.
          - "minmax": (I - min) / (max - min) rescaled to [0, 1] across the brain foreground.

    Returns
    -------
    normalized : np.ndarray
        3D float32 normalized image with non-brain voxels set to 0.0.
    stat_1 : float
        Mean (for zscore) or Min (for minmax) of the foreground brain voxels.
    stat_2 : float
        Std (for zscore) or Range (max - min for minmax) of the foreground brain voxels.

    Raises
    ------
    ValueError
        If the mask is empty, has degenerate intensity variance, or mode is unsupported.
    """
    boolean_mask = np.asarray(mask, dtype=bool)
    if image.shape != boolean_mask.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match mask shape {boolean_mask.shape}"
        )

    values = image[boolean_mask]
    if values.size == 0 or not np.any(boolean_mask):
        raise ValueError("Cannot normalize an empty brain mask (zero foreground voxels).")

    if not np.isfinite(values).all():
        raise ValueError("Brain voxels contain NaN or Inf values prior to normalization.")

    if mode == "zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(std) or std < 1e-8:
            raise ValueError(
                f"Degenerate intensity variance inside brain mask: std={std:.2e}"
            )
        result = (image - mean) / std
        stat_1, stat_2 = mean, std

    elif mode == "minmax":
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        vrange = vmax - vmin
        if not np.isfinite(vrange) or vrange < 1e-8:
            raise ValueError(
                f"Degenerate intensity range inside brain mask: range={vrange:.2e}"
            )
        result = (image - vmin) / vrange
        stat_1, stat_2 = vmin, vrange

    else:
        raise ValueError(
            f"Unsupported normalization mode '{mode}'. Choose 'zscore' or 'minmax'."
        )

    normalized = np.asarray(result, dtype=np.float32)
    # Strictly zero out non-brain voxels
    normalized[~boolean_mask] = 0.0
    return normalized, stat_1, stat_2
