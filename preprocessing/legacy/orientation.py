"""NIfTI Validation and Canonical RAS+ Orientation Module.
======================================================
This module acts as the "quality-control border check" for 3D MRI scans.
Before any heavy processing (skull stripping, bias correction, registration) begins,
this file ensures:
  1. The file is a valid, uncorrupted 3D anatomical volume.
  2. The head is reoriented into standard RAS+ coordinate space
     (Left->Right, Back->Front, Bottom->Top).
  3. Spatial affine metadata (qform, sform) are synchronized and mathematically valid.
"""

from pathlib import Path
import nibabel as nib
import numpy as np


def load_canonical_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a 3D NIfTI volume, reorient it to canonical RAS+, and repair spatial headers.

    Parameters
    ----------
    path : str or Path
        The file system path to the input NIfTI image (.nii or .nii.gz).

    Returns
    -------
    nib.Nifti1Image
        A validated, finite, 3D float32 image oriented in canonical RAS+ space.

    Raises
    ------
    ValueError
        If the image is not 3D, is empty, contains NaN/Inf values, or has an
        invalid/singular affine transformation matrix.
    """
    # -------------------------------------------------------------------------
    # Load the NIfTI file from disk into memory using nibabel.
    # We wrap `path` in `str()` to support both standard strings and Path objects.
    # -------------------------------------------------------------------------
    image = nib.load(str(path))

    # -------------------------------------------------------------------------
    # Verify that the scan is strictly a 3D volumetric image (X, Y, Z).
    # If the file is a 2D slice or a 4D fMRI time series, stop immediately with
    # a clear error so downstream 3D deep learning models do not crash later.
    # -------------------------------------------------------------------------
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D T1 volume, received shape {image.shape}")

    # -------------------------------------------------------------------------
    # Reorient the 3D grid into canonical RAS+ orientation.
    # Different scanners acquire brains from different angles (head-first, feet-first,
    # tilted). This flips/permutes axes so that:
    #   Axis 0 = Left to Right (R)
    #   Axis 1 = Posterior to Anterior / Back to Front (A)
    #   Axis 2 = Inferior to Superior / Bottom to Top (S)
    # -------------------------------------------------------------------------
    canonical = nib.as_closest_canonical(image)

    # -------------------------------------------------------------------------
    # Extract voxel intensity values as 32-bit floating point numbers (float32).
    # Deep learning frameworks (PyTorch) require float32 to maximize GPU memory
    # efficiency and computational speed while maintaining high numerical precision.
    # -------------------------------------------------------------------------
    data = np.asarray(canonical.dataobj, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Check for data corruption (empty scans or broken numbers).
    # `not np.isfinite(data).all()` detects NaN (Not a Number) or Inf (Infinity).
    # `not np.any(data)` detects an empty image that is pure zeros (completely black).
    # Passing NaN into a neural network permanently ruins all model weights.
    # -------------------------------------------------------------------------
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError("Input volume is empty or contains NaN/Inf values")

    # -------------------------------------------------------------------------
    # Extract the 4x4 Affine Matrix as double-precision (float64).
    # The affine matrix maps voxel indices (i, j, k) to physical space (mm).
    # It stores voxel sizes (e.g., 1.0mm x 1.0mm x 1.2mm) and physical head coordinates.
    # -------------------------------------------------------------------------
    affine = np.asarray(canonical.affine, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Mathematically validate the Affine Matrix.
    # We compute the matrix determinant: `np.linalg.det(affine[:3, :3])`.
    # The determinant represents the physical volume of a voxel. If it is 0 or
    # close to 0 (< 1e-8), the coordinate system has collapsed (e.g., flattened),
    # meaning the spatial geometry is invalid and cannot be registered to MNI space.
    # -------------------------------------------------------------------------
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("Input NIfTI has an invalid spatial affine")

    # -------------------------------------------------------------------------
    # Make a safe copy of the NIfTI header metadata.
    # This allows us to modify header properties without altering original cache.
    # -------------------------------------------------------------------------
    header = canonical.header.copy()

    # -------------------------------------------------------------------------
    # Explicitly record data type as float32 in the header metadata.
    # -------------------------------------------------------------------------
    header.set_data_dtype(np.float32)

    # -------------------------------------------------------------------------
    # Reset slope and intercept to 1.0 and 0.0.
    # Older MRI formats used formula: Real_Value = (Voxel * slope) + intercept.
    # Setting slope=1.0 and intercept=0.0 prevents unintentional automatic intensity scaling.
    # -------------------------------------------------------------------------
    header.set_slope_inter(1.0, 0.0)

    # -------------------------------------------------------------------------
    # Package the cleaned 3D array, validated affine, and updated header
    # into a clean, new NIfTI-1 image object.
    # -------------------------------------------------------------------------
    output = nib.Nifti1Image(data, affine, header=header)
    output.header.set_data_dtype(np.float32)
    output.header.set_slope_inter(1.0, 0.0)
    output.set_qform(affine, code=1)
    output.set_sform(affine, code=1)

    # -------------------------------------------------------------------------
    # Return the validated, canonical RAS+ NIfTI volume, ready for
    # N4 Bias Field Correction, Skull Stripping, and MNI-152 Registration.
    # -------------------------------------------------------------------------
    return output
