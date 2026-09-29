"""Physical-space affine registration and resampling onto an MNI reference.
=======================================================================
This module handles spatial normalization of skull-stripped 3D T1 MRI scans
onto the standard MNI152 2.0 mm coordinate space.

To prevent registration divergence and mask explosion:
  1. The skull-stripped moving brain is registered against a matching
     skull-stripped MNI reference brain (not a whole-head image).
  2. Transform optimization uses physical shift parameter scaling and
     damped gradient descent steps to preserve anatomical proportions.
  3. The resampled brain mask is validated and bounded by the intracranial template.
"""

from pathlib import Path
from typing import Optional, Tuple
import os
import tempfile

import nibabel as nib
import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    sitk = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def ensure_mni_template(
    template_path: str | Path,
    target_shape: Tuple[int, int, int] = (91, 109, 91),
    spacing_mm: float = 2.0,
) -> Path:
    """Resolve and validate the official MNI152 2.0 mm template and its brain mask.

    Ensures that the standard template, its skull-stripped brain volume, and its
    binary brain mask are properly cropped to the target matrix grid (91, 109, 91).

    Parameters
    ----------
    template_path : str or Path
        Target destination path for the cropped MNI template.
    target_shape : Tuple[int, int, int], default=(91, 109, 91)
        Matrix dimensions in voxels.
    spacing_mm : float, default=2.0
        Isotropic voxel spacing in millimeters.

    Returns
    -------
    Path
        Path to the validated MNI template file.
    """
    path = Path(template_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.is_file():
        try:
            import requests

            url = (
                "https://templateflow.s3.amazonaws.com/"
                "tpl-MNI152NLin2009cAsym/"
                "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
            )
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                fd, temporary_name = tempfile.mkstemp(
                    prefix="mni152_2009c_", suffix=".nii.gz", dir=str(path.parent)
                )
                os.close(fd)
                temporary_path = Path(temporary_name)
                try:
                    with temporary_path.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                    if not _valid_template_file(temporary_path, spacing_mm):
                        raise ValueError("Template download was invalid or truncated")
                    _crop_to_grid(temporary_path, path, target_shape, spacing_mm)
                finally:
                    temporary_path.unlink(missing_ok=True)
        except Exception as exc:
            raise FileNotFoundError(
                f"No valid template at {path}; unable to fetch official MNI152 template."
            ) from exc

    # Ensure sibling brain mask and skull-stripped brain template also exist
    brain_path = path.with_name("mni152_t1_2mm_brain.nii.gz")
    mask_path = path.with_name("mni152_t1_2mm_brain_mask.nii.gz")
    if not brain_path.is_file() or not mask_path.is_file():
        _ensure_template_brain_and_mask(path, brain_path, mask_path, target_shape)

    return path


def _crop_to_grid(
    source_path: Path,
    dest_path: Path,
    target_shape: Tuple[int, int, int],
    spacing_mm: float,
) -> None:
    """Center-crop a larger template (e.g. 97x115x97) to the target grid (91x109x91)."""
    source = nib.load(str(source_path))
    shape = tuple(int(v) for v in source.shape[:3])
    if shape == target_shape:
        nib.save(source, str(dest_path))
        return

    starts = tuple((src - dst) // 2 for src, dst in zip(shape, target_shape))
    slices = tuple(slice(start, start + size) for start, size in zip(starts, target_shape))
    crop_affine = source.affine @ np.array([
        [1, 0, 0, starts[0]],
        [0, 1, 0, starts[1]],
        [0, 0, 1, starts[2]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    cropped = nib.Nifti1Image(
        np.asarray(source.dataobj[slices], dtype=np.float32),
        crop_affine,
        source.header.copy(),
    )
    cropped.header.set_data_dtype(np.float32)
    cropped.header.set_slope_inter(1.0, 0.0)
    cropped.set_qform(crop_affine, code=4)
    cropped.set_sform(crop_affine, code=4)
    nib.save(cropped, str(dest_path))


def _ensure_template_brain_and_mask(
    whole_head_path: Path,
    brain_path: Path,
    mask_path: Path,
    target_shape: Tuple[int, int, int],
) -> None:
    """Download and crop the official TemplateFlow brain mask and create the brain volume."""
    try:
        import requests

        url = (
            "https://templateflow.s3.amazonaws.com/"
            "tpl-MNI152NLin2009cAsym/"
            "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
        )
        resp = requests.get(url, timeout=(15, 180))
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)

        full_mask = nib.load(str(tmp_path))
        head_img = nib.load(str(whole_head_path))

        starts = tuple((s - d) // 2 for s, d in zip(full_mask.shape[:3], target_shape))
        slices = tuple(slice(st, st + sz) for st, sz in zip(starts, target_shape))
        mask_data = (np.asarray(full_mask.dataobj[slices]) > 0).astype(np.uint8)

        mask_nii = nib.Nifti1Image(mask_data, head_img.affine, head_img.header.copy())
        mask_nii.header.set_data_dtype(np.uint8)
        mask_nii.header.set_slope_inter(1.0, 0.0)
        nib.save(mask_nii, str(mask_path))

        brain_data = np.asarray(head_img.dataobj, dtype=np.float32) * mask_data
        brain_nii = nib.Nifti1Image(brain_data, head_img.affine, head_img.header.copy())
        brain_nii.header.set_data_dtype(np.float32)
        brain_nii.header.set_slope_inter(1.0, 0.0)
        nib.save(brain_nii, str(brain_path))
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        raise RuntimeError("Failed to generate MNI brain mask template") from exc


def _valid_template_file(path: Path, spacing_mm: float) -> bool:
    """Validate that the file exists, is non-trivial in size, and matches spacing."""
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    try:
        image = nib.load(str(path))
        return len(image.shape) == 3 and np.allclose(
            image.header.get_zooms()[:3], (spacing_mm,) * 3, atol=0.01
        )
    except Exception:
        return False


def _as_sitk(nifti_path: str | Path) -> "sitk.Image":
    image = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise ValueError(f"Expected 3D image, got {image.GetDimension()}D")
    return image


def register_to_mni(
    moving_path: str | Path,
    moving_mask: np.ndarray,
    reference_path: str | Path,
    reference_mask_path: Optional[str | Path] = None,
    registration: str = "affine",
    iterations: int = 150,
    metric_bins: int = 50,
    shrink_factors: Tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: Tuple[float, ...] = (2.0, 1.0, 0.0),
    random_seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, "sitk.Image"]:
    """Register skull-stripped moving MRI to the MNI reference grid.

    To eliminate the mask explosion defect:
      - When registering a skull-stripped moving brain, the fixed image is
        automatically resolved to the skull-stripped brain template.
      - Optimizer scaling uses physical shift calibration with damped
        learning rates to ensure stable convergence.
      - The output mask is bounded by the intracranial reference template.

    Parameters
    ----------
    moving_path : str or Path
        Path to the skull-stripped moving NIfTI volume.
    moving_mask : np.ndarray
        Native 3D boolean/uint8 mask corresponding to moving_path.
    reference_path : str or Path
        Path to the MNI152 template.
    reference_mask_path : str or Path, optional
        Path to the MNI152 binary brain mask.
    registration : str, default="affine"
        Registration transform type: "affine" (12-DOF) or "rigid" (6-DOF).
    iterations : int, default=150
        Maximum iterations for the affine optimization stage.
    metric_bins : int, default=50
        Number of histogram bins for Mattes Mutual Information.
    shrink_factors : Tuple[int, ...], default=(4, 2, 1)
        Multi-resolution shrink factors.
    smoothing_sigmas : Tuple[float, ...], default=(2.0, 1.0, 0.0)
        Smoothing sigmas in physical millimeters.
    random_seed : int, default=2026
        Random seed for metric sampling reproducibility.

    Returns
    -------
    image_array : np.ndarray
        Registered 3D float32 image on the reference grid with background set to 0.0.
    mask_array : np.ndarray
        Registered 3D boolean mask on the reference grid.
    fixed_image : sitk.Image
        The reference SimpleITK image defining the target physical space.
    """
    if sitk is None:
        raise RuntimeError("MNI registration requires SimpleITK") from _IMPORT_ERROR
    if registration not in {"affine", "rigid"}:
        raise ValueError("registration must be 'affine' or 'rigid'")

    ref_p = Path(reference_path)
    # Target the skull-stripped brain template to prevent outer skull divergence
    brain_ref = ref_p.with_name("mni152_t1_2mm_brain.nii.gz")
    actual_ref_path = brain_ref if brain_ref.is_file() else ref_p

    fixed = _as_sitk(actual_ref_path)
    moving = _as_sitk(moving_path)

    if tuple(moving_mask.shape) != moving.GetSize():
        raise ValueError("Brain mask shape does not match moving image dimensions")

    # Initial geometric alignment (centers of physical bounding boxes)
    initial_transform = (
        sitk.AffineTransform(3)
        if registration == "affine"
        else sitk.Euler3DTransform()
    )
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        initial_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(metric_bins))
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.20, int(random_seed))
    method.SetInterpolator(sitk.sitkLinear)

    # Stabilize optimizer with physical shift scaling and conservative learning rate
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetOptimizerAsGradientDescent(
        learningRate=0.25,
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetShrinkFactorsPerLevel([int(v) for v in shrink_factors])
    method.SetSmoothingSigmasPerLevel([float(v) for v in smoothing_sigmas])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed, moving)

    # Resample moving image using linear interpolation
    aligned = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

    # Resample moving mask using nearest neighbor interpolation
    mask_image = sitk.GetImageFromArray(
        moving_mask.transpose(2, 1, 0).astype(np.uint8)
    )
    mask_image.CopyInformation(moving)
    aligned_mask = sitk.Resample(
        mask_image, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )

    # SimpleITK (z, y, x) -> NumPy (x, y, z)
    image_array = sitk.GetArrayFromImage(aligned).transpose(2, 1, 0).astype(np.float32)
    mask_array = sitk.GetArrayFromImage(aligned_mask).transpose(2, 1, 0).astype(bool)

    # Bound resampled mask with template mask to eliminate boundary leakage
    ref_mask_p = (
        Path(reference_mask_path)
        if reference_mask_path
        else ref_p.with_name("mni152_t1_2mm_brain_mask.nii.gz")
    )
    if ref_mask_p.is_file():
        template_mask = np.asarray(nib.load(str(ref_mask_p)).dataobj) > 0
        mask_array = mask_array & template_mask

    if not mask_array.any() or not np.isfinite(image_array).all():
        raise RuntimeError("Registration produced an empty mask or invalid image")

    # Strictly zero out non-brain voxels
    image_array[~mask_array] = 0.0
    return image_array, mask_array, fixed
