"""Image operations used by the MRI preprocessing pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REGISTRATION_SAMPLING_SEED = 42

def sitk_to_xyz(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK's z,y,x array convention to x,y,z for plotting/storage."""
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.float32)


def n4_bias_field_correction(
    image_or_path: str | Path | sitk.Image, shrink_factor: int = 4
):
    """Run N4 bias correction and return the Li foreground mask.

    An in-memory image is assumed to already be in RPS orientation. Paths are
    read and oriented here so callers can use the helper independently.
    """
    if shrink_factor < 1:
        raise ValueError("shrink_factor must be at least 1")
    if isinstance(image_or_path, sitk.Image):
        raw_img = sitk.Cast(image_or_path, sitk.sitkFloat32)
    else:
        raw_img = sitk.ReadImage(str(image_or_path), sitk.sitkFloat32)
        raw_img = sitk.DICOMOrient(raw_img, "RPS")
    if raw_img.GetDimension() != 3:
        raise ValueError(f"Expected 3D MRI, got {raw_img.GetDimension()}D")
    rescaled_img = sitk.RescaleIntensity(raw_img, 0, 255)
    head_mask = sitk.LiThreshold(rescaled_img, 0, 1)
    factors = [int(shrink_factor)] * raw_img.GetDimension()
    small_img = sitk.Shrink(raw_img, factors)
    small_mask = sitk.Shrink(head_mask, factors)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(small_img, small_mask)
    log_field = corrector.GetLogBiasFieldAsImage(raw_img)
    corrected = raw_img / sitk.Exp(log_field)
    return sitk.Cast(corrected, sitk.sitkFloat32), head_mask


@lru_cache(maxsize=1)
def _deepbrain_extractor():
    # DeepBrain expects these TensorFlow 1.x names.
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1

    tf.Session = tf1.Session
    tf.gfile = tf1.gfile
    tf.GraphDef = tf1.GraphDef
    tf1.disable_v2_behavior()
    from deepbrain import Extractor

    return Extractor()


def skull_strip(sitk_image: sitk.Image):
    """Run the same DeepBrain orientation conversion and >0.3 threshold."""
    data = sitk.GetArrayFromImage(sitk_image)
    model_input = data.transpose(2, 1, 0)
    probability = np.asarray(_deepbrain_extractor().run(model_input))
    if probability.shape != model_input.shape:
        raise RuntimeError(
            f"DeepBrain returned shape {probability.shape}; expected {model_input.shape}"
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("DeepBrain returned NaN or infinite probabilities")
    mask_xyz = probability > 0.3
    mask_zyx = mask_xyz.transpose(2, 1, 0)
    stripped = sitk.GetImageFromArray(data * mask_zyx)
    stripped.CopyInformation(sitk_image)
    mask = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask.CopyInformation(sitk_image)
    return stripped, mask


def _largest_foreground_component(mask: sitk.Image) -> sitk.Image:
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(mask))
    return sitk.Cast(labels == 1, sitk.sitkUInt8)


def _mask_centroid(mask: sitk.Image) -> tuple[float, float, float]:
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask)
    if not statistics.HasLabel(1):
        raise RuntimeError("Cannot initialize registration from an empty brain mask")
    return tuple(float(value) for value in statistics.GetCentroid(1))


def _registration_stage(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: sitk.Transform,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    iterations: int,
) -> sitk.Transform:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.1, seed=REGISTRATION_SAMPLING_SEED)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetOptimizerAsGradientDescent(
        learningRate=0.1, numberOfIterations=iterations
    )
    method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    # Keep the concrete transform type. With inPlace=False SimpleITK wraps
    # the result in a CompositeTransform, which does not expose Euler/Affine
    # accessors such as GetCenter() used by the affine refinement stage.
    method.SetInitialTransform(initial_transform, inPlace=True)
    method.SetInterpolator(sitk.sitkLinear)
    method.Execute(fixed, moving)
    return initial_transform


def _outside_template_fraction(
    moving_mask: sitk.Image,
    transform: sitk.Transform,
    fixed: sitk.Image,
    padding_voxels: int = 32,
) -> float:
    """Measure extracted-mask voxels that land outside the fixed template FOV."""
    fixed_size = fixed.GetSize()
    padded_size = tuple(size + 2 * padding_voxels for size in fixed_size)
    padded = sitk.Image(padded_size, sitk.sitkUInt8)
    padded.SetSpacing(fixed.GetSpacing())
    padded.SetDirection(fixed.GetDirection())
    padded.SetOrigin(
        fixed.TransformIndexToPhysicalPoint((-padding_voxels,) * fixed.GetDimension())
    )
    mapped = sitk.Resample(
        moving_mask, padded, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    mapped_array = sitk.GetArrayViewFromImage(mapped)
    total = int(np.count_nonzero(mapped_array))
    if total == 0:
        raise RuntimeError("Registration mapped no extracted-brain voxels to the template")
    inner = sitk.RegionOfInterest(
        mapped, fixed_size, [padding_voxels] * fixed.GetDimension()
    )
    inside = int(np.count_nonzero(sitk.GetArrayViewFromImage(inner)))
    return max(0, total - inside) / total


def registration(
    moving_sitk_img: sitk.Image,
    mni_template_path: str | Path,
    moving_mask: sitk.Image | None = None,
    max_outside_fraction: float = 0.03,
):
    """Register a brain with masked rigid alignment followed by affine refinement.

    Brain masks keep background from driving the mutual-information fit. The
    moving mask is checked against an expanded template grid so a fit that
    crops more than 3% of extracted brain tissue is rejected before output.
    """
    fixed_img = sitk.ReadImage(str(mni_template_path), sitk.sitkFloat32)
    moving_img = sitk.Cast(moving_sitk_img, sitk.sitkFloat32)
    fixed_mask = _largest_foreground_component(
        sitk.OtsuThreshold(fixed_img, 0, 1, 200)
    )
    if moving_mask is None:
        moving_mask = sitk.OtsuThreshold(moving_img, 0, 1, 200)
        moving_mask = _largest_foreground_component(moving_mask)
    else:
        if moving_mask.GetSize() != moving_img.GetSize():
            raise ValueError("Moving brain mask and image sizes do not match")
        moving_mask = sitk.Cast(moving_mask > 0, sitk.sitkUInt8)

    fixed_center = _mask_centroid(fixed_mask)
    moving_center = _mask_centroid(moving_mask)
    initial = sitk.Euler3DTransform()
    initial.SetCenter(fixed_center)
    initial.SetTranslation(
        tuple(moving - fixed for moving, fixed in zip(moving_center, fixed_center))
    )

    rigid = _registration_stage(
        fixed_img, moving_img, initial, fixed_mask, moving_mask, iterations=100
    )
    affine_initial = sitk.AffineTransform(3)
    affine_initial.SetCenter(rigid.GetCenter())
    affine_initial.SetMatrix(rigid.GetMatrix())
    affine_initial.SetTranslation(rigid.GetTranslation())
    transform = _registration_stage(
        fixed_img,
        moving_img,
        affine_initial,
        fixed_mask,
        moving_mask,
        iterations=100,
    )
    outside_fraction = _outside_template_fraction(moving_mask, transform, fixed_img)
    if outside_fraction > max_outside_fraction:
        raise RuntimeError(
            "Affine registration would crop "
            f"{outside_fraction:.1%} of the extracted brain outside the MNI template "
            f"(limit {max_outside_fraction:.1%}); rejecting this scan."
        )
    aligned = sitk.Resample(
        moving_img, fixed_img, transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID()
    )
    return aligned, transform, fixed_img


def resize_image(image: sitk.Image, new_size=(128, 128, 128)) -> sitk.Image:
    """Resize the image while preserving its physical extent."""
    new_size = tuple(int(v) for v in new_size)
    if len(new_size) != 3 or any(v < 1 for v in new_size):
        raise ValueError("new_size must contain three positive dimensions")
    spacing = [old_n * old_s / new_n for old_n, old_s, new_n in zip(
        image.GetSize(), image.GetSpacing(), new_size
    )]
    output_grid = sitk.Image(new_size, image.GetPixelIDValue())
    output_grid.SetOrigin(image.GetOrigin())
    output_grid.SetDirection(image.GetDirection())
    output_grid.SetSpacing(spacing)
    return sitk.Resample(image, output_grid, sitk.Transform(), sitk.sitkLinear)


def normalize_zscore(image: sitk.Image) -> sitk.Image:
    """Normalize intensities across the complete image grid."""
    return sitk.Normalize(image)
