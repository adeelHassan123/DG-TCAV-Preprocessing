"""End-to-end preprocessing for 3D T1 MRI scans."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from .config import DEFAULT_CONFIG, PipelineConfig
from .image_ops import (
    n4_bias_field_correction,
    normalize_zscore,
    registration,
    resize_image,
    sitk_to_xyz,
    skull_strip,
)


@dataclass
class PreprocessingResult:
    subject_id: str
    raw_shape: Tuple[int, int, int]
    final_shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    brain_mean: float
    brain_std: float
    brain_voxels: int
    brain_volume_liters: float
    elapsed_sec: float
    output_path: str
    mask_path: str
    step_volumes: Optional[Dict[str, np.ndarray]] = None
    step_spacings: Optional[Dict[str, Tuple[float, float, float]]] = None


def _array_mask(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0) > 0


def preprocess_mri_scan(
    input_path: str,
    output_path: Optional[str] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    collect_steps: bool = False,
    config: PipelineConfig = DEFAULT_CONFIG,
    mask_output_path: Optional[str] = None,
) -> PreprocessingResult:
    """Run orientation, bias correction, extraction, alignment, normalization, and resize.

    The brain mask guides alignment and is also saved as a sidecar for
    inspection. It is not used during intensity normalization.
    """
    started = time.perf_counter()
    source = Path(input_path)
    raw = sitk.ReadImage(str(source), sitk.sitkFloat32)
    if raw.GetDimension() != 3:
        raise ValueError(f"Expected a 3D MRI, received {raw.GetDimension()}D")
    raw_size = tuple(int(v) for v in raw.GetSize())
    steps: Dict[str, np.ndarray] = {}
    step_spacings: Dict[str, Tuple[float, float, float]] = {}
    if collect_steps:
        steps["0_raw"] = sitk_to_xyz(raw)
        step_spacings["0_raw"] = tuple(float(v) for v in raw.GetSpacing())

    oriented = sitk.DICOMOrient(raw, "RPS")
    if collect_steps:
        steps["1_rps"] = sitk_to_xyz(oriented)
        step_spacings["1_rps"] = tuple(float(v) for v in oriented.GetSpacing())

    corrected, _head_mask = n4_bias_field_correction(oriented)
    if collect_steps:
        steps["2_n4"] = sitk_to_xyz(corrected)
        steps["2_head_mask"] = _array_mask(_head_mask).astype(np.uint8)
        step_spacings["2_n4"] = tuple(float(v) for v in corrected.GetSpacing())
        step_spacings["2_head_mask"] = step_spacings["2_n4"]

    stripped, native_mask = skull_strip(corrected)
    if collect_steps:
        steps["3_stripped"] = sitk_to_xyz(stripped)
        steps["3_mask"] = _array_mask(native_mask).astype(np.uint8)
        step_spacings["3_stripped"] = tuple(float(v) for v in stripped.GetSpacing())
        step_spacings["3_mask"] = step_spacings["3_stripped"]

    aligned, transform, fixed = registration(
        stripped, config.template_path, moving_mask=native_mask
    )
    aligned_mask = sitk.Resample(
        native_mask,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    if collect_steps:
        steps["4_registered"] = sitk_to_xyz(aligned)
        steps["4_mask"] = _array_mask(aligned_mask).astype(np.uint8)
        step_spacings["4_registered"] = tuple(float(v) for v in aligned.GetSpacing())
        step_spacings["4_mask"] = step_spacings["4_registered"]

    normalized = normalize_zscore(aligned)
    if collect_steps:
        steps["5_normalized"] = sitk_to_xyz(normalized)
        step_spacings["5_normalized"] = tuple(float(v) for v in normalized.GetSpacing())

    final_size = tuple(target_shape or config.target_shape)
    resized = resize_image(normalized, final_size)
    resized_mask = resize_image(aligned_mask, final_size)
    resized_mask = sitk.Cast(resized_mask > 0.5, sitk.sitkUInt8)
    if collect_steps:
        steps["6_resized"] = sitk_to_xyz(resized)
        steps["6_mask"] = _array_mask(resized_mask).astype(np.uint8)
        step_spacings["6_resized"] = tuple(float(v) for v in resized.GetSpacing())
        step_spacings["6_mask"] = step_spacings["6_resized"]

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resized, str(output), True)
        mask_path = Path(mask_output_path) if mask_output_path else output.with_name(
            output.name.removesuffix(".nii.gz").removesuffix(".nii") + "_mask.nii.gz"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resized_mask, str(mask_path), True)
    else:
        output, mask_path = Path(""), Path("")

    arr = sitk_to_xyz(resized)
    zooms = tuple(float(v) for v in resized.GetSpacing())
    voxels = int(np.count_nonzero(_array_mask(resized_mask)))
    volume_l = float(voxels * np.prod(zooms) / 1e6)
    values = arr[np.isfinite(arr)]
    return PreprocessingResult(
        subject_id=source.name.removesuffix(".nii.gz").removesuffix(".nii"),
        raw_shape=raw_size,
        final_shape=tuple(int(v) for v in resized.GetSize()),
        voxel_spacing=zooms,
        brain_mean=float(values.mean()) if values.size else float("nan"),
        brain_std=float(values.std()) if values.size else float("nan"),
        brain_voxels=voxels,
        brain_volume_liters=volume_l,
        elapsed_sec=time.perf_counter() - started,
        output_path=str(output) if output_path else "",
        mask_path=str(mask_path) if output_path else "",
        step_volumes=steps if collect_steps else None,
        step_spacings=step_spacings if collect_steps else None,
    )
