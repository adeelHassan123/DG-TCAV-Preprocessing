"""End-to-end deterministic 3D T1 MRI preprocessing pipeline.
===========================================================
Stages:
  1. Canonical RAS+ reorientation and affine header repair.
  2. N4ITK bias field correction for RF shading artifacts.
  3. Deep learning skull-stripping (HD-BET or FreeSurfer SynthStrip).
  4. Two-stage stabilized Affine registration to MNI152 (2.0 mm).
  5. Foreground Z-score intensity normalization with non-brain zeroing.
  6. Biological QA validation (brain volume, bounding box, template Dice).
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .bias_correction import n4_correct
from .config import DEFAULT_CONFIG, PipelineConfig
from .normalization import normalize_brain_volume
from .orientation import load_canonical_nifti
from .registration import ensure_mni_template, register_to_mni
from .skull_stripping import extract_brain


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
    template_dice: float
    qa_passed: bool
    qa_message: str
    elapsed_sec: float
    output_path: str
    mask_path: str
    step_volumes: Optional[Dict[str, np.ndarray]] = None


def validate_brain_mask(
    mask: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    voxel_spacing_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    min_volume_liters: float = 0.90,
    max_volume_liters: float = 1.90,
    min_dice: float = 0.70,
) -> Tuple[bool, str, float, int, float]:
    """Verify that a registered brain mask adheres to human neuroanatomy bounds.

    Returns
    -------
    passed : bool
    message : str
    volume_liters : float
    voxel_count : int
    dice_score : float
    """
    boolean_mask = np.asarray(mask, dtype=bool)
    voxel_count = int(np.count_nonzero(boolean_mask))
    voxel_vol_mm3 = float(np.prod(voxel_spacing_mm))
    volume_liters = float(voxel_count * voxel_vol_mm3 / 1e6)

    # 1. Non-zero check
    if voxel_count == 0:
        return False, "EMPTY_MASK: Zero brain voxels detected", 0.0, 0, 0.0

    # 2. Plausible adult human brain volume check (normally 1.1L - 1.5L)
    if volume_liters < min_volume_liters:
        return (
            False,
            f"UNDERSIZED_MASK: {volume_liters:.2f}L < {min_volume_liters:.2f}L threshold",
            volume_liters,
            voxel_count,
            0.0,
        )
    if volume_liters > max_volume_liters:
        return (
            False,
            f"OVERSIZED_MASK: {volume_liters:.2f}L > {max_volume_liters:.2f}L (mask explosion)",
            volume_liters,
            voxel_count,
            0.0,
        )

    # 3. Bounding box boundary collision check (brain should not touch lateral or superior edges)
    touches_boundary = (
        boolean_mask[0, :, :].any()
        or boolean_mask[-1, :, :].any()
        or boolean_mask[:, 0, :].any()
        or boolean_mask[:, -1, :].any()
        or boolean_mask[:, :, -1].any()
    )
    if touches_boundary:
        return (
            False,
            "BOUNDARY_COLLISION: Brain mask touches lateral or superior field boundaries",
            volume_liters,
            voxel_count,
            0.0,
        )

    # 4. Dice overlap with standard MNI brain mask
    dice_score = 1.0
    if template_mask is not None:
        tmpl_bool = np.asarray(template_mask, dtype=bool)
        intersection = np.count_nonzero(boolean_mask & tmpl_bool)
        total = voxel_count + np.count_nonzero(tmpl_bool)
        dice_score = float(2.0 * intersection / total) if total > 0 else 0.0
        if dice_score < min_dice:
            return (
                False,
                f"LOW_DICE_OVERLAP: Dice {dice_score:.3f} < {min_dice:.2f} with MNI template",
                volume_liters,
                voxel_count,
                dice_score,
            )

    return True, "VALID", volume_liters, voxel_count, dice_score


def preprocess_mri_scan(
    input_path: str,
    output_path: Optional[str] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    collect_steps: bool = False,
    config: PipelineConfig = DEFAULT_CONFIG,
    mask_output_path: Optional[str] = None,
) -> PreprocessingResult:
    """Run full 5-stage deterministic 3D T1 MRI preprocessing pipeline."""
    started = time.perf_counter()
    source = Path(input_path)
    subject_id = source.name.removesuffix(".nii.gz").removesuffix(".nii")
    input_shape = tuple(int(v) for v in nib.load(str(source)).shape[:3])

    print(f"[{subject_id}] Stage 1/5: Loading & canonical RAS+ orientation.", flush=True)
    canonical = load_canonical_nifti(source)
    raw_shape = input_shape

    steps: Dict[str, np.ndarray] = {}
    if collect_steps:
        steps["0_raw"] = np.asarray(nib.load(str(source)).dataobj, dtype=np.float32)
        steps["1_ras"] = np.asarray(canonical.dataobj, dtype=np.float32).copy()

    expected_shape = tuple(target_shape or config.target_shape)
    template_path = ensure_mni_template(
        config.template_path,
        target_shape=expected_shape,
        spacing_mm=config.output_spacing_mm[0],
    )
    template_mask_path = template_path.with_name("mni152_t1_2mm_brain_mask.nii.gz")
    template_mask = (
        np.asarray(nib.load(str(template_mask_path)).dataobj) > 0
        if template_mask_path.is_file()
        else None
    )

    with tempfile.TemporaryDirectory(prefix=f"mri_{subject_id}_") as temp_name:
        temp = Path(temp_name)
        canonical_path = temp / "canonical_ras.nii.gz"
        n4_path = temp / "n4_corrected.nii.gz"
        nib.save(canonical, str(canonical_path))

        print(f"[{subject_id}] Stage 2/5: N4ITK bias field correction.", flush=True)
        n4_img = n4_correct(
            sitk.ReadImage(str(canonical_path), sitk.sitkFloat32),
            shrink_factor=config.n4_shrink_factor,
            iterations=config.n4_iterations,
            convergence_threshold=config.n4_convergence_threshold,
        )
        sitk.WriteImage(n4_img, str(n4_path), useCompression=True)
        n4_array = sitk.GetArrayFromImage(n4_img).transpose(2, 1, 0).astype(np.float32)
        if collect_steps:
            steps["2_n4"] = n4_array.copy()

        print(
            f"[{subject_id}] Stage 3/5: Deep learning skull-stripping ({config.brain_extractor}).",
            flush=True,
        )
        stripped, native_mask = extract_brain(
            n4_path,
            temp / "extract",
            extractor=config.brain_extractor,
            device=config.device,
            disable_tta=config.disable_tta,
        )
        if collect_steps:
            steps["3_stripped"] = stripped.copy()
            steps["3_mask"] = native_mask.astype(np.uint8)

        stripped_path = temp / "stripped.nii.gz"
        stripped_nifti = nib.Nifti1Image(stripped, canonical.affine, canonical.header)
        stripped_nifti.set_qform(canonical.affine, code=1)
        stripped_nifti.set_sform(canonical.affine, code=1)
        nib.save(stripped_nifti, str(stripped_path))

        print(f"[{subject_id}] Stage 4/5: Stabilized affine registration to MNI152.", flush=True)
        registered, registered_mask, fixed = register_to_mni(
            stripped_path,
            native_mask,
            template_path,
            reference_mask_path=template_mask_path,
            registration=config.registration,
            iterations=config.registration_iterations,
            metric_bins=config.metric_bins,
            shrink_factors=config.shrink_factors,
            smoothing_sigmas=config.smoothing_sigmas,
            random_seed=config.random_seed,
        )
        if registered.shape != expected_shape:
            raise RuntimeError(
                f"Registered volume has unexpected shape {registered.shape} != {expected_shape}"
            )

        print(
            f"[{subject_id}] Stage 5/5: Intensity normalization ({config.normalization}).",
            flush=True,
        )
        normalized, _, _ = normalize_brain_volume(
            registered, registered_mask, mode=config.normalization
        )
        brain_values = normalized[registered_mask]
        brain_mean = float(brain_values.mean())
        brain_std = float(brain_values.std())

        # Stage 6: Biological Quality Assurance validation
        qa_passed, qa_msg, brain_vol_l, brain_vox, dice = validate_brain_mask(
            registered_mask,
            template_mask=template_mask,
            voxel_spacing_mm=config.output_spacing_mm,
        )
        print(
            f"[{subject_id}] QA Status: {qa_msg} | Volume: {brain_vol_l:.2f}L ({brain_vox} voxels) | Dice: {dice:.3f}",
            flush=True,
        )

        if collect_steps:
            steps["4_registered"] = registered.copy()
            steps["4_mask"] = registered_mask.astype(np.uint8)
            steps["5_normalized"] = normalized.copy()

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            reference = nib.load(str(template_path))
            out_img = nib.Nifti1Image(
                normalized, reference.affine, reference.header.copy()
            )
            out_img.header.set_data_dtype(np.float32)
            out_img.header.set_slope_inter(1.0, 0.0)
            qform, qcode = reference.get_qform(coded=True)
            sform, scode = reference.get_sform(coded=True)
            out_img.set_qform(qform if qcode else reference.affine, code=int(qcode) or 4)
            out_img.set_sform(sform if scode else reference.affine, code=int(scode) or 4)
            nib.save(out_img, str(output))

            if mask_output_path:
                mask_output = Path(mask_output_path)
            elif output.name.endswith(".nii.gz"):
                mask_output = output.with_name(output.name[:-7] + "_mask.nii.gz")
            elif output.suffix == ".nii":
                mask_output = output.with_name(output.stem + "_mask.nii")
            else:
                mask_output = output.with_name(output.name + "_mask.nii.gz")
            mask_output.parent.mkdir(parents=True, exist_ok=True)

            mask_img = nib.Nifti1Image(
                registered_mask.astype(np.uint8), reference.affine, reference.header.copy()
            )
            mask_img.header.set_data_dtype(np.uint8)
            mask_img.header.set_slope_inter(1.0, 0.0)
            mask_img.set_qform(qform if qcode else reference.affine, code=int(qcode) or 4)
            mask_img.set_sform(sform if scode else reference.affine, code=int(scode) or 4)
            nib.save(mask_img, str(mask_output))
        else:
            output, mask_output = Path(""), Path("")

    return PreprocessingResult(
        subject_id=subject_id,
        raw_shape=raw_shape,
        final_shape=expected_shape,
        voxel_spacing=config.output_spacing_mm,
        brain_mean=brain_mean,
        brain_std=brain_std,
        brain_voxels=brain_vox,
        brain_volume_liters=brain_vol_l,
        template_dice=dice,
        qa_passed=qa_passed,
        qa_message=qa_msg,
        elapsed_sec=time.perf_counter() - started,
        output_path=str(output) if output_path else "",
        mask_path=str(mask_output) if output_path else "",
        step_volumes=steps if collect_steps else None,
    )
