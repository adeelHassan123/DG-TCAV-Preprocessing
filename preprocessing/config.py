"""Typed configuration for reproducible structural MRI preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    template_path: Path = Path("templates/mni152_t1_2mm.nii.gz")
    template_resolution_mm: float = 2.0
    target_shape: Tuple[int, int, int] = (91, 109, 91)
    registration: str = "affine"
    registration_iterations: int = 200
    metric_bins: int = 50
    shrink_factors: Tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas: Tuple[float, ...] = (2.0, 1.0, 0.0)
    n4_shrink_factor: int = 4
    n4_iterations: Tuple[int, ...] = (50, 40, 30)
    n4_convergence_threshold: float = 0.001
    brain_extractor: str = "hd-bet"
    device: str = "auto"
    disable_tta: bool = False
    normalization: str = "zscore"
    random_seed: int = 2026
    output_spacing_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0)


DEFAULT_CONFIG = PipelineConfig()
