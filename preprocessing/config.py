"""Typed configuration for reproducible structural MRI preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for the preprocessing pipeline."""

    # MNI image used as the registration target.
    template_path: Path = Path("templates/mni152.nii.gz")
    target_shape: Tuple[int, int, int] = (128, 128, 128)


DEFAULT_CONFIG = PipelineConfig()
