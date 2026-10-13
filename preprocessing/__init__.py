"""Structural MRI preprocessing package."""

from .config import PipelineConfig
from .pipeline import PreprocessingResult, preprocess_mri_scan
from .image_ops import normalize_zscore, registration, resize_image, skull_strip

__all__ = [
    "PipelineConfig",
    "PreprocessingResult",
    "preprocess_mri_scan",
    "normalize_zscore",
    "registration",
    "resize_image",
    "skull_strip",
]
