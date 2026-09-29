"""Headless tri-planar preprocessing quality-control figures.
==========================================================
Displays 3D MRI volumes adhering to radiological viewing conventions:
  - Axial Plane: Nose pointing UP (Anterior), Left/Right symmetrical.
  - Coronal Plane: Vertex of head pointing UP (Superior), Brainstem pointing DOWN.
  - Sagittal Plane: Vertex of head pointing UP (Superior), Brainstem pointing DOWN.
  - Background Rendering: Zero-valued exterior voxels render as deep pitch-black.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_ortho_slices(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract standard upright orthoview slices from a 3D canonical RAS+ volume.

    RAS+ coordinate axes:
      Axis 0 (X): Left -> Right
      Axis 1 (Y): Posterior -> Anterior (Back -> Front/Nose)
      Axis 2 (Z): Inferior -> Superior (Neck -> Top of Head)

    Returns
    -------
    axial : np.ndarray (Y, X)
    coronal : np.ndarray (Z, X)
    sagittal : np.ndarray (Z, Y)
    """
    i, j, k = (n // 2 for n in volume.shape)
    # Axial (fixed Z=k): row=Y (pointing UP to Nose), col=X (Left -> Right)
    axial = volume[:, :, k].T
    # Coronal (fixed Y=j): row=Z (pointing UP to Vertex), col=X (Left -> Right)
    coronal = volume[:, j, :].T
    # Sagittal (fixed X=i): row=Z (pointing UP to Vertex), col=Y (Back -> Front)
    sagittal = volume[i, :, :].T
    return axial, coronal, sagittal


def generate_5step_progression_figure(
    subject_id: str,
    steps_dict: Dict[str, np.ndarray],
    output_png_path: str,
    raw_shape: tuple,
    final_shape: tuple = (91, 109, 91),
) -> None:
    """Save raw, stripped, registered, and normalized overlays in axial/coronal/sagittal planes."""
    panels = [
        ("0_raw", f"1. Raw Input ({raw_shape})", None),
        ("3_stripped", "2. Deep Learning Brain Extraction", "3_mask"),
        ("4_registered", f"3. Affine Registered to MNI152 ({final_shape})", "4_mask"),
        ("5_normalized", "4. Brain-Mask Intensity Normalization", "4_mask"),
    ]

    fig, axes = plt.subplots(
        len(panels), 3, figsize=(13, 15), dpi=160, facecolor="#0d1117"
    )
    fig.suptitle(
        f"Subject: {subject_id} | Preprocessing QA Report | Final Grid: {final_shape}",
        fontsize=14,
        color="white",
        fontweight="bold",
        y=0.99,
    )
    plane_names = ("Axial (Nose UP)", "Coronal (Head UP)", "Sagittal (Head UP)")

    for row, (key, title, mask_key) in enumerate(panels):
        volume = steps_dict.get(key)
        if volume is None:
            continue

        views = _extract_ortho_slices(volume)
        mask_vol = steps_dict.get(mask_key) if mask_key else None
        mask_views = _extract_ortho_slices(mask_vol) if mask_vol is not None else None

        # Compute robust foreground contrast percentiles
        if mask_vol is not None and np.any(mask_vol):
            fg = volume[mask_vol > 0]
            vmin = float(np.percentile(fg, 0.5)) if fg.size else float(volume.min())
            vmax = float(np.percentile(fg, 99.5)) if fg.size else float(volume.max())
        else:
            vmin = float(np.percentile(volume, 1.0))
            vmax = float(np.percentile(volume, 99.0))

        for col, view in enumerate(views):
            ax = axes[row, col]
            ax.set_facecolor("black")

            # For masked volumes, set background strictly to vmin so it renders pitch-black
            if mask_views is not None and mask_views[col] is not None:
                mask_slice = mask_views[col] > 0
                display_slice = np.where(mask_slice, view, vmin)
            else:
                display_slice = view

            ax.imshow(
                display_slice,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            # Draw crisp cyan contour of brain mask
            if mask_views is not None and np.any(mask_views[col]):
                ax.contour(
                    mask_views[col] > 0,
                    levels=[0.5],
                    colors=["#00e5ff"],
                    linewidths=0.9,
                )

            ax.set_title(
                f"{title}\n{plane_names[col]}",
                fontsize=9,
                color="#c9d1d9",
                pad=6,
            )
            ax.axis("off")

    path = Path(output_png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
