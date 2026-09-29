# DG-TCAV: Production 3D T1 MRI Preprocessing Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deterministic, reproducible, production-grade 3D T1-weighted brain MRI preprocessing pipeline designed for **Alzheimer's Disease classification**, **Concept Activation Vectors (TCAV)**, and **Generative Counterfactual Modeling (LDM)**.

---

## 1. Pipeline Architecture

Every 3D T1 scan undergoes a 5-stage standardized transformation sequence into standard stereotactic space (ICBM 2009c / MNI152) at isotropic 2.0 mm resolution (`91 × 109 × 91` matrix), verified by an automated biological Quality Assurance (QA) gate:

```
Raw 3D T1 MRI (.nii / .nii.gz)
   │
   ▼
[ Stage 1: Geometry & Orientation ]
   └── Reorient to canonical RAS+ space, repair header sform/qform codes.
   │
   ▼
[ Stage 2: Inhomogeneity Correction ]
   └── N4ITK bias field correction on low-frequency RF coil shading.
   │
   ▼
[ Stage 3: Deep Skull-Stripping ]
   └── State-of-the-art HD-BET neural network extraction (Dice > 0.95+).
   │
   ▼
[ Stage 4: Stereotactic Registration ]
   └── Two-stage stabilized Affine registration to MNI152 (2.0 mm isotropic).
   │
   ▼
[ Stage 5: Mask-Constrained Normalization ]
   └── Foreground Z-scoring (μ=0, σ=1) inside brain; non-brain exterior zeroed.
   │
   ▼
[ Stage 6: Automated Biological QA Gate ]
   ├── Adult brain volume plausibility (0.90 L ≤ V ≤ 1.90 L)
   ├── Template Dice overlap score against MNI152 mask (Dice ≥ 0.70)
   └── Bounding box boundary collision verification (zero edge clipping)
```

---

## 2. Directory Structure

```text
DG-TCAV/
├── preprocessing/             # Core modular Python package
│   ├── __init__.py            # Package interface
│   ├── bias_correction.py     # SimpleITK N4 bias field correction
│   ├── config.py              # Pipeline configuration defaults
│   ├── normalization.py       # Brain-mask Z-score intensity scaling
│   ├── orientation.py         # Canonical RAS+ reorientation & affine checks
│   ├── pipeline.py            # End-to-end pipeline runner & QA validator
│   ├── registration.py        # MNI152 template alignment & fetcher
│   ├── skull_stripping.py     # HD-BET GPU brain extraction
│   └── visualizer.py          # Tri-planar QC visualization
├── tests/                     # Unit and synthetic pipeline regression tests
│   ├── test_pipeline_synthetic.py
│   └── test_real_scan.py
├── run_preprocessing.py       # Production batch CLI runner
├── requirements.txt           # Python package requirements
├── .gitignore                 # Excludes data, virtualenvs, and outputs
└── README.md                  # Project documentation
```

---

## 3. Installation

### Local Setup (Windows / Linux / macOS)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/DG-TCAV.git
   cd DG-TCAV
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Running the Pipeline

### Batch Execution CLI

Run batch preprocessing across an entire scan directory:

```bash
python run_preprocessing.py \
    --input_dir "path/to/raw_scans" \
    --output_dir "path/to/preprocessed_output" \
    --report_dir "reports/preprocessing" \
    --device auto \
    --max_subjects 0 \
    --num_workers 1 \
    --visualize
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input_dir` | Path | *Required* | Directory containing raw `.nii` or `.nii.gz` scans |
| `--output_dir` | Path | *Required* | Target directory for preprocessed images and masks |
| `--report_dir` | Path | `reports/preprocessing` | Destination for `qa_manifest.csv` and figures |
| `--device` | String | `auto` | Execution device (`auto`, `cuda`, `cpu`, `mps`) |
| `--max_subjects`| Int | `2` | Max scans to process (`0` for all discovered scans) |
| `--num_workers` | Int | `1` | Concurrent worker processes |
| `--visualize` | Flag | `False` | Generates tri-planar QC comparison figures |
| `--disable_tta` | Flag | `False` | Disables HD-BET test-time augmentation for speed |

---

## 5. Running at Scale on Kaggle GPUs

Kaggle provides free NVIDIA T4/P100 GPUs, reducing per-scan processing time from **~4 minutes (CPU)** to **~30–40 seconds (GPU)**.

### Kaggle Execution Steps:

1. **Upload Dataset:** Upload your raw MRI scans as a private Kaggle Dataset (e.g. `adni-raw-t1`).
2. **Create Kaggle Notebook:**
   * Set **Accelerator** to `GPU T4 x1`.
   * Enable **Internet: ON** in settings.
3. **Run Setup & Execution in Notebook Cells:**

```python
# Cell 1: Dependencies
!pip install -q SimpleITK nibabel hd-bet

# Cell 2: Run Batch Preprocessing on GPU
!python run_preprocessing.py \
    --input_dir "/kaggle/input/adni-raw-t1" \
    --output_dir "/kaggle/working/preprocessed_mni152" \
    --report_dir "/kaggle/working/qa_reports" \
    --device cuda \
    --max_subjects 0 \
    --num_workers 1

# Cell 3: Compress Output for Download
!zip -r -q /kaggle/working/preprocessed_dataset.zip /kaggle/working/preprocessed_mni152 /kaggle/working/qa_reports
```

---

## 6. Quality Assurance & Ground Truth Validation

In clinical neuroimaging, no individual in-vivo patient scan has a universal physical ground truth. The pipeline enforces authenticity and anatomical validity through three pillars:

1. **Reference Standard:** Spatial alignment against the NIH-standard **ICBM 2009c MNI152 template** at $91 \times 109 \times 91$ (2.0 mm isotropic).
2. **Automated Quantitative QA:**
   * **Dice Overlap:** Verified against the standard MNI brain mask ($\text{Dice} \ge 0.70$).
   * **Intracranial Volume (ICV):** Adult human biological range check ($0.90\text{ L} \le V \le 1.90\text{ L}$).
   * **Boundary Collisions:** Strict check that zero brain voxels clip the outer grid boundaries.
3. **Tri-Planar Visual QC:** Slices extracted in Axial (Nose pointing UP), Coronal (Vertex UP), and Sagittal orientations with mask contour overlays for audit review.

---

## 7. License

This project is licensed under the MIT License.
