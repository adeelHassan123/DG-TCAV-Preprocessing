# MRI preprocessing

This project prepares 3D T1 MRI scans as `128 × 128 × 128` images for the
DG-TCAV model.

## What happens to each scan

1. Reorient the image to RPS.
2. Correct uneven image brightness with N4.
3. Extract the brain with DeepBrain, using a probability threshold of `0.3`.
4. Align the scan to `templates/mni152.nii.gz` using the extracted brain mask.
5. Normalize image intensities, then resize to `128 × 128 × 128` while keeping
   the same physical field of view.

The alignment uses a rigid fit followed by an affine fit. It rejects a result
if more than 1% of the extracted brain would fall outside the template. The
mask also gets saved for visual inspection; it
does not change the image intensities during normalization.

## Install

Use a Python environment that supports TensorFlow and DeepBrain, then install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

DeepBrain uses an older TensorFlow interface. If it fails to install or load,
use a Python and TensorFlow version supported by DeepBrain.

## Process scans

Run commands from the repository directory. Start with one scan:

```powershell
python run_preprocessing.py `
  --input_dir "data/raw_scans" `
  --output_dir "data/preprocessed" `
  --max_subjects 1 `
  --num_workers 1
```

After checking that output, process all scans by changing `--max_subjects 1`
to `--max_subjects 0`. The runner searches the input folder and its subfolders
for `.nii` and `.nii.gz` files. It uses the MNI template at
`templates/mni152.nii.gz` by default.

Each scan produces a preprocessed image and a brain-mask file in the output
folder. The runner also writes `reports/preprocessing/preprocessing_manifest.csv`
and `reports/preprocessing/failures.log`. The manifest is updated after each
scan. Use `--resume` to skip a scan when both output files are present and
non-empty **and** the manifest records a successful run with the current
pipeline version.

Useful options:

- `--template PATH` uses a different MNI template.
- `--target_size N` changes the output to `N × N × N` voxels (default: `128`).
- `--report_dir PATH` changes where the manifest and failure log are saved.
- `--num_workers N` sets the number of parallel worker processes (default: `1`).

## View the steps

Open `preprocessing_walkthrough.ipynb`, set `INPUT_SCAN` to a raw NIfTI file,
and run the cells from top to bottom. The notebook shows the scan after each
step in axial, coronal, and sagittal views.
