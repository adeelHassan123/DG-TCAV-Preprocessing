# Structural T1 preprocessing

The runner expects one 3D T1-weighted NIfTI per scan. It writes a float32,
brain-masked normalized volume and a uint8 brain mask on the MNI template grid.
The default grid is 91 × 109 × 91 at 2 mm isotropic spacing.

Install the dependencies in a dedicated Python 3.10+ environment. HD-BET
downloads its model weights on first use; the host needs access to the upstream
weight server for that initial run. CPU mode is supported and disables test
time augmentation automatically:

```powershell
python -m pip install -r requirements.txt
python run_preprocessing.py --input_dir DATA --output_dir data/preprocessed --device cpu --visualize
```

`--num_workers` enables process based batch execution. For CUDA, install the
PyTorch build matching the host and pass `--device cuda`. FreeSurfer users may
select `--brain_extractor synthstrip`.

If `templates/mni152_t1_2mm.nii.gz` is absent, the runner fetches the official
ICBM152 2009cSAsym 2 mm T1w template directly from the TemplateFlow archive.
Its full image is
97 × 115 × 97; the runner center-crops it to the required 91 × 109 × 91 grid
and shifts the affine so voxel coordinates remain in MNI space. A supplied
template can be passed with `--template`. The reference geometry and spatial
header codes are copied to both outputs.

The report directory contains `qa_manifest.csv`, `failures.log`, and optional
tri-planar PNGs under `preprocessing_figures/`. A failed model invocation or
registration marks only that scan as failed; no heuristic skull-strip fallback
is applied.
