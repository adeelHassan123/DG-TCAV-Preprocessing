"""Batch CLI for reproducible 3D T1 MRI preprocessing."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import traceback

from preprocessing.config import PipelineConfig
from preprocessing.pipeline import preprocess_mri_scan

PIPELINE_VERSION = "masked_rigid_affine_centroid_seed42_v4"


def discover_scans(input_dir: str | Path) -> list[tuple[str, str]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    paths = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])
    result = []
    counts = {}
    for path in paths:
        lower_name = path.name.lower()
        if "_mask" in lower_name or "_preprocessed_128" in lower_name:
            continue
        counts[path.parent] = counts.get(path.parent, 0) + 1
    for path in paths:
        lower_name = path.name.lower()
        if "_mask" in lower_name or "_preprocessed_128" in lower_name:
            continue
        stem = path.name.removesuffix(".nii.gz").removesuffix(".nii")
        relative_parent = path.parent.relative_to(root)
        sid = "_".join(relative_parent.parts) if relative_parent.parts else stem
        if counts[path.parent] > 1:
            sid = f"{sid}_{stem}"
        result.append((sid, str(path)))
    return result


def _process_one(
    subject_id: str,
    input_path: str,
    output_dir: str,
    config: PipelineConfig,
) -> dict:
    size = config.target_shape[0]
    output_path = Path(output_dir) / f"{subject_id}_preprocessed_{size}.nii.gz"
    result = preprocess_mri_scan(
        input_path, str(output_path), collect_steps=False, config=config
    )

    return {
        "subject_id": subject_id,
        "pipeline_version": PIPELINE_VERSION,
        "input_shape": str(result.raw_shape),
        "output_shape": str(result.final_shape),
        "voxel_spacing": str(result.voxel_spacing),
        "brain_mean": round(result.brain_mean, 4),
        "brain_std": round(result.brain_std, 4),
        "brain_voxels": result.brain_voxels,
        "brain_volume_liters": round(result.brain_volume_liters, 3),
        "processing_time_sec": round(result.elapsed_sec, 3),
        "status": "SUCCESS",
        "output_file": result.output_path,
        "mask_file": result.mask_path,
        "error": "",
    }


def _has_complete_output(output_path: Path) -> bool:
    mask_path = output_path.with_name(
        output_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        + "_mask.nii.gz"
    )
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (output_path, mask_path)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess 3D T1 MRI scans for DG-TCAV"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing 3D NIfTI T1 scans"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for normalized images and masks",
    )
    parser.add_argument(
        "--report_dir", default="reports/preprocessing", help="Manifest and failure-log directory"
    )
    parser.add_argument(
        "--template",
        default="templates/mni152.nii.gz",
        help="MNI152 template used for alignment",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=0,
        help="Maximum scans to process (0 processes all discovered scans)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="ProcessPoolExecutor workers",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=128,
        help="Cubic output size (default: 128)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip non-empty image/mask outputs recorded as successful with this pipeline version",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num_workers must be positive")
    if args.target_size < 1:
        parser.error("--target_size must be positive")

    all_inputs = discover_scans(args.input_dir)
    inputs = all_inputs
    if args.max_subjects > 0:
        inputs = inputs[: args.max_subjects]
    if not inputs:
        parser.error(f"No .nii or .nii.gz scans found under {args.input_dir}")
    print(
        f"Found {len(all_inputs)} scan(s); selected {len(inputs)} for processing.",
        flush=True,
    )
    print(
        f"Extractor: DeepBrain | Target size: {args.target_size}^3 | Workers: {args.num_workers}",
        flush=True,
    )
    output_dir, report_dir = Path(args.output_dir), Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        template_path=Path(args.template),
        target_shape=(args.target_size,) * 3,
    )
    if not config.template_path.is_file():
        parser.error(f"MNI template not found: {config.template_path}")

    fields = [
        "subject_id",
        "pipeline_version",
        "input_shape",
        "output_shape",
        "voxel_spacing",
        "brain_mean",
        "brain_std",
        "brain_voxels",
        "brain_volume_liters",
        "processing_time_sec",
        "status",
        "output_file",
        "mask_file",
        "error",
    ]
    manifest_path = report_dir / "preprocessing_manifest.csv"
    rows = []
    known_successes = set()
    manifest_has_content = (
        args.resume and manifest_path.is_file() and manifest_path.stat().st_size > 0
    )
    if manifest_has_content:
        with manifest_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        latest_existing = {}
        for row in rows:
            latest_existing[row.get("subject_id")] = row
        known_successes = {
            (
                row.get("subject_id"),
                row.get("output_file"),
                row.get("mask_file"),
                row.get("pipeline_version"),
            )
            for row in latest_existing.values()
            if row.get("status") in {"SUCCESS", "SKIPPED_EXISTING"}
            and row.get("pipeline_version") == PIPELINE_VERSION
        }

    manifest_needs_header = not manifest_has_content
    with manifest_path.open(
        "w" if manifest_needs_header else "a", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if manifest_needs_header:
            writer.writeheader()
            stream.flush()

        def record(row: dict) -> None:
            writer.writerow({key: row.get(key, "") for key in fields})
            stream.flush()
            os.fsync(stream.fileno())

        pending = []
        for sid, input_path in inputs:
            target_size = config.target_shape[0]
            expected_output = output_dir / f"{sid}_preprocessed_{target_size}.nii.gz"
            expected_mask = expected_output.with_name(
                expected_output.name.removesuffix(".nii.gz") + "_mask.nii.gz"
            )
            prior_key = (
                sid,
                str(expected_output),
                str(expected_mask),
                PIPELINE_VERSION,
            )
            if (
                args.resume
                and _has_complete_output(expected_output)
                and prior_key in known_successes
            ):
                print(f"Resume: existing image and mask found for {sid}; skipping.", flush=True)
            else:
                pending.append((sid, input_path))

        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {}
            for index, (sid, path) in enumerate(pending, 1):
                print(f"Queued {index}/{len(pending)}: {sid}", flush=True)
                future = pool.submit(
                    _process_one,
                    sid,
                    path,
                    str(output_dir),
                    config,
                )
                futures[future] = (sid, path)

            for completed, future in enumerate(as_completed(futures), 1):
                sid, path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "subject_id": sid,
                        "pipeline_version": PIPELINE_VERSION,
                        "input_shape": "",
                        "output_shape": "",
                        "voxel_spacing": "",
                        "brain_mean": "",
                        "brain_std": "",
                        "brain_voxels": "",
                        "brain_volume_liters": "",
                        "processing_time_sec": "",
                        "status": "FAILED",
                        "output_file": "",
                        "mask_file": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    row["_traceback"] = traceback.format_exc()
                    print(
                        f"[{completed}/{len(futures)}] CRASHED {sid} ({path}): {row['error']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{completed}/{len(futures)}] PASS {sid}: {row['output_shape']} "
                        f"| mask volume: {row['brain_volume_liters']}L "
                        f"({row['processing_time_sec']}s)",
                        flush=True,
                    )
                record(row)
                rows.append(row)

    latest_rows = {}
    for row in rows:
        latest_rows[row["subject_id"]] = row
    failures = [row for row in latest_rows.values() if row["status"] == "FAILED"]
    completed_count = sum(
        row["status"] in {"SUCCESS", "SKIPPED_EXISTING"}
        for row in latest_rows.values()
    )
    with (report_dir / "failures.log").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(f"{row['subject_id']}: {row['error']}\n")
            stream.write(row.get("_traceback", "") + "\n")

    print(
        f"\nBatch complete: {completed_count} successful or resumed, {len(failures)} failed."
    )
    print(f"Manifest updated incrementally at: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
