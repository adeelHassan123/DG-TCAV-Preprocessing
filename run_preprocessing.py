"""Batch CLI for reproducible 3D T1 MRI preprocessing."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

from preprocessing.config import PipelineConfig
from preprocessing.pipeline import preprocess_mri_scan
from preprocessing.registration import ensure_mni_template
from preprocessing.visualizer import generate_5step_progression_figure


def discover_scans(input_dir: str | Path) -> list[tuple[str, str]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    paths = sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])
    result = []
    counts = {}
    for path in paths:
        lower_name = path.name.lower()
        if "_mask" in lower_name or "_preprocessed_mni152_2mm" in lower_name:
            continue
        counts[path.parent] = counts.get(path.parent, 0) + 1
    for path in paths:
        lower_name = path.name.lower()
        if "_mask" in lower_name or "_preprocessed_mni152_2mm" in lower_name:
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
    report_dir: str,
    visualize: bool,
    config: PipelineConfig,
) -> dict:
    output_path = Path(output_dir) / f"{subject_id}_preprocessed_mni152_2mm.nii.gz"
    result = preprocess_mri_scan(
        input_path, str(output_path), collect_steps=visualize, config=config
    )
    figure = ""
    if visualize and result.step_volumes:
        figure_path = (
            Path(report_dir) / "preprocessing_figures" / f"{subject_id}_qc.png"
        )
        generate_5step_progression_figure(
            subject_id,
            result.step_volumes,
            str(figure_path),
            result.raw_shape,
            result.final_shape,
        )
        figure = str(figure_path)

    status = "SUCCESS" if result.qa_passed else "QA_FAILED"
    error = "" if result.qa_passed else result.qa_message

    return {
        "subject_id": subject_id,
        "input_shape": str(result.raw_shape),
        "output_shape": str(result.final_shape),
        "voxel_spacing": str(result.voxel_spacing),
        "brain_mean": round(result.brain_mean, 4),
        "brain_std": round(result.brain_std, 4),
        "brain_voxels": result.brain_voxels,
        "brain_volume_liters": round(result.brain_volume_liters, 3),
        "template_dice": round(result.template_dice, 4),
        "processing_time_sec": round(result.elapsed_sec, 3),
        "status": status,
        "output_file": result.output_path,
        "mask_file": result.mask_path,
        "qa_figure": figure,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production 3D T1 MRI preprocessing to MNI152 2 mm"
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
        "--report_dir", default="reports/preprocessing", help="QA report directory"
    )
    parser.add_argument(
        "--template",
        default="templates/mni152_t1_2mm.nii.gz",
        help="MNI template NIfTI; fetches ICBM152 2009c from the archive if absent",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=2,
        help="Maximum scans to process (default: 2; use 0 for all scans)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="ProcessPoolExecutor workers",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="HD-BET execution device; CPU automatically disables TTA",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        help="Disable HD-BET test-time augmentation",
    )
    parser.add_argument(
        "--brain_extractor", choices=("hd-bet", "synthstrip"), default="hd-bet"
    )
    parser.add_argument(
        "--registration", choices=("affine", "rigid"), default="affine"
    )
    parser.add_argument(
        "--normalization", choices=("zscore", "minmax"), default="zscore"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Write tri-planar QC PNGs"
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num_workers must be positive")

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
        f"Device: {args.device} | Brain extractor: {args.brain_extractor} | "
        f"Workers: {args.num_workers} | Visualization: {args.visualize}",
        flush=True,
    )
    output_dir, report_dir = Path(args.output_dir), Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (report_dir / "preprocessing_figures").mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        template_path=Path(args.template),
        brain_extractor=args.brain_extractor,
        device=args.device,
        disable_tta=args.disable_tta,
        registration=args.registration,
        normalization=args.normalization,
    )
    if inputs:
        ensure_mni_template(
            config.template_path,
            target_shape=config.target_shape,
            spacing_mm=config.output_spacing_mm[0],
        )

    rows = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {}
        for index, (sid, path) in enumerate(inputs, 1):
            print(f"Queued {index}/{len(inputs)}: {sid}", flush=True)
            future = pool.submit(
                _process_one,
                sid,
                path,
                str(output_dir),
                str(report_dir),
                args.visualize,
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
                    "input_shape": "",
                    "output_shape": "",
                    "voxel_spacing": "",
                    "brain_mean": "",
                    "brain_std": "",
                    "brain_voxels": "",
                    "brain_volume_liters": "",
                    "template_dice": "",
                    "processing_time_sec": "",
                    "status": "FAILED",
                    "output_file": "",
                    "mask_file": "",
                    "qa_figure": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                row["_traceback"] = traceback.format_exc()
                print(
                    f"[{completed}/{len(inputs)}] CRASHED {sid} ({path}): {row['error']}",
                    flush=True,
                )
            else:
                if row["status"] == "SUCCESS":
                    print(
                        f"[{completed}/{len(inputs)}] PASS {sid}: {row['output_shape']} "
                        f"| Vol: {row['brain_volume_liters']}L | Dice: {row['template_dice']} "
                        f"({row['processing_time_sec']}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[{completed}/{len(inputs)}] REJECTED {sid}: {row['error']} "
                        f"| Vol: {row['brain_volume_liters']}L ({row['processing_time_sec']}s)",
                        flush=True,
                    )
            rows.append(row)

    fields = [
        "subject_id",
        "input_shape",
        "output_shape",
        "voxel_spacing",
        "brain_mean",
        "brain_std",
        "brain_voxels",
        "brain_volume_liters",
        "template_dice",
        "processing_time_sec",
        "status",
        "output_file",
        "mask_file",
        "qa_figure",
        "error",
    ]
    rows.sort(key=lambda item: item["subject_id"])
    with (report_dir / "qa_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in fields} for row in rows
        )

    failures = [row for row in rows if row["status"] != "SUCCESS"]
    with (report_dir / "failures.log").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(f"{row['subject_id']}: {row['error']}\n")
            stream.write(row.get("_traceback", "") + "\n")

    print(
        f"\nBatch Complete: {len(rows) - len(failures)} passed QA, {len(failures)} failed."
    )
    print(f"Manifest written to: {report_dir / 'qa_manifest.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
