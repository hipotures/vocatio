#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from rich.console import Console

from lib.pipeline_io import atomic_write_json


console = Console(stderr=True)

TRAIN_MODES = ("tabular_only", "tabular_plus_thumbnail")
THUMBNAIL_IMAGE_COLUMNS = [
    "frame_01_thumb_path",
    "frame_02_thumb_path",
    "frame_03_thumb_path",
    "frame_04_thumb_path",
    "frame_05_thumb_path",
]
TRAINING_PLAN_FILENAME = "training_plan.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"
REQUIRED_BASE_COLUMNS = ["segment_type", "boundary"]


def _validate_mode(mode: str) -> str:
    if mode not in TRAIN_MODES:
        choices = ", ".join(TRAIN_MODES)
        raise ValueError(f"mode must be one of: {choices}")
    return mode


def build_training_plan(mode: str) -> list[dict[str, str]]:
    _validate_mode(mode)
    return [
        {"name": "segment_type", "problem_type": "multiclass"},
        {"name": "boundary", "problem_type": "binary"},
    ]


def default_boundary_threshold_policy() -> dict[str, float | str]:
    return {"policy": "fixed", "threshold": 0.5}


def image_feature_columns_for_mode(mode: str) -> list[str]:
    _validate_mode(mode)
    if mode != "tabular_plus_thumbnail":
        return []
    return list(THUMBNAIL_IMAGE_COLUMNS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write ML boundary verifier training contract artifacts from a CSV dataset.",
    )
    parser.add_argument(
        "dataset_path",
        help="Path to ml_boundary_candidates CSV dataset.",
    )
    parser.add_argument(
        "--mode",
        default="tabular_only",
        choices=TRAIN_MODES,
        help="Training mode to scaffold.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where training artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing training contract artifacts in the output directory.",
    )
    return parser.parse_args(argv)


def _build_training_plan_payload(dataset_path: Path, mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "predictors": build_training_plan(mode),
        "boundary_threshold_policy": default_boundary_threshold_policy(),
        "image_feature_columns": image_feature_columns_for_mode(mode),
        "dataset_path": str(dataset_path),
    }


def _build_training_metadata_payload(dataset_path: Path, output_dir: Path, mode: str) -> dict[str, object]:
    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "mode": mode,
        "predictor_names": [predictor["name"] for predictor in build_training_plan(mode)],
        "threshold_policy": default_boundary_threshold_policy(),
        "image_feature_columns": image_feature_columns_for_mode(mode),
    }


def _require_csv_headers(dataset_path: Path, *, required_headers: Sequence[str]) -> None:
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
    missing = [header for header in required_headers if header not in fieldnames]
    if missing:
        raise ValueError(
            f"{dataset_path.name} missing required columns: {', '.join(missing)}"
        )


def validate_dataset_contract(dataset_path: Path, mode: str) -> None:
    _validate_mode(mode)
    suffix = dataset_path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(
            f"Unsupported dataset extension for {dataset_path.name}: expected .csv"
        )
    if suffix == ".parquet":
        raise ValueError(
            "Parquet dataset schema inspection is not supported in this scaffold; "
            "use a CSV dataset for now"
        )

    required_headers = list(REQUIRED_BASE_COLUMNS)
    required_headers.extend(image_feature_columns_for_mode(mode))
    _require_csv_headers(dataset_path, required_headers=required_headers)


def _artifact_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / TRAINING_PLAN_FILENAME,
        output_dir / TRAINING_METADATA_FILENAME,
    ]


def _guard_existing_artifacts(output_dir: Path, *, overwrite: bool) -> None:
    existing_paths = [path for path in _artifact_paths(output_dir) if path.exists()]
    if existing_paths and not overwrite:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Output artifacts already exist in {output_dir}: {names}. "
            "Use --overwrite to replace them."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    validate_dataset_contract(dataset_path, args.mode)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_existing_artifacts(output_dir, overwrite=args.overwrite)

    training_plan_payload = _build_training_plan_payload(dataset_path, args.mode)
    training_metadata_payload = _build_training_metadata_payload(dataset_path, output_dir, args.mode)

    atomic_write_json(output_dir / TRAINING_PLAN_FILENAME, training_plan_payload)
    atomic_write_json(output_dir / TRAINING_METADATA_FILENAME, training_metadata_payload)

    console.print(
        f"Wrote training contract artifacts to {output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
