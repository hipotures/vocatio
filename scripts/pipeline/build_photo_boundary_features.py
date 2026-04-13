#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lib.image_pipeline_contracts import PHOTO_MANIFEST_REQUIRED_COLUMNS, validate_required_columns
from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_QUALITY_FILENAME = "photo_quality.csv"
PHOTO_BOUNDARY_FEATURE_FILENAME = "photo_boundary_features.csv"
FEATURES_DIRNAME = "features"
EMBEDDINGS_FILENAME = "dinov2_embeddings.npy"
INDEX_FILENAME = "dinov2_index.csv"
DEFAULT_ROLLING_WINDOW_SIZE = 3

PHOTO_BOUNDARY_FEATURE_HEADERS = [
    "left_relative_path",
    "right_relative_path",
    "left_start_local",
    "right_start_local",
    "time_gap_seconds",
    "dino_cosine_distance",
    "rolling_dino_distance_mean",
    "rolling_dino_distance_std",
    "distance_zscore",
    "left_flag_blurry",
    "right_flag_blurry",
    "left_flag_dark",
    "right_flag_dark",
    "brightness_delta",
    "contrast_delta",
]

PHOTO_BOUNDARY_MANIFEST_REQUIRED_COLUMNS = frozenset(set(PHOTO_MANIFEST_REQUIRED_COLUMNS) | {"start_local"})
PHOTO_QUALITY_REQUIRED_COLUMNS = frozenset(
    {
        "relative_path",
        "brightness_mean",
        "contrast_score",
        "flag_blurry",
        "flag_dark",
    }
)
DINO_INDEX_REQUIRED_COLUMNS = frozenset(
    {
        "relative_path",
        "row_index",
    }
)


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic adjacent-photo boundary features for image-only stage 1."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds stage-1 photo artifacts. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Input manifest CSV filename or absolute path. Default: WORKSPACE/photo_manifest.csv",
    )
    parser.add_argument(
        "--quality-csv",
        help="Input quality CSV filename or absolute path. Default: WORKSPACE/photo_quality.csv",
    )
    parser.add_argument(
        "--features-dir",
        help="Directory that holds dinov2_embeddings.npy and dinov2_index.csv. Default: WORKSPACE/features",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename or absolute path. Default: WORKSPACE/photo_boundary_features.csv",
    )
    parser.add_argument(
        "--rolling-window-size",
        type=positive_int_arg,
        default=DEFAULT_ROLLING_WINDOW_SIZE,
        help=f"Rolling window size for DINO distance stats. Default: {DEFAULT_ROLLING_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing photo_boundary_features.csv output",
    )
    return parser.parse_args(argv)


def resolve_manifest_path(workspace_dir: Path, manifest_value: Optional[str]) -> Path:
    if not manifest_value:
        return workspace_dir / "photo_manifest.csv"
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_quality_path(workspace_dir: Path, quality_value: Optional[str]) -> Path:
    if not quality_value:
        return workspace_dir / PHOTO_QUALITY_FILENAME
    candidate = Path(quality_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_features_dir(workspace_dir: Path, features_value: Optional[str]) -> Path:
    if not features_value:
        return workspace_dir / FEATURES_DIRNAME
    candidate = Path(features_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_output_path(workspace_dir: Path, output_value: Optional[str]) -> Path:
    if not output_value:
        return workspace_dir / PHOTO_BOUNDARY_FEATURE_FILENAME
    candidate = Path(output_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_feature_input_paths(features_dir: Path) -> Dict[str, Path]:
    return {
        "embeddings_path": features_dir / EMBEDDINGS_FILENAME,
        "index_path": features_dir / INDEX_FILENAME,
    }


def normalize_day_relative_path(relative_path: str, column_name: str) -> str:
    value = str(relative_path)
    if value != value.strip():
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    if "\\" in value:
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    candidate = PurePosixPath(value)
    if not value or not candidate.parts:
        raise ValueError(f"{column_name} is empty")
    if candidate.is_absolute():
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    normalized = candidate.as_posix()
    if value != normalized:
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    return normalized


def parse_local_datetime(value: str, column_name: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError(f"{column_name} is empty")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{column_name} must be a valid ISO local datetime: {value}") from exc


def parse_non_negative_int(value: str, column_name: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column_name} is empty")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{column_name} must be an integer: {value}") from exc
    if parsed < 0:
        raise ValueError(f"{column_name} must be non-negative: {value}")
    return parsed


def parse_float(value: str, column_name: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column_name} is empty")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{column_name} must be a float: {value}") from exc


def parse_flag(value: str, column_name: str) -> str:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"{column_name} must be 0 or 1: {value}")
    return text


def format_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}"


def read_photo_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_BOUNDARY_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    normalized_rows: List[Dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        relative_path = normalize_day_relative_path(
            str(row.get("relative_path") or ""),
            f"{path.name} relative_path",
        )
        photo_order_index = parse_non_negative_int(
            str(row.get("photo_order_index") or ""),
            f"{path.name} photo_order_index",
        )
        if photo_order_index != row_number - 1:
            raise ValueError(
                f"{path.name} photo_order_index contract mismatch at row {row_number}: "
                f"expected {row_number - 1}, got {photo_order_index}"
            )
        start_local = str(row.get("start_local") or "").strip()
        parse_local_datetime(start_local, f"{path.name} start_local")
        normalized_row = dict(row)
        normalized_row["relative_path"] = relative_path
        normalized_row["photo_order_index"] = str(photo_order_index)
        normalized_row["start_local"] = start_local
        normalized_rows.append(normalized_row)
    return normalized_rows


def read_photo_quality(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_QUALITY_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    normalized_rows: List[Dict[str, str]] = []
    for row in rows:
        relative_path = normalize_day_relative_path(
            str(row.get("relative_path") or ""),
            f"{path.name} relative_path",
        )
        brightness_mean = parse_float(str(row.get("brightness_mean") or ""), f"{path.name} brightness_mean")
        contrast_score = parse_float(str(row.get("contrast_score") or ""), f"{path.name} contrast_score")
        flag_blurry = parse_flag(str(row.get("flag_blurry") or ""), f"{path.name} flag_blurry")
        flag_dark = parse_flag(str(row.get("flag_dark") or ""), f"{path.name} flag_dark")
        normalized_rows.append(
            {
                "relative_path": relative_path,
                "brightness_mean": format_float(brightness_mean),
                "contrast_score": format_float(contrast_score),
                "flag_blurry": flag_blurry,
                "flag_dark": flag_dark,
            }
        )
    return normalized_rows


def read_embedding_index(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, DINO_INDEX_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    normalized_rows: List[Dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        relative_path = normalize_day_relative_path(
            str(row.get("relative_path") or ""),
            f"{path.name} relative_path",
        )
        row_index = parse_non_negative_int(str(row.get("row_index") or ""), f"{path.name} row_index")
        if row_index != row_number - 1:
            raise ValueError(
                f"{path.name} row_index contract mismatch at row {row_number}: "
                f"expected {row_number - 1}, got {row_index}"
            )
        normalized_rows.append(
            {
                "relative_path": relative_path,
                "row_index": str(row_index),
            }
        )
    return normalized_rows


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(f"{path.name} must be a 2D array, got shape {embeddings.shape}")
    if embeddings.shape[1] <= 0:
        raise ValueError(f"{path.name} must have a positive embedding width")
    return embeddings.astype(np.float32, copy=False)


def validate_row_counts(
    manifest_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    index_rows: Sequence[Mapping[str, str]],
    embeddings: np.ndarray,
) -> None:
    counts = {
        "photo_manifest.csv": len(manifest_rows),
        "photo_quality.csv": len(quality_rows),
        "dinov2_index.csv": len(index_rows),
        "dinov2_embeddings.npy": int(embeddings.shape[0]),
    }
    if len(set(counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(f"Row count mismatch across stage-1 artifacts: {details}")


def validate_row_alignment(
    manifest_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    index_rows: Sequence[Mapping[str, str]],
) -> None:
    for row_number, manifest_row in enumerate(manifest_rows, start=1):
        expected_relative_path = str(manifest_row["relative_path"])
        quality_relative_path = str(quality_rows[row_number - 1]["relative_path"])
        index_relative_path = str(index_rows[row_number - 1]["relative_path"])
        if quality_relative_path != expected_relative_path:
            raise ValueError(
                f"photo_quality.csv relative_path mismatch at row {row_number}: "
                f"expected {expected_relative_path}, got {quality_relative_path}"
            )
        if index_relative_path != expected_relative_path:
            raise ValueError(
                f"dinov2_index.csv relative_path mismatch at row {row_number}: "
                f"expected {expected_relative_path}, got {index_relative_path}"
            )


def cosine_distance(left: np.ndarray, right: np.ndarray, pair_key: str) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError(f"DINOv2 embeddings must be non-zero for {pair_key}")
    cosine_similarity = float(np.dot(left, right) / (left_norm * right_norm))
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return float(1.0 - cosine_similarity)


def compute_boundary_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    embeddings: np.ndarray,
    rolling_window_size: int,
) -> List[Dict[str, str]]:
    boundary_count = max(len(manifest_rows) - 1, 0)
    distances: List[float] = []
    for index in range(boundary_count):
        left_relative_path = str(manifest_rows[index]["relative_path"])
        right_relative_path = str(manifest_rows[index + 1]["relative_path"])
        distances.append(
            cosine_distance(
                embeddings[index],
                embeddings[index + 1],
                f"{left_relative_path} -> {right_relative_path}",
            )
        )
    rows: List[Dict[str, str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    ) as progress:
        task = progress.add_task("Build boundaries".ljust(25), total=boundary_count)
        for index in range(boundary_count):
            left_manifest = manifest_rows[index]
            right_manifest = manifest_rows[index + 1]
            left_quality = quality_rows[index]
            right_quality = quality_rows[index + 1]
            left_start_local = str(left_manifest["start_local"])
            right_start_local = str(right_manifest["start_local"])
            left_dt = parse_local_datetime(left_start_local, "photo_manifest.csv start_local")
            right_dt = parse_local_datetime(right_start_local, "photo_manifest.csv start_local")
            time_gap_seconds = float((right_dt - left_dt).total_seconds())
            if time_gap_seconds < 0.0:
                raise ValueError(
                    "photo_manifest.csv start_local must be non-decreasing in manifest order: "
                    f"{left_manifest['relative_path']} -> {right_manifest['relative_path']}"
                )
            distance = distances[index]
            window_start = max(0, index - rolling_window_size + 1)
            window = np.asarray(distances[window_start : index + 1], dtype=np.float32)
            rolling_mean = float(window.mean()) if window.size else 0.0
            rolling_std = float(window.std()) if window.size else 0.0
            distance_zscore = 0.0 if rolling_std <= 0.0 else float((distance - rolling_mean) / rolling_std)
            brightness_delta = abs(
                parse_float(str(right_quality["brightness_mean"]), "photo_quality.csv brightness_mean")
                - parse_float(str(left_quality["brightness_mean"]), "photo_quality.csv brightness_mean")
            )
            contrast_delta = abs(
                parse_float(str(right_quality["contrast_score"]), "photo_quality.csv contrast_score")
                - parse_float(str(left_quality["contrast_score"]), "photo_quality.csv contrast_score")
            )
            rows.append(
                {
                    "left_relative_path": str(left_manifest["relative_path"]),
                    "right_relative_path": str(right_manifest["relative_path"]),
                    "left_start_local": left_start_local,
                    "right_start_local": right_start_local,
                    "time_gap_seconds": format_float(time_gap_seconds),
                    "dino_cosine_distance": format_float(distance),
                    "rolling_dino_distance_mean": format_float(rolling_mean),
                    "rolling_dino_distance_std": format_float(rolling_std),
                    "distance_zscore": format_float(distance_zscore),
                    "left_flag_blurry": str(left_quality["flag_blurry"]),
                    "right_flag_blurry": str(right_quality["flag_blurry"]),
                    "left_flag_dark": str(left_quality["flag_dark"]),
                    "right_flag_dark": str(right_quality["flag_dark"]),
                    "brightness_delta": format_float(brightness_delta),
                    "contrast_delta": format_float(contrast_delta),
                }
            )
            progress.advance(task)
    return rows


def build_photo_boundary_features(
    workspace_dir: Path,
    manifest_csv: Path,
    quality_csv: Path,
    features_dir: Path,
    output_path: Path,
    rolling_window_size: int = DEFAULT_ROLLING_WINDOW_SIZE,
) -> int:
    _ = workspace_dir
    if rolling_window_size <= 0:
        raise ValueError("rolling_window_size must be positive")
    feature_inputs = resolve_feature_input_paths(features_dir)
    manifest_rows = read_photo_manifest(manifest_csv)
    quality_rows = read_photo_quality(quality_csv)
    index_rows = read_embedding_index(feature_inputs["index_path"])
    embeddings = load_embeddings(feature_inputs["embeddings_path"])
    validate_row_counts(manifest_rows, quality_rows, index_rows, embeddings)
    validate_row_alignment(manifest_rows, quality_rows, index_rows)
    boundary_rows = compute_boundary_rows(manifest_rows, quality_rows, embeddings, rolling_window_size)
    atomic_write_csv(output_path, PHOTO_BOUNDARY_FEATURE_HEADERS, boundary_rows)
    return len(boundary_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    manifest_csv = resolve_manifest_path(workspace_dir, args.manifest_csv)
    quality_csv = resolve_quality_path(workspace_dir, args.quality_csv)
    features_dir = resolve_features_dir(workspace_dir, args.features_dir)
    output_path = resolve_output_path(workspace_dir, args.output)
    feature_inputs = resolve_feature_input_paths(features_dir)
    for input_path, label in (
        (manifest_csv, "Manifest CSV"),
        (quality_csv, "Quality CSV"),
        (feature_inputs["index_path"], "DINO index CSV"),
        (feature_inputs["embeddings_path"], "DINO embeddings array"),
    ):
        if not input_path.exists():
            raise SystemExit(f"{label} does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output CSV already exists: {output_path}. Use --overwrite to replace it.")
    row_count = build_photo_boundary_features(
        workspace_dir=workspace_dir,
        manifest_csv=manifest_csv,
        quality_csv=quality_csv,
        features_dir=features_dir,
        output_path=output_path,
        rolling_window_size=args.rolling_window_size,
    )
    console.print(f"Wrote {row_count} photo boundary feature rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
