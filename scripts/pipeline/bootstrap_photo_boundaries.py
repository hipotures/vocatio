#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence

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

from lib.image_pipeline_contracts import validate_required_columns
from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_BOUNDARY_FEATURE_FILENAME = "photo_boundary_features.csv"
PHOTO_BOUNDARY_SCORE_FILENAME = "photo_boundary_scores.csv"

DEFAULT_ZSCORE_THRESHOLD = 1.5
DEFAULT_SOFT_GAP_SECONDS = 20.0
DEFAULT_HARD_GAP_SECONDS = 90.0

LOCAL_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3}|\.\d{6})?$")

PHOTO_BOUNDARY_FEATURE_REQUIRED_COLUMNS = frozenset(
    {
        "left_relative_path",
        "right_relative_path",
        "left_start_local",
        "right_start_local",
        "time_gap_seconds",
        "dino_cosine_distance",
        "distance_zscore",
    }
)

PHOTO_BOUNDARY_SCORE_HEADERS = [
    "left_relative_path",
    "right_relative_path",
    "left_start_local",
    "right_start_local",
    "time_gap_seconds",
    "dino_cosine_distance",
    "distance_zscore",
    "smoothed_distance_zscore",
    "time_gap_boost",
    "boundary_score",
    "boundary_label",
    "boundary_reason",
    "model_source",
]


def positive_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap deterministic stage-1 photo boundary scores from adjacent-photo features."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds stage-1 photo artifacts. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--boundary-features-csv",
        help="Input boundary feature CSV filename or absolute path. Default: WORKSPACE/photo_boundary_features.csv",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename or absolute path. Default: WORKSPACE/photo_boundary_scores.csv",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=positive_float_arg,
        default=DEFAULT_ZSCORE_THRESHOLD,
        help=f"Smoothed z-score threshold for likely cuts. Default: {DEFAULT_ZSCORE_THRESHOLD}",
    )
    parser.add_argument(
        "--soft-gap-seconds",
        type=positive_float_arg,
        default=DEFAULT_SOFT_GAP_SECONDS,
        help=f"Time-gap boost threshold in seconds. Default: {DEFAULT_SOFT_GAP_SECONDS}",
    )
    parser.add_argument(
        "--hard-gap-seconds",
        type=positive_float_arg,
        default=DEFAULT_HARD_GAP_SECONDS,
        help=f"Hard-cut time-gap threshold in seconds. Default: {DEFAULT_HARD_GAP_SECONDS}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing photo_boundary_scores.csv output",
    )
    return parser.parse_args(argv)


def resolve_boundary_features_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_BOUNDARY_FEATURE_FILENAME
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_output_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_BOUNDARY_SCORE_FILENAME
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


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
    text = str(value)
    if not text:
        raise ValueError(f"{column_name} is empty")
    if text != text.strip():
        raise ValueError(f"{column_name} must be a valid ISO local datetime: {value}")
    if not LOCAL_DATETIME_RE.match(text):
        raise ValueError(f"{column_name} must be a valid ISO local datetime: {value}")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{column_name} must be a valid ISO local datetime: {value}") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{column_name} must be a naive ISO local datetime without timezone offset: {value}")
    return parsed


def parse_float(value: str, column_name: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column_name} is empty")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{column_name} must be a float: {value}") from exc
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{column_name} must be finite: {value}")
    return parsed


def parse_flag(value: str, column_name: str) -> str:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"{column_name} must be 0 or 1: {value}")
    return text


def format_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def read_boundary_features(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_BOUNDARY_FEATURE_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    normalized_rows: List[Dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    previous_right_path: Optional[str] = None
    previous_right_start_local: Optional[str] = None
    for row_number, row in enumerate(rows, start=1):
        left_relative_path = normalize_day_relative_path(
            str(row.get("left_relative_path") or ""),
            f"{path.name} left_relative_path",
        )
        right_relative_path = normalize_day_relative_path(
            str(row.get("right_relative_path") or ""),
            f"{path.name} right_relative_path",
        )
        left_start_local = str(row.get("left_start_local") or "")
        right_start_local = str(row.get("right_start_local") or "")
        left_dt = parse_local_datetime(left_start_local, f"{path.name} left_start_local")
        right_dt = parse_local_datetime(right_start_local, f"{path.name} right_start_local")
        if right_dt < left_dt:
            raise ValueError(
                f"{path.name} adjacent timestamps must be non-decreasing at row {row_number}: "
                f"{left_start_local} -> {right_start_local}"
            )
        time_gap_seconds = parse_float(str(row.get("time_gap_seconds") or ""), f"{path.name} time_gap_seconds")
        expected_gap_seconds = float((right_dt - left_dt).total_seconds())
        if abs(time_gap_seconds - expected_gap_seconds) > 0.0000005:
            raise ValueError(
                f"{path.name} time_gap_seconds does not match adjacent timestamps at row {row_number}: "
                f"expected {format_float(expected_gap_seconds)}, got {format_float(time_gap_seconds)}"
            )
        dino_cosine_distance = parse_float(
            str(row.get("dino_cosine_distance") or ""),
            f"{path.name} dino_cosine_distance",
        )
        if dino_cosine_distance < 0.0:
            raise ValueError(f"{path.name} dino_cosine_distance must be non-negative")
        parse_float(str(row.get("distance_zscore") or ""), f"{path.name} distance_zscore")
        key = (left_relative_path, right_relative_path)
        if key in seen_keys:
            raise ValueError(
                f"{path.name} duplicate boundary pair at row {row_number}: {left_relative_path} -> {right_relative_path}"
            )
        seen_keys.add(key)
        if previous_right_path is not None and left_relative_path != previous_right_path:
            raise ValueError(
                f"{path.name} adjacency mismatch at row {row_number}: expected left_relative_path "
                f"{previous_right_path}, got {left_relative_path}"
            )
        if previous_right_start_local is not None and left_start_local != previous_right_start_local:
            raise ValueError(
                f"{path.name} timestamp adjacency mismatch at row {row_number}: expected left_start_local "
                f"{previous_right_start_local}, got {left_start_local}"
            )
        previous_right_path = right_relative_path
        previous_right_start_local = right_start_local
        normalized_rows.append(
            {
                "left_relative_path": left_relative_path,
                "right_relative_path": right_relative_path,
                "left_start_local": left_start_local,
                "right_start_local": right_start_local,
                "time_gap_seconds": format_float(time_gap_seconds),
                "dino_cosine_distance": format_float(dino_cosine_distance),
                "distance_zscore": format_float(
                    parse_float(str(row.get("distance_zscore") or ""), f"{path.name} distance_zscore")
                ),
                "left_flag_blurry": parse_flag(str(row.get("left_flag_blurry") or "0"), f"{path.name} left_flag_blurry"),
                "right_flag_blurry": parse_flag(
                    str(row.get("right_flag_blurry") or "0"),
                    f"{path.name} right_flag_blurry",
                ),
                "left_flag_dark": parse_flag(str(row.get("left_flag_dark") or "0"), f"{path.name} left_flag_dark"),
                "right_flag_dark": parse_flag(str(row.get("right_flag_dark") or "0"), f"{path.name} right_flag_dark"),
            }
        )
    return normalized_rows


def smooth_distance_zscores(rows: Sequence[Mapping[str, str]]) -> List[float]:
    if not rows:
        return []
    values = [parse_float(str(row["distance_zscore"]), "photo_boundary_features.csv distance_zscore") for row in rows]
    smoothed: List[float] = []
    for index, value in enumerate(values):
        total = 0.0
        weighted = 0.0
        for offset, weight in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            neighbor_index = index + offset
            if 0 <= neighbor_index < len(values):
                total += weight
                weighted += values[neighbor_index] * weight
        smoothed.append(value if total <= 0.0 else weighted / total)
    return smoothed


def compute_score_row(
    row: Mapping[str, str],
    smoothed_distance_zscore: float,
    zscore_threshold: float,
    soft_gap_seconds: float,
    hard_gap_seconds: float,
) -> Dict[str, str]:
    time_gap_seconds = parse_float(str(row["time_gap_seconds"]), "photo_boundary_features.csv time_gap_seconds")
    dino_cosine_distance = parse_float(
        str(row["dino_cosine_distance"]),
        "photo_boundary_features.csv dino_cosine_distance",
    )
    hard_cut = time_gap_seconds >= hard_gap_seconds
    time_gap_boost = 1.0 if hard_cut else 0.25 if time_gap_seconds >= soft_gap_seconds else 0.0
    z_strength = clamp(smoothed_distance_zscore / (zscore_threshold * 2.0), 0.0, 1.0)
    distance_strength = clamp(dino_cosine_distance / 0.35, 0.0, 1.0)
    main_signal = (0.75 * z_strength) + (0.25 * distance_strength)
    boosted = main_signal
    if time_gap_boost > 0.0 and (smoothed_distance_zscore >= (zscore_threshold * 0.75) or dino_cosine_distance >= 0.20):
        boosted += time_gap_boost
    boundary_score = 1.0 if hard_cut else clamp(boosted, 0.0, 1.0)
    boundary_label = "none"
    boundary_reason = "distance_only"
    if hard_cut:
        boundary_label = "hard"
        boundary_reason = "hard_gap"
    elif boundary_score >= 0.75:
        boundary_label = "soft"
        boundary_reason = "gap_and_distance" if time_gap_seconds >= soft_gap_seconds else "distance_zscore"
    return {
        "left_relative_path": str(row["left_relative_path"]),
        "right_relative_path": str(row["right_relative_path"]),
        "left_start_local": str(row["left_start_local"]),
        "right_start_local": str(row["right_start_local"]),
        "time_gap_seconds": format_float(time_gap_seconds),
        "dino_cosine_distance": format_float(dino_cosine_distance),
        "distance_zscore": format_float(
            parse_float(str(row["distance_zscore"]), "photo_boundary_features.csv distance_zscore")
        ),
        "smoothed_distance_zscore": format_float(smoothed_distance_zscore),
        "time_gap_boost": format_float(time_gap_boost),
        "boundary_score": format_float(boundary_score),
        "boundary_label": boundary_label,
        "boundary_reason": boundary_reason,
        "model_source": "bootstrap_heuristic",
    }


def build_score_rows(
    boundary_rows: Sequence[Mapping[str, str]],
    zscore_threshold: float,
    soft_gap_seconds: float,
    hard_gap_seconds: float,
) -> List[Dict[str, str]]:
    smoothed_distance_zscores = smooth_distance_zscores(boundary_rows)
    score_rows: List[Dict[str, str]] = []
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
        task = progress.add_task("Bootstrap cuts".ljust(25), total=len(boundary_rows))
        for index, row in enumerate(boundary_rows):
            score_rows.append(
                compute_score_row(
                    row=row,
                    smoothed_distance_zscore=smoothed_distance_zscores[index],
                    zscore_threshold=zscore_threshold,
                    soft_gap_seconds=soft_gap_seconds,
                    hard_gap_seconds=hard_gap_seconds,
                )
            )
            progress.advance(task)
    return score_rows


def bootstrap_photo_boundaries(
    workspace_dir: Path,
    boundary_features_csv: Path,
    output_path: Path,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    soft_gap_seconds: float = DEFAULT_SOFT_GAP_SECONDS,
    hard_gap_seconds: float = DEFAULT_HARD_GAP_SECONDS,
) -> int:
    _ = workspace_dir
    if zscore_threshold <= 0.0:
        raise ValueError("zscore_threshold must be positive")
    if soft_gap_seconds <= 0.0:
        raise ValueError("soft_gap_seconds must be positive")
    if hard_gap_seconds <= 0.0:
        raise ValueError("hard_gap_seconds must be positive")
    if hard_gap_seconds < soft_gap_seconds:
        raise ValueError("hard_gap_seconds must be greater than or equal to soft_gap_seconds")
    boundary_rows = read_boundary_features(boundary_features_csv)
    score_rows = build_score_rows(boundary_rows, zscore_threshold, soft_gap_seconds, hard_gap_seconds)
    atomic_write_csv(output_path, PHOTO_BOUNDARY_SCORE_HEADERS, score_rows)
    return len(score_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    boundary_features_csv = resolve_boundary_features_path(workspace_dir, args.boundary_features_csv)
    output_path = resolve_output_path(workspace_dir, args.output)
    if not boundary_features_csv.exists():
        raise SystemExit(f"Boundary features CSV does not exist: {boundary_features_csv}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output CSV already exists: {output_path}. Use --overwrite to replace it.")
    row_count = bootstrap_photo_boundaries(
        workspace_dir=workspace_dir,
        boundary_features_csv=boundary_features_csv,
        output_path=output_path,
        zscore_threshold=args.zscore_threshold,
        soft_gap_seconds=args.soft_gap_seconds,
        hard_gap_seconds=args.hard_gap_seconds,
    )
    console.print(f"Wrote {row_count} photo boundary score rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
