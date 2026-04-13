#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

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

from bootstrap_photo_boundaries import PHOTO_BOUNDARY_SCORE_HEADERS
from lib.image_pipeline_contracts import PHOTO_MANIFEST_REQUIRED_COLUMNS, validate_required_columns
from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_SEGMENT_FILENAME = "photo_segments.csv"
PHOTO_BOUNDARY_SCORE_FILENAME = "photo_boundary_scores.csv"

DEFAULT_MIN_SEGMENT_PHOTOS = 8
DEFAULT_MIN_SEGMENT_SECONDS = 5.0

LOCAL_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3}|\.\d{6})?$")
CANONICAL_NON_NEGATIVE_INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")

PHOTO_SEGMENT_HEADERS = [
    "set_id",
    "performance_number",
    "segment_index",
    "start_relative_path",
    "end_relative_path",
    "start_local",
    "end_local",
    "photo_count",
    "segment_confidence",
]

PHOTO_SEGMENT_MANIFEST_REQUIRED_COLUMNS = frozenset(set(PHOTO_MANIFEST_REQUIRED_COLUMNS) | {"start_local"})


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic image-only stage-1 photo segments from boundary scores."
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
        "--boundary-scores-csv",
        help="Input boundary score CSV filename or absolute path. Default: WORKSPACE/photo_boundary_scores.csv",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename or absolute path. Default: WORKSPACE/photo_segments.csv",
    )
    parser.add_argument(
        "--min-segment-photos",
        type=positive_int_arg,
        default=DEFAULT_MIN_SEGMENT_PHOTOS,
        help=f"Minimum photos per segment before merge safeguards apply. Default: {DEFAULT_MIN_SEGMENT_PHOTOS}",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=positive_float_arg,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help=f"Minimum segment span in seconds before merge safeguards apply. Default: {DEFAULT_MIN_SEGMENT_SECONDS}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing photo_segments.csv output",
    )
    return parser.parse_args(argv)


def resolve_manifest_path(workspace_dir: Path, manifest_value: Optional[str]) -> Path:
    if not manifest_value:
        return workspace_dir / "photo_manifest.csv"
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_boundary_scores_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_BOUNDARY_SCORE_FILENAME
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_output_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_SEGMENT_FILENAME
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


def parse_non_negative_int(value: str, column_name: str) -> int:
    text = str(value)
    if not text:
        raise ValueError(f"{column_name} is empty")
    if not CANONICAL_NON_NEGATIVE_INT_RE.match(text):
        raise ValueError(f"{column_name} must be a canonical non-negative integer: {value}")
    return int(text)


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


def format_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}"


def read_photo_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_SEGMENT_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    normalized_rows: List[Dict[str, str]] = []
    previous_start_dt: Optional[datetime] = None
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
        start_local = str(row.get("start_local") or "")
        start_dt = parse_local_datetime(start_local, f"{path.name} start_local")
        if previous_start_dt is not None and start_dt < previous_start_dt:
            raise ValueError(
                f"{path.name} start_local must be non-decreasing at row {row_number}: "
                f"{start_local} is earlier than previous row"
            )
        previous_start_dt = start_dt
        normalized_rows.append(
            {
                "relative_path": relative_path,
                "path": str(row.get("path") or ""),
                "photo_order_index": str(photo_order_index),
                "start_local": start_local,
            }
        )
    return normalized_rows


def read_boundary_scores(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_BOUNDARY_SCORE_HEADERS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    normalized_rows: List[Dict[str, str]] = []
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
        parse_local_datetime(left_start_local, f"{path.name} left_start_local")
        parse_local_datetime(right_start_local, f"{path.name} right_start_local")
        parse_float(str(row.get("time_gap_seconds") or ""), f"{path.name} time_gap_seconds")
        parse_float(str(row.get("dino_cosine_distance") or ""), f"{path.name} dino_cosine_distance")
        parse_float(str(row.get("distance_zscore") or ""), f"{path.name} distance_zscore")
        parse_float(str(row.get("smoothed_distance_zscore") or ""), f"{path.name} smoothed_distance_zscore")
        parse_float(str(row.get("time_gap_boost") or ""), f"{path.name} time_gap_boost")
        boundary_score = parse_float(str(row.get("boundary_score") or ""), f"{path.name} boundary_score")
        if not 0.0 <= boundary_score <= 1.0:
            raise ValueError(f"{path.name} boundary_score must stay within [0, 1]: {boundary_score}")
        boundary_label = str(row.get("boundary_label") or "")
        if boundary_label not in {"none", "soft", "hard"}:
            raise ValueError(f"{path.name} boundary_label must be one of none, soft, hard: {boundary_label}")
        if boundary_label == "none" and boundary_score >= 0.75:
            raise ValueError(f"{path.name} boundary_label none cannot carry cut score {boundary_score}")
        if boundary_label in {"soft", "hard"} and boundary_score < 0.75:
            raise ValueError(f"{path.name} boundary_label {boundary_label} requires boundary_score >= 0.75")
        boundary_reason = str(row.get("boundary_reason") or "")
        if not boundary_reason:
            raise ValueError(f"{path.name} boundary_reason is empty")
        model_source = str(row.get("model_source") or "")
        if not model_source:
            raise ValueError(f"{path.name} model_source is empty")
        if previous_right_path is not None and left_relative_path != previous_right_path:
            raise ValueError(
                f"{path.name} relative_path mismatch at row {row_number}: expected left_relative_path "
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
                "time_gap_seconds": format_float(
                    parse_float(str(row.get("time_gap_seconds") or ""), f"{path.name} time_gap_seconds")
                ),
                "dino_cosine_distance": format_float(
                    parse_float(str(row.get("dino_cosine_distance") or ""), f"{path.name} dino_cosine_distance")
                ),
                "distance_zscore": format_float(
                    parse_float(str(row.get("distance_zscore") or ""), f"{path.name} distance_zscore")
                ),
                "smoothed_distance_zscore": format_float(
                    parse_float(str(row.get("smoothed_distance_zscore") or ""), f"{path.name} smoothed_distance_zscore")
                ),
                "time_gap_boost": format_float(
                    parse_float(str(row.get("time_gap_boost") or ""), f"{path.name} time_gap_boost")
                ),
                "boundary_score": format_float(boundary_score),
                "boundary_label": boundary_label,
                "boundary_reason": boundary_reason,
                "model_source": model_source,
            }
        )
    return normalized_rows


def validate_boundary_alignment(
    manifest_rows: Sequence[Mapping[str, str]],
    boundary_rows: Sequence[Mapping[str, str]],
) -> None:
    expected_boundary_count = max(len(manifest_rows) - 1, 0)
    if len(boundary_rows) != expected_boundary_count:
        raise ValueError(
            "Row count mismatch across stage-1 artifacts: "
            f"photo_manifest.csv={len(manifest_rows)}, photo_boundary_scores.csv={len(boundary_rows)}"
        )
    for row_number, boundary_row in enumerate(boundary_rows, start=1):
        expected_left_path = str(manifest_rows[row_number - 1]["relative_path"])
        expected_right_path = str(manifest_rows[row_number]["relative_path"])
        if str(boundary_row["left_relative_path"]) != expected_left_path:
            raise ValueError(
                f"photo_boundary_scores.csv relative_path mismatch at row {row_number}: "
                f"expected {expected_left_path}, got {boundary_row['left_relative_path']}"
            )
        if str(boundary_row["right_relative_path"]) != expected_right_path:
            raise ValueError(
                f"photo_boundary_scores.csv relative_path mismatch at row {row_number}: "
                f"expected {expected_right_path}, got {boundary_row['right_relative_path']}"
            )
        if str(boundary_row["left_start_local"]) != str(manifest_rows[row_number - 1]["start_local"]):
            raise ValueError(
                f"photo_boundary_scores.csv start_local mismatch at row {row_number}: "
                f"expected {manifest_rows[row_number - 1]['start_local']}, got {boundary_row['left_start_local']}"
            )
        if str(boundary_row["right_start_local"]) != str(manifest_rows[row_number]["start_local"]):
            raise ValueError(
                f"photo_boundary_scores.csv start_local mismatch at row {row_number}: "
                f"expected {manifest_rows[row_number]['start_local']}, got {boundary_row['right_start_local']}"
            )


def build_segment_ranges(photo_count: int, cut_indexes: Sequence[int]) -> List[Tuple[int, int]]:
    if photo_count <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    segment_start = 0
    for cut_index in sorted(cut_indexes):
        ranges.append((segment_start, cut_index))
        segment_start = cut_index + 1
    ranges.append((segment_start, photo_count - 1))
    return ranges


def segment_duration_seconds(segment_range: Tuple[int, int], manifest_rows: Sequence[Mapping[str, str]]) -> float:
    start_dt = parse_local_datetime(str(manifest_rows[segment_range[0]]["start_local"]), "photo_manifest.csv start_local")
    end_dt = parse_local_datetime(str(manifest_rows[segment_range[1]]["start_local"]), "photo_manifest.csv start_local")
    return max(0.0, float((end_dt - start_dt).total_seconds()))


def choose_cut_to_remove(
    segment_range: Tuple[int, int],
    boundary_rows: Sequence[Mapping[str, str]],
    kept_cuts: Set[int],
) -> int:
    start_index, end_index = segment_range
    left_cut = start_index - 1 if start_index > 0 and (start_index - 1) in kept_cuts else None
    right_cut = end_index if end_index < len(boundary_rows) and end_index in kept_cuts else None
    if left_cut is None and right_cut is None:
        raise ValueError("Cannot merge an undersized segment without an adjacent cut")
    if left_cut is None:
        return right_cut  # type: ignore[return-value]
    if right_cut is None:
        return left_cut
    left_score = parse_float(str(boundary_rows[left_cut]["boundary_score"]), "photo_boundary_scores.csv boundary_score")
    right_score = parse_float(str(boundary_rows[right_cut]["boundary_score"]), "photo_boundary_scores.csv boundary_score")
    return left_cut if left_score <= right_score else right_cut


def apply_minimum_segment_safeguards(
    manifest_rows: Sequence[Mapping[str, str]],
    boundary_rows: Sequence[Mapping[str, str]],
    min_segment_photos: int,
    min_segment_seconds: float,
) -> List[int]:
    kept_cuts: Set[int] = {
        index
        for index, row in enumerate(boundary_rows)
        if str(row["boundary_label"]) in {"soft", "hard"}
    }
    while True:
        ranges = build_segment_ranges(len(manifest_rows), sorted(kept_cuts))
        if len(ranges) <= 1:
            return sorted(kept_cuts)
        invalid_range: Optional[Tuple[int, int]] = None
        for segment_range in ranges:
            photo_count = segment_range[1] - segment_range[0] + 1
            duration_seconds = segment_duration_seconds(segment_range, manifest_rows)
            if photo_count < min_segment_photos or duration_seconds < min_segment_seconds:
                invalid_range = segment_range
                break
        if invalid_range is None:
            return sorted(kept_cuts)
        kept_cuts.remove(choose_cut_to_remove(invalid_range, boundary_rows, kept_cuts))


def segment_confidence(
    segment_range: Tuple[int, int],
    boundary_rows: Sequence[Mapping[str, str]],
    kept_cuts: Set[int],
) -> float:
    confidences: List[float] = []
    start_index, end_index = segment_range
    if start_index > 0 and (start_index - 1) in kept_cuts:
        confidences.append(
            parse_float(
                str(boundary_rows[start_index - 1]["boundary_score"]),
                "photo_boundary_scores.csv boundary_score",
            )
        )
    if end_index < len(boundary_rows) and end_index in kept_cuts:
        confidences.append(
            parse_float(
                str(boundary_rows[end_index]["boundary_score"]),
                "photo_boundary_scores.csv boundary_score",
            )
        )
    if not confidences:
        return 0.0
    return sum(confidences) / float(len(confidences))


def build_segment_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    boundary_rows: Sequence[Mapping[str, str]],
    min_segment_photos: int,
    min_segment_seconds: float,
) -> List[Dict[str, str]]:
    kept_cut_indexes = apply_minimum_segment_safeguards(
        manifest_rows,
        boundary_rows,
        min_segment_photos,
        min_segment_seconds,
    )
    kept_cuts = set(kept_cut_indexes)
    segment_ranges = build_segment_ranges(len(manifest_rows), kept_cut_indexes)
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
        task = progress.add_task("Build segments".ljust(25), total=len(segment_ranges))
        for segment_index, segment_range in enumerate(segment_ranges):
            start_index, end_index = segment_range
            rows.append(
                {
                    "set_id": f"imgset-{segment_index + 1:06d}",
                    "performance_number": f"SEG{segment_index + 1:04d}",
                    "segment_index": str(segment_index),
                    "start_relative_path": str(manifest_rows[start_index]["relative_path"]),
                    "end_relative_path": str(manifest_rows[end_index]["relative_path"]),
                    "start_local": str(manifest_rows[start_index]["start_local"]),
                    "end_local": str(manifest_rows[end_index]["start_local"]),
                    "photo_count": str(end_index - start_index + 1),
                    "segment_confidence": format_float(segment_confidence(segment_range, boundary_rows, kept_cuts)),
                }
            )
            progress.advance(task)
    return rows


def build_photo_segments(
    workspace_dir: Path,
    manifest_csv: Path,
    boundary_scores_csv: Path,
    output_path: Path,
    min_segment_photos: int = DEFAULT_MIN_SEGMENT_PHOTOS,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
) -> int:
    _ = workspace_dir
    if min_segment_photos <= 0:
        raise ValueError("min_segment_photos must be positive")
    if min_segment_seconds <= 0.0:
        raise ValueError("min_segment_seconds must be positive")
    manifest_rows = read_photo_manifest(manifest_csv)
    boundary_rows = read_boundary_scores(boundary_scores_csv)
    validate_boundary_alignment(manifest_rows, boundary_rows)
    segment_rows = build_segment_rows(
        manifest_rows,
        boundary_rows,
        min_segment_photos=min_segment_photos,
        min_segment_seconds=min_segment_seconds,
    )
    atomic_write_csv(output_path, PHOTO_SEGMENT_HEADERS, segment_rows)
    return len(segment_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    manifest_csv = resolve_manifest_path(workspace_dir, args.manifest_csv)
    boundary_scores_csv = resolve_boundary_scores_path(workspace_dir, args.boundary_scores_csv)
    output_path = resolve_output_path(workspace_dir, args.output)
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_csv}")
    if not boundary_scores_csv.exists():
        raise SystemExit(f"Boundary scores CSV does not exist: {boundary_scores_csv}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output CSV already exists: {output_path}. Use --overwrite to replace it.")
    row_count = build_photo_segments(
        workspace_dir=workspace_dir,
        manifest_csv=manifest_csv,
        boundary_scores_csv=boundary_scores_csv,
        output_path=output_path,
        min_segment_photos=args.min_segment_photos,
        min_segment_seconds=args.min_segment_seconds,
    )
    console.print(f"Wrote {row_count} photo segment rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
