#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

from build_photo_segments import (
    PHOTO_BOUNDARY_SCORE_FILENAME,
    PHOTO_SEGMENT_FILENAME,
    PHOTO_SEGMENT_HEADERS,
    format_float,
    parse_float,
    parse_local_datetime,
    parse_non_negative_int,
    read_boundary_scores,
    validate_boundary_alignment,
)
from lib.image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1, PHOTO_MANIFEST_REQUIRED_COLUMNS, validate_required_columns
from lib.pipeline_io import atomic_write_json


console = Console()

PHOTO_EMBEDDED_MANIFEST_FILENAME = "photo_embedded_manifest.csv"
PHOTO_REVIEW_INDEX_FILENAME = "performance_proxy_index.image.json"
MANIFEST_REQUIRED_COLUMNS = frozenset(set(PHOTO_MANIFEST_REQUIRED_COLUMNS) | {"start_local"})
EMBEDDED_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "preview_path"})
UNCERTAIN_BOUNDARY_SCORE_THRESHOLD = 0.95
IMAGE_ONLY_TIMELINE_STATUS = "image_only"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-only review index JSON from stage-1 manifests, boundaries, and segments."
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
        "--segments-csv",
        help="Input segments CSV filename or absolute path. Default: WORKSPACE/photo_segments.csv",
    )
    parser.add_argument(
        "--embedded-manifest-csv",
        help="Input embedded manifest CSV filename or absolute path. Default: WORKSPACE/photo_embedded_manifest.csv",
    )
    parser.add_argument(
        "--boundary-scores-csv",
        help="Input boundary scores CSV filename or absolute path. Default: WORKSPACE/photo_boundary_scores.csv",
    )
    parser.add_argument(
        "--output",
        default=PHOTO_REVIEW_INDEX_FILENAME,
        help=f"Output filename or absolute path. Default: {PHOTO_REVIEW_INDEX_FILENAME}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing performance_proxy_index.image.json output",
    )
    return parser.parse_args(argv)


def resolve_manifest_path(workspace_dir: Path, manifest_value: Optional[str]) -> Path:
    if not manifest_value:
        return workspace_dir / "photo_manifest.csv"
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_segments_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_SEGMENT_FILENAME
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_embedded_manifest_path(workspace_dir: Path, value: Optional[str]) -> Path:
    if not value:
        return workspace_dir / PHOTO_EMBEDDED_MANIFEST_FILENAME
    candidate = Path(value)
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
        return workspace_dir / PHOTO_REVIEW_INDEX_FILENAME
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


def normalize_workspace_relative_path(relative_path: str, column_name: str) -> Path:
    value = str(relative_path)
    if value != value.strip():
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    if "\\" in value:
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    candidate = PurePosixPath(value)
    if not value or not candidate.parts:
        raise ValueError(f"{column_name} is empty")
    if candidate.is_absolute():
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    normalized_parts: List[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
        normalized_parts.append(part)
    if not normalized_parts:
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    normalized = Path(*normalized_parts)
    if value != normalized.as_posix():
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    return normalized


def resolve_day_source_path(day_dir: Path, relative_path: str, source_value: str, column_name: str) -> Path:
    expected_relative_path = normalize_day_relative_path(relative_path, column_name)
    expected_path = (day_dir / expected_relative_path).resolve()
    source_candidate = Path(str(source_value).strip())
    source_path = source_candidate.resolve() if source_candidate.is_absolute() else (day_dir / source_candidate).resolve()
    try:
        expected_path.relative_to(day_dir.resolve())
        source_path.relative_to(day_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{column_name} must stay under {day_dir}: {source_value}") from exc
    if source_path != expected_path:
        raise ValueError(f"{column_name} does not match relative_path {relative_path}: {source_value}")
    return source_path


def resolve_workspace_path(workspace_dir: Path, relative_path: Path, column_name: str) -> Path:
    resolved_path = (workspace_dir.resolve() / relative_path).resolve()
    try:
        resolved_path.relative_to(workspace_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{column_name} must stay under workspace: {relative_path.as_posix()}") from exc
    return resolved_path


def parse_bool(value: str, column_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{column_name} must be a boolean")


def read_photo_manifest(day_dir: Path, path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    normalized_rows: List[Dict[str, str]] = []
    previous_start_dt: Optional[datetime] = None
    for row_number, row in enumerate(rows, start=1):
        relative_path = normalize_day_relative_path(str(row.get("relative_path") or ""), f"{path.name} relative_path")
        photo_order_index = parse_non_negative_int(str(row.get("photo_order_index") or ""), f"{path.name} photo_order_index")
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
        source_value = str(row.get("path") or "").strip()
        if not source_value:
            raise ValueError(f"{path.name} row missing path for {relative_path}")
        resolve_day_source_path(day_dir, relative_path, source_value, f"{path.name} path")
        normalized_rows.append(
            {
                "relative_path": relative_path,
                "path": source_value,
                "photo_order_index": str(photo_order_index),
                "start_local": start_local,
                "stream_id": str(row.get("stream_id") or "").strip(),
                "device": str(row.get("device") or "").strip(),
                "filename": str(row.get("filename") or Path(relative_path).name).strip() or Path(relative_path).name,
            }
        )
    return normalized_rows


def read_embedded_manifest(workspace_dir: Path, path: Path, manifest_rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, EMBEDDED_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if len(rows) != len(manifest_rows):
        raise ValueError(
            "Row count mismatch across stage-1 artifacts: "
            f"photo_manifest.csv={len(manifest_rows)}, {path.name}={len(rows)}"
        )
    preview_by_relative_path: Dict[str, Dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        expected_relative_path = str(manifest_rows[row_number - 1]["relative_path"])
        relative_path = normalize_day_relative_path(str(row.get("relative_path") or ""), f"{path.name} relative_path")
        if relative_path != expected_relative_path:
            raise ValueError(
                f"{path.name} relative_path mismatch at row {row_number}: "
                f"expected {expected_relative_path}, got {relative_path}"
            )
        preview_relative = normalize_workspace_relative_path(str(row.get("preview_path") or ""), f"{path.name} preview_path")
        preview_path = resolve_workspace_path(workspace_dir, preview_relative, f"{path.name} preview_path")
        if "preview_exists" in row and not parse_bool(str(row.get("preview_exists") or ""), f"{path.name} preview_exists"):
            raise ValueError(f"{path.name} preview_exists is false for {relative_path}")
        if not preview_path.exists():
            raise ValueError(f"Preview JPEG listed in {path.name} does not exist: {preview_path}")
        preview_by_relative_path[relative_path] = {
            "preview_path": preview_relative.as_posix(),
            "proxy_exists": "true",
        }
    return preview_by_relative_path


def read_segments(path: Path, manifest_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_SEGMENT_HEADERS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    manifest_index_by_path = {str(row["relative_path"]): index for index, row in enumerate(manifest_rows)}
    normalized_rows: List[Dict[str, object]] = []
    previous_end_index = -1
    for row_number, row in enumerate(rows, start=1):
        set_id = str(row.get("set_id") or "").strip()
        performance_number = str(row.get("performance_number") or "").strip()
        if not set_id:
            raise ValueError(f"{path.name} set_id is empty at row {row_number}")
        if not performance_number:
            raise ValueError(f"{path.name} performance_number is empty at row {row_number}")
        segment_index = parse_non_negative_int(str(row.get("segment_index") or ""), f"{path.name} segment_index")
        if segment_index != row_number - 1:
            raise ValueError(
                f"{path.name} segment_index contract mismatch at row {row_number}: "
                f"expected {row_number - 1}, got {segment_index}"
            )
        start_relative_path = normalize_day_relative_path(
            str(row.get("start_relative_path") or ""),
            f"{path.name} start_relative_path",
        )
        end_relative_path = normalize_day_relative_path(
            str(row.get("end_relative_path") or ""),
            f"{path.name} end_relative_path",
        )
        if start_relative_path not in manifest_index_by_path:
            raise ValueError(f"{path.name} start_relative_path not found in photo_manifest.csv: {start_relative_path}")
        if end_relative_path not in manifest_index_by_path:
            raise ValueError(f"{path.name} end_relative_path not found in photo_manifest.csv: {end_relative_path}")
        start_index = manifest_index_by_path[start_relative_path]
        end_index = manifest_index_by_path[end_relative_path]
        if start_index > end_index:
            raise ValueError(
                f"{path.name} segment range is reversed at row {row_number}: "
                f"{start_relative_path} .. {end_relative_path}"
            )
        if start_index != previous_end_index + 1:
            raise ValueError(
                f"{path.name} segment coverage mismatch at row {row_number}: "
                f"expected start index {previous_end_index + 1}, got {start_index}"
            )
        photo_count = parse_non_negative_int(str(row.get("photo_count") or ""), f"{path.name} photo_count")
        expected_photo_count = end_index - start_index + 1
        if photo_count != expected_photo_count:
            raise ValueError(
                f"{path.name} photo_count mismatch at row {row_number}: "
                f"expected {expected_photo_count}, got {photo_count}"
            )
        start_local = str(row.get("start_local") or "")
        end_local = str(row.get("end_local") or "")
        if start_local != str(manifest_rows[start_index]["start_local"]):
            raise ValueError(
                f"{path.name} start_local mismatch at row {row_number}: "
                f"expected {manifest_rows[start_index]['start_local']}, got {start_local}"
            )
        if end_local != str(manifest_rows[end_index]["start_local"]):
            raise ValueError(
                f"{path.name} end_local mismatch at row {row_number}: "
                f"expected {manifest_rows[end_index]['start_local']}, got {end_local}"
            )
        segment_confidence = parse_float(str(row.get("segment_confidence") or ""), f"{path.name} segment_confidence")
        if not 0.0 <= segment_confidence <= 1.0:
            raise ValueError(f"{path.name} segment_confidence must stay within [0, 1]: {segment_confidence}")
        normalized_rows.append(
            {
                "set_id": set_id,
                "performance_number": performance_number,
                "segment_index": str(segment_index),
                "start_relative_path": start_relative_path,
                "end_relative_path": end_relative_path,
                "start_index": start_index,
                "end_index": end_index,
                "start_local": start_local,
                "end_local": end_local,
                "photo_count": str(photo_count),
                "segment_confidence": format_float(segment_confidence),
                "segment_confidence_value": segment_confidence,
            }
        )
        previous_end_index = end_index
    if previous_end_index != len(manifest_rows) - 1:
        raise ValueError(
            f"{path.name} segment coverage mismatch: expected final photo index {len(manifest_rows) - 1}, "
            f"got {previous_end_index}"
        )
    return normalized_rows


def boundary_review_reason(
    boundary_row: Mapping[str, str],
    left_segment_confidence: float,
    right_segment_confidence: float,
) -> str:
    boundary_score = parse_float(
        str(boundary_row.get("boundary_score") or ""),
        "photo_boundary_scores.csv boundary_score",
    )
    boundary_label = str(boundary_row.get("boundary_label") or "")
    if boundary_score < UNCERTAIN_BOUNDARY_SCORE_THRESHOLD:
        return "boundary_score"
    if left_segment_confidence < UNCERTAIN_BOUNDARY_SCORE_THRESHOLD:
        return "segment_confidence"
    if right_segment_confidence < UNCERTAIN_BOUNDARY_SCORE_THRESHOLD:
        return "segment_confidence"
    if boundary_label != "hard":
        return "boundary_label"
    return ""


def nearest_boundary_seconds(
    manifest_rows: Sequence[Mapping[str, str]],
    photo_index: int,
    left_anchor_index: Optional[int],
    right_anchor_index: Optional[int],
) -> str:
    anchor_indexes = [index for index in [left_anchor_index, right_anchor_index] if index is not None]
    if not anchor_indexes:
        return ""
    photo_dt = parse_local_datetime(str(manifest_rows[photo_index]["start_local"]), "photo_manifest.csv start_local")
    distances = []
    for anchor_index in anchor_indexes:
        anchor_dt = parse_local_datetime(
            str(manifest_rows[anchor_index]["start_local"]),
            "photo_manifest.csv start_local",
        )
        distances.append(abs((photo_dt - anchor_dt).total_seconds()))
    return format_float(min(distances))


def build_photo_payload(
    manifest_row: Mapping[str, str],
    preview_row: Mapping[str, str],
    *,
    assignment_status: str,
    assignment_reason: str,
    seconds_to_boundary: str,
) -> Dict[str, object]:
    relative_path = str(manifest_row["relative_path"])
    return {
        "relative_path": relative_path,
        "source_path": relative_path,
        "proxy_path": str(preview_row["preview_path"]),
        "proxy_exists": True,
        "filename": Path(relative_path).name,
        "photo_start_local": str(manifest_row["start_local"]),
        "adjusted_start_local": str(manifest_row["start_local"]),
        "assignment_status": assignment_status,
        "assignment_reason": assignment_reason,
        "seconds_to_nearest_boundary": seconds_to_boundary,
        "stream_id": str(manifest_row.get("stream_id") or ""),
        "device": str(manifest_row.get("device") or ""),
    }


def build_performance_payloads(
    manifest_rows: Sequence[Mapping[str, str]],
    segment_rows: Sequence[Mapping[str, object]],
    preview_by_relative_path: Mapping[str, Mapping[str, str]],
    boundary_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, object]]:
    performance_rows: List[Dict[str, object]] = []
    segment_confidence_by_cut_index: Dict[int, tuple[float, float]] = {}
    for segment_index in range(len(segment_rows) - 1):
        left_segment = segment_rows[segment_index]
        right_segment = segment_rows[segment_index + 1]
        cut_index = int(left_segment["end_index"])
        if int(right_segment["start_index"]) != cut_index + 1:
            raise ValueError(
                "photo_segments.csv segment coverage mismatch between adjacent rows: "
                f"row {segment_index + 1} and row {segment_index + 2}"
            )
        segment_confidence_by_cut_index[cut_index] = (
            float(left_segment["segment_confidence_value"]),
            float(right_segment["segment_confidence_value"]),
        )

    uncertain_cut_reasons = {
        cut_index: boundary_review_reason(boundary_rows[cut_index], left_confidence, right_confidence)
        for cut_index, (left_confidence, right_confidence) in segment_confidence_by_cut_index.items()
    }
    uncertain_cut_indexes = {cut_index for cut_index, reason in uncertain_cut_reasons.items() if reason}

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
        task = progress.add_task("Build review index".ljust(25), total=len(segment_rows))
        for segment_row in segment_rows:
            start_index = int(segment_row["start_index"])
            end_index = int(segment_row["end_index"])
            left_cut_index = start_index - 1 if start_index > 0 else None
            right_cut_index = end_index if end_index < len(manifest_rows) - 1 else None
            review_first_photo = left_cut_index is not None and left_cut_index in uncertain_cut_indexes
            review_last_photo = right_cut_index is not None and right_cut_index in uncertain_cut_indexes
            photos: List[Dict[str, object]] = []
            for photo_index in range(start_index, end_index + 1):
                manifest_row = manifest_rows[photo_index]
                preview_row = preview_by_relative_path[str(manifest_row["relative_path"])]
                assignment_status = "assigned"
                assignment_reason = ""
                if review_first_photo and photo_index == start_index:
                    assignment_status = "review"
                    assignment_reason = uncertain_cut_reasons.get(left_cut_index, "") if left_cut_index is not None else ""
                if review_last_photo and photo_index == end_index:
                    assignment_status = "review"
                    assignment_reason = (
                        uncertain_cut_reasons.get(right_cut_index, "") if right_cut_index is not None else ""
                    )
                photos.append(
                    build_photo_payload(
                        manifest_row,
                        preview_row,
                        assignment_status=assignment_status,
                        assignment_reason=assignment_reason,
                        seconds_to_boundary=nearest_boundary_seconds(
                            manifest_rows,
                            photo_index,
                            start_index if left_cut_index is not None else None,
                            end_index if right_cut_index is not None else None,
                        ),
                    )
                )
            performance_rows.append(
                {
                    "set_id": str(segment_row["set_id"]),
                    "performance_number": str(segment_row["performance_number"]),
                    "segment_index": str(segment_row["segment_index"]),
                    "timeline_status": IMAGE_ONLY_TIMELINE_STATUS,
                    "duplicate_status": "normal",
                    "performance_start_local": str(segment_row["start_local"]),
                    "performance_end_local": str(segment_row["end_local"]),
                    "segment_confidence": str(segment_row["segment_confidence"]),
                    "photos": photos,
                }
            )
            progress.advance(task)
    return performance_rows


def build_photo_review_index(
    day_dir: Optional[Path] = None,
    *,
    workspace_dir: Path,
    manifest_csv: Path,
    segments_csv: Path,
    embedded_manifest_csv: Path,
    boundary_scores_csv: Path,
    output_path: Path,
) -> int:
    declared_day_dir = day_dir if day_dir is not None else workspace_dir.parent
    if workspace_dir.parent.resolve() != declared_day_dir.resolve():
        raise ValueError(
            f"workspace_dir must stay under day_dir for image-only review index: {workspace_dir} vs {declared_day_dir}"
        )
    manifest_rows = read_photo_manifest(declared_day_dir, manifest_csv)
    preview_by_relative_path = read_embedded_manifest(workspace_dir, embedded_manifest_csv, manifest_rows)
    boundary_rows = read_boundary_scores(boundary_scores_csv)
    validate_boundary_alignment(manifest_rows, boundary_rows)
    segment_rows = read_segments(segments_csv, manifest_rows)
    performances = build_performance_payloads(manifest_rows, segment_rows, preview_by_relative_path, boundary_rows)
    payload = {
        "day": declared_day_dir.name,
        "workspace_dir": str(workspace_dir),
        "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
        "performance_count": len(performances),
        "photo_count": len(manifest_rows),
        "performances": performances,
    }
    atomic_write_json(output_path, payload)
    return len(performances)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    if args.workspace_dir:
        workspace_dir = Path(args.workspace_dir).expanduser()
        if not workspace_dir.is_absolute():
            workspace_dir = Path.cwd() / workspace_dir
    else:
        workspace_dir = day_dir / "_workspace"
    manifest_csv = resolve_manifest_path(workspace_dir, args.manifest_csv)
    segments_csv = resolve_segments_path(workspace_dir, args.segments_csv)
    embedded_manifest_csv = resolve_embedded_manifest_path(workspace_dir, args.embedded_manifest_csv)
    boundary_scores_csv = resolve_boundary_scores_path(workspace_dir, args.boundary_scores_csv)
    output_path = resolve_output_path(workspace_dir, args.output)
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_csv}")
    if not segments_csv.exists():
        raise SystemExit(f"Segments CSV does not exist: {segments_csv}")
    if not embedded_manifest_csv.exists():
        raise SystemExit(f"Embedded manifest CSV does not exist: {embedded_manifest_csv}")
    if not boundary_scores_csv.exists():
        raise SystemExit(f"Boundary scores CSV does not exist: {boundary_scores_csv}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output JSON already exists: {output_path}. Use --overwrite to replace it.")
    performance_count = build_photo_review_index(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        manifest_csv=manifest_csv,
        segments_csv=segments_csv,
        embedded_manifest_csv=embedded_manifest_csv,
        boundary_scores_csv=boundary_scores_csv,
        output_path=output_path,
    )
    console.print(
        f"Wrote {performance_count} image-only review sets "
        f"({len(read_photo_manifest(day_dir, manifest_csv))} photos) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
