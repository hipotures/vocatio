#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

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

from lib.media_manifest import read_media_manifest, select_photo_rows
from lib.pipeline_io import atomic_write_csv, atomic_write_json
from lib.workspace_dir import resolve_workspace_dir
from lib.ml_boundary_dataset import canonical_candidate_id, normalize_timestamp, sort_photo_rows
from lib.ml_boundary_truth import FinalPhotoTruth, build_final_photo_truth
from lib.window_radius_contract import positive_window_radius_arg, window_radius_to_window_size


console = Console()

DEFAULT_WINDOW_RADIUS = 2
DEFAULT_CANDIDATE_RULE_NAME = "gap_threshold"
DEFAULT_TRUTH_FILENAME = "ml_boundary_reviewed_truth.csv"
DEFAULT_OUTPUT_FILENAME = "ml_boundary_candidates.csv"
DEFAULT_ATTRITION_FILENAME = "ml_boundary_attrition.json"
DEFAULT_REPORT_FILENAME = "ml_boundary_dataset_report.json"
DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1 = "not_included_v1"


def frame_numbers_for_radius(window_radius: int) -> list[int]:
    return list(range(1, window_radius_to_window_size(window_radius) + 1))


def candidate_row_headers(*, window_radius: int, include_thumbnail: bool) -> list[str]:
    headers = [
        "candidate_id",
        "day_id",
        "window_radius",
        "center_left_photo_id",
        "center_right_photo_id",
        "left_segment_id",
        "right_segment_id",
        "left_segment_type",
        "right_segment_type",
        "segment_type",
        "boundary",
        "candidate_rule_name",
        "candidate_rule_version",
        "candidate_rule_params_json",
        "descriptor_schema_version",
        "split_name",
        "window_photo_ids",
        "window_relative_paths",
    ]
    for frame_index in frame_numbers_for_radius(window_radius):
        suffix = f"{frame_index:02d}"
        headers.extend(
            [
                f"frame_{suffix}_photo_id",
                f"frame_{suffix}_relpath",
                f"frame_{suffix}_timestamp",
            ]
        )
        if include_thumbnail:
            headers.append(f"frame_{suffix}_thumb_path")
        headers.append(f"frame_{suffix}_preview_path")
    return headers


CANDIDATE_ROW_HEADERS = candidate_row_headers(
    window_radius=DEFAULT_WINDOW_RADIUS,
    include_thumbnail=True,
)
ATTRITION_REPORT_KEYS = [
    "candidate_count_generated",
    "candidate_count_excluded_missing_window",
    "candidate_count_excluded_missing_artifacts",
    "candidate_count_retained",
    "true_boundary_coverage_before_exclusions",
    "true_boundary_coverage_after_exclusions",
]


def _parse_gap_threshold_seconds(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"gap threshold seconds must be a finite number: {value}"
        ) from exc
    if not math.isfinite(threshold):
        raise argparse.ArgumentTypeError(
            f"gap threshold seconds must be a finite number: {value}"
        )
    if threshold <= 0.0:
        raise argparse.ArgumentTypeError(
            f"gap threshold seconds must be greater than zero: {value}"
        )
    return threshold


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ML boundary candidate dataset CSV and attrition reports from media_manifest.csv and reviewed truth."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds ML boundary artifacts. Default: DAY/.vocatio WORKSPACE_DIR or DAY/_workspace",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Input media manifest filename or absolute path. Default: WORKSPACE/media_manifest.csv",
    )
    parser.add_argument(
        "--truth-csv",
        help=f"Input reviewed truth CSV filename or absolute path. Default: WORKSPACE/{DEFAULT_TRUTH_FILENAME}",
    )
    parser.add_argument(
        "--output-csv",
        help=f"Output candidate CSV filename or absolute path. Default: WORKSPACE/{DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--attrition-json",
        help=f"Output attrition JSON filename or absolute path. Default: WORKSPACE/{DEFAULT_ATTRITION_FILENAME}",
    )
    parser.add_argument(
        "--report-json",
        help=f"Output dataset report JSON filename or absolute path. Default: WORKSPACE/{DEFAULT_REPORT_FILENAME}",
    )
    parser.add_argument(
        "--gap-threshold-seconds",
        type=_parse_gap_threshold_seconds,
        default=20.0,
        help="Minimum center-gap size in seconds for generating a candidate. Default: 20.0",
    )
    parser.add_argument(
        "--window-radius",
        type=positive_window_radius_arg,
        default=DEFAULT_WINDOW_RADIUS,
        help=f"Number of frames per side in each candidate window. Default: {DEFAULT_WINDOW_RADIUS}",
    )
    parser.add_argument(
        "--candidate-rule-version",
        default="gap-v1",
        help="Candidate generation rule version recorded in the dataset. Default: gap-v1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args(argv)


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    )


def _require_non_blank_string(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    return str(value)


def _extract_relpath(row: Mapping[str, object]) -> str:
    for field_name in ("relpath", "relative_path"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("relpath is required and must not be blank")


def _extract_optional_workspace_path(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        return ""
    return str(value).strip()


def _require_window_radius(value: object, *, field_name: str = "window_radius") -> int:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        window_radius = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if window_radius < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return window_radius


def _build_rule_params_json(*, gap_threshold_seconds: float, window_radius: int) -> str:
    return json.dumps(
        {
            "gap_threshold_seconds": float(gap_threshold_seconds),
            "window_radius": int(window_radius),
        },
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )


def _candidate_identity_rule_version(
    *,
    candidate_rule_version: str,
    candidate_rule_params_json: str,
) -> str:
    return f"{candidate_rule_version}|{candidate_rule_params_json}"


def _resolve_workspace_path(workspace_dir: Path, value: Optional[str], default_name: str) -> Path:
    if not value:
        return workspace_dir / default_name
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def _validate_distinct_paths(
    *,
    manifest_path: Path,
    truth_path: Path,
    output_csv_path: Path,
    attrition_json_path: Path,
    report_json_path: Path,
) -> None:
    labeled_paths = {
        "manifest_csv": manifest_path.resolve(),
        "truth_csv": truth_path.resolve(),
        "output_csv": output_csv_path.resolve(),
        "attrition_json": attrition_json_path.resolve(),
        "report_json": report_json_path.resolve(),
    }
    seen: dict[Path, str] = {}
    for label, path in labeled_paths.items():
        prior = seen.get(path)
        if prior is not None:
            raise ValueError(f"Path collision between {prior} and {label}: {path}")
        seen[path] = label


def _read_reviewed_truth_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Truth CSV does not exist: {path}. "
            "Run export_ml_boundary_reviewed_truth.py DAY first."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"photo_id", "segment_id", "segment_type"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"{path.name} is empty")
        return rows


def _manifest_photo_to_candidate_row(row: Mapping[str, str]) -> dict[str, object]:
    start_epoch_ms = str(row.get("start_epoch_ms") or "").strip()
    if not start_epoch_ms:
        raise ValueError("start_epoch_ms is required and must not be blank")
    photo_id = str(row.get("photo_id") or "").strip() or "<unknown>"

    try:
        start_epoch_ms_value = float(start_epoch_ms)
    except ValueError as exc:
        raise ValueError(
            f"Invalid start_epoch_ms for photo_id={photo_id}: {start_epoch_ms}"
        ) from exc
    if not math.isfinite(start_epoch_ms_value):
        raise ValueError(
            f"Invalid start_epoch_ms for photo_id={photo_id}: {start_epoch_ms}"
        )

    timestamp_seconds = start_epoch_ms_value / 1000.0
    return {
        "photo_id": row.get("photo_id", ""),
        "order_idx": row.get("photo_order_index", ""),
        "timestamp": timestamp_seconds,
        "relative_path": row.get("relative_path", ""),
        "thumb_path": row.get("thumb_path", ""),
        "preview_path": row.get("preview_path", ""),
    }


def _serialize_candidate_row(row: Mapping[str, object]) -> dict[str, object]:
    window_radius = _require_window_radius(row.get("window_radius"))
    headers = candidate_row_headers(
        window_radius=window_radius,
        include_thumbnail=True,
    )
    serialized: dict[str, object] = {}
    for header in headers:
        value = row.get(header, "")
        if header in {"window_photo_ids", "window_relative_paths"}:
            serialized[header] = json.dumps(
                value if isinstance(value, list) else [],
                separators=(",", ":"),
                ensure_ascii=True,
            )
            continue
        serialized[header] = value
    return serialized


def _build_dataset_report(
    *,
    day_id: str,
    gap_threshold_seconds: float,
    candidate_rule_version: str,
    window_radius: int,
    candidate_rule_params_json: str,
    descriptor_schema_version: str,
    attrition: Mapping[str, int],
) -> dict[str, object]:
    return {
        "day_id": day_id,
        "candidate_rule_name": DEFAULT_CANDIDATE_RULE_NAME,
        "candidate_rule_version": candidate_rule_version,
        "candidate_rule_params_json": candidate_rule_params_json,
        "descriptor_schema_version": descriptor_schema_version,
        "window_radius": int(window_radius),
        "gap_threshold_seconds": float(gap_threshold_seconds),
        **{key: int(attrition[key]) for key in ATTRITION_REPORT_KEYS},
    }


def build_candidate_rows(
    *,
    photos: list[dict[str, object]],
    truth: Mapping[str, FinalPhotoTruth],
    gap_threshold_seconds: float,
    day_id: str,
    candidate_rule_version: str,
    window_radius: int = DEFAULT_WINDOW_RADIUS,
    candidate_rule_name: str = DEFAULT_CANDIDATE_RULE_NAME,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if gap_threshold_seconds <= 0.0:
        raise ValueError("gap_threshold_seconds must be greater than zero")
    if not day_id.strip():
        raise ValueError("day_id is required and must not be blank")
    if not candidate_rule_version.strip():
        raise ValueError("candidate_rule_version is required and must not be blank")
    if not candidate_rule_name.strip():
        raise ValueError("candidate_rule_name is required and must not be blank")

    ordered_photos = sort_photo_rows(photos)
    candidate_rows: list[dict[str, object]] = []
    report = {
        "candidate_count_generated": 0,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 0,
        "true_boundary_coverage_before_exclusions": 0,
        "true_boundary_coverage_after_exclusions": 0,
    }
    candidate_rule_params_json = _build_rule_params_json(
        gap_threshold_seconds=gap_threshold_seconds,
        window_radius=window_radius,
    )
    window_size = window_radius_to_window_size(window_radius)

    for index in range(len(ordered_photos) - 1):
        left_photo = ordered_photos[index]
        right_photo = ordered_photos[index + 1]
        center_gap_seconds = normalize_timestamp(right_photo.get("timestamp")) - normalize_timestamp(
            left_photo.get("timestamp")
        )
        if center_gap_seconds <= gap_threshold_seconds:
            continue

        report["candidate_count_generated"] += 1

        try:
            left_photo_id = _require_non_blank_string(left_photo, "photo_id")
            right_photo_id = _require_non_blank_string(right_photo, "photo_id")
            left_truth = truth[left_photo_id]
            right_truth = truth[right_photo_id]
            is_true_boundary = left_truth.segment_id != right_truth.segment_id
        except (KeyError, ValueError):
            report["candidate_count_excluded_missing_artifacts"] += 1
            continue

        if is_true_boundary:
            report["true_boundary_coverage_before_exclusions"] += 1

        window_start = index - window_radius + 1
        window_end = window_start + window_size
        if window_start < 0 or window_end > len(ordered_photos):
            report["candidate_count_excluded_missing_window"] += 1
            continue

        try:
            window = ordered_photos[window_start:window_end]

            row: dict[str, object] = {
                "candidate_id": canonical_candidate_id(
                    day_id=day_id,
                    center_left_photo_id=left_photo_id,
                    center_right_photo_id=right_photo_id,
                    candidate_rule_version=_candidate_identity_rule_version(
                        candidate_rule_version=candidate_rule_version,
                        candidate_rule_params_json=candidate_rule_params_json,
                    ),
                ),
                "day_id": day_id,
                "window_radius": window_radius,
                "center_left_photo_id": left_photo_id,
                "center_right_photo_id": right_photo_id,
                "left_segment_id": left_truth.segment_id,
                "right_segment_id": right_truth.segment_id,
                "left_segment_type": left_truth.segment_type,
                "right_segment_type": right_truth.segment_type,
                "segment_type": right_truth.segment_type,
                "boundary": left_truth.segment_id != right_truth.segment_id,
                "candidate_rule_name": candidate_rule_name,
                "candidate_rule_version": candidate_rule_version,
                "candidate_rule_params_json": candidate_rule_params_json,
                "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1,
                "split_name": "",
                "window_photo_ids": [],
                "window_relative_paths": [],
            }

            for frame_offset, frame in enumerate(window, start=1):
                suffix = f"{frame_offset:02d}"
                frame_photo_id = _require_non_blank_string(frame, "photo_id")
                frame_relpath = _extract_relpath(frame)
                frame_timestamp = normalize_timestamp(frame.get("timestamp"))
                row[f"frame_{suffix}_photo_id"] = frame_photo_id
                row[f"frame_{suffix}_relpath"] = frame_relpath
                row[f"frame_{suffix}_timestamp"] = frame_timestamp
                row["window_photo_ids"].append(frame_photo_id)
                row["window_relative_paths"].append(frame_relpath)
                row[f"frame_{suffix}_thumb_path"] = _extract_optional_workspace_path(
                    frame, "thumb_path"
                )
                row[f"frame_{suffix}_preview_path"] = _extract_optional_workspace_path(
                    frame, "preview_path"
                )

        except (KeyError, ValueError):
            report["candidate_count_excluded_missing_artifacts"] += 1
            continue

        candidate_rows.append(row)
        report["candidate_count_retained"] += 1
        if is_true_boundary:
            report["true_boundary_coverage_after_exclusions"] += 1

    return candidate_rows, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).expanduser().resolve()
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    manifest_path = _resolve_workspace_path(workspace_dir, args.manifest_csv, "media_manifest.csv")
    truth_path = _resolve_workspace_path(workspace_dir, args.truth_csv, DEFAULT_TRUTH_FILENAME)
    output_csv_path = _resolve_workspace_path(workspace_dir, args.output_csv, DEFAULT_OUTPUT_FILENAME)
    attrition_json_path = _resolve_workspace_path(
        workspace_dir, args.attrition_json, DEFAULT_ATTRITION_FILENAME
    )
    report_json_path = _resolve_workspace_path(workspace_dir, args.report_json, DEFAULT_REPORT_FILENAME)
    _validate_distinct_paths(
        manifest_path=manifest_path,
        truth_path=truth_path,
        output_csv_path=output_csv_path,
        attrition_json_path=attrition_json_path,
        report_json_path=report_json_path,
    )

    if not args.overwrite:
        existing_outputs = [
            path
            for path in (output_csv_path, attrition_json_path, report_json_path)
            if path.exists()
        ]
        if existing_outputs:
            raise SystemExit(
                f"Refusing to overwrite existing outputs: {', '.join(str(path) for path in existing_outputs)}"
            )

    day_id = day_dir.name

    with _progress() as progress:
        read_manifest_task = progress.add_task("Read manifest".ljust(25), total=1)
        manifest_rows = read_media_manifest(manifest_path)
        photo_rows = select_photo_rows(manifest_rows)
        if not photo_rows:
            raise ValueError(f"{manifest_path.name} contains no photo rows")
        candidate_photo_rows = [_manifest_photo_to_candidate_row(row) for row in photo_rows]
        progress.advance(read_manifest_task)

        read_truth_task = progress.add_task("Read reviewed truth".ljust(25), total=1)
        truth_rows = _read_reviewed_truth_csv(truth_path)
        truth = build_final_photo_truth(truth_rows)
        progress.advance(read_truth_task)

        build_task = progress.add_task("Build candidates".ljust(25), total=1)
        candidate_rows, attrition = build_candidate_rows(
            photos=candidate_photo_rows,
            truth=truth,
            gap_threshold_seconds=args.gap_threshold_seconds,
            day_id=day_id,
            candidate_rule_version=args.candidate_rule_version,
            window_radius=args.window_radius,
        )
        progress.advance(build_task)

        write_task = progress.add_task("Write outputs".ljust(25), total=3)
        serialized_rows = [_serialize_candidate_row(row) for row in candidate_rows]
        dataset_report = _build_dataset_report(
            day_id=day_id,
            gap_threshold_seconds=args.gap_threshold_seconds,
            candidate_rule_version=args.candidate_rule_version,
            window_radius=args.window_radius,
            candidate_rule_params_json=_build_rule_params_json(
                gap_threshold_seconds=args.gap_threshold_seconds,
                window_radius=args.window_radius,
            ),
            descriptor_schema_version=DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1,
            attrition=attrition,
        )
        headers = candidate_row_headers(window_radius=args.window_radius, include_thumbnail=True)
        atomic_write_csv(output_csv_path, headers, serialized_rows)
        progress.advance(write_task)
        atomic_write_json(
            attrition_json_path,
            {key: int(attrition[key]) for key in ATTRITION_REPORT_KEYS},
        )
        progress.advance(write_task)
        atomic_write_json(report_json_path, dataset_report)
        progress.advance(write_task)

    console.print(
        f"Wrote {len(candidate_rows)} ML boundary candidate row(s) to {output_csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
