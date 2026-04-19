#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
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

from build_ml_boundary_candidate_dataset import (
    ATTRITION_REPORT_KEYS,
    DEFAULT_ATTRITION_FILENAME,
    DEFAULT_OUTPUT_FILENAME,
    candidate_row_headers,
)
from lib.ml_boundary_dataset import canonical_candidate_id, normalize_timestamp
from lib.ml_boundary_truth import VALID_SEGMENT_TYPES
from lib.window_radius_contract import window_radius_to_window_size
from lib.workspace_dir import resolve_workspace_dir


console = Console(stderr=True)

DEFAULT_REPORT_FILENAME = "ml_boundary_validation_report.json"
DEFAULT_CORPUS_CANDIDATES_FILENAME = "ml_boundary_candidates.corpus.csv"
HELDOUT_SPLIT_NAMES = ("validation", "test")
VALID_SPLIT_NAMES = {"train", "validation", "test"}
LEGACY_EXTERNAL_COLUMNS = ("window_size", "overlap")
STATIC_CANDIDATE_HEADERS = [
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


def _require_fieldnames(path: Path, fieldnames: Sequence[str], required: Sequence[str]) -> None:
    available = set(fieldnames)
    missing = [field_name for field_name in required if field_name not in available]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")


def _read_csv_rows(path: Path, *, required_headers: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        legacy_columns = [
            field_name
            for field_name in LEGACY_EXTERNAL_COLUMNS
            if field_name in (reader.fieldnames or [])
        ]
        if legacy_columns:
            raise ValueError(
                f"{path.name} legacy columns are not allowed: {', '.join(legacy_columns)}"
            )
        _require_fieldnames(path, reader.fieldnames or (), required_headers)
        return [dict(row) for row in reader]


def _detect_split_manifest_key(fieldnames: Sequence[str]) -> str:
    available = set(fieldnames)
    has_candidate_id = {"candidate_id", "split_name"} <= available
    has_day_id = {"day_id", "split_name"} <= available
    if has_candidate_id and has_day_id:
        raise ValueError("split manifest must not contain both day_id and candidate_id columns")
    if has_candidate_id:
        return "candidate_id"
    if has_day_id:
        return "day_id"
    raise ValueError("split manifest must contain either day_id/split_name or candidate_id/split_name")


def _read_split_manifest_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest_key = _detect_split_manifest_key(reader.fieldnames or ())
        return [dict(row) for row in reader], manifest_key


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _parse_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if str(parsed) != str(value).strip():
        raise ValueError(f"{field_name} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _parse_positive_int(value: object, *, field_name: str) -> int:
    parsed = _parse_non_negative_int(value, field_name=field_name)
    if parsed < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return parsed


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field_name} must be a boolean")


def _require_non_blank_text(row: Mapping[str, object], field_name: str, *, row_number: int) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"row {row_number}: {field_name} must not be blank")
    return str(value).strip()


def _require_string_value(row: Mapping[str, object], field_name: str, *, row_number: int) -> str:
    value = row.get(field_name)
    if value is None:
        raise ValueError(f"row {row_number}: {field_name} must be present")
    if not isinstance(value, str):
        raise ValueError(f"row {row_number}: {field_name} must be a string")
    return value


def _parse_json_array_of_strings(
    value: object,
    *,
    field_name: str,
    row_number: int,
    expected_length: int,
) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_number}: {field_name} must be a non-blank JSON array")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"row {row_number}: {field_name} must decode to a JSON array")
    if len(parsed) != expected_length:
        raise ValueError(
            f"row {row_number}: {field_name} must contain exactly {expected_length} items"
        )
    values: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"row {row_number}: {field_name} must contain only non-blank strings"
            )
        values.append(item)
    return values


def _parse_json_object(
    value: object,
    *,
    field_name: str,
    row_number: int,
) -> dict[str, object]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_number}: {field_name} must be a non-blank JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"row {row_number}: {field_name} must decode to a JSON object")
    return parsed


def validate_attrition_report(report: Mapping[str, object]) -> None:
    missing = [key for key in ATTRITION_REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"Attrition report missing required keys: {', '.join(missing)}")

    generated = _parse_non_negative_int(
        report["candidate_count_generated"],
        field_name="candidate_count_generated",
    )
    excluded_missing_window = _parse_non_negative_int(
        report["candidate_count_excluded_missing_window"],
        field_name="candidate_count_excluded_missing_window",
    )
    excluded_missing_artifacts = _parse_non_negative_int(
        report["candidate_count_excluded_missing_artifacts"],
        field_name="candidate_count_excluded_missing_artifacts",
    )
    retained = _parse_non_negative_int(
        report["candidate_count_retained"],
        field_name="candidate_count_retained",
    )
    before = _parse_non_negative_int(
        report["true_boundary_coverage_before_exclusions"],
        field_name="true_boundary_coverage_before_exclusions",
    )
    after = _parse_non_negative_int(
        report["true_boundary_coverage_after_exclusions"],
        field_name="true_boundary_coverage_after_exclusions",
    )

    if generated != excluded_missing_window + excluded_missing_artifacts + retained:
        raise ValueError(
            "candidate_count_generated must equal "
            "candidate_count_excluded_missing_window + "
            "candidate_count_excluded_missing_artifacts + candidate_count_retained"
        )
    if after > before:
        raise ValueError(
            "true_boundary_coverage_after_exclusions must not exceed true_boundary_coverage_before_exclusions"
        )
    if before > generated:
        raise ValueError(
            "true_boundary_coverage_before_exclusions must not exceed candidate_count_generated"
        )
    if after > retained:
        raise ValueError(
            "true_boundary_coverage_after_exclusions must not exceed candidate_count_retained"
        )
    if generated == 0 and before != 0:
        raise ValueError(
            "true_boundary_coverage_before_exclusions must be 0 when candidate_count_generated is 0"
        )
    if retained == 0 and after != 0:
        raise ValueError(
            "true_boundary_coverage_after_exclusions must be 0 when candidate_count_retained is 0"
        )


def validate_candidate_row(row: Mapping[str, object], *, row_number: int) -> None:
    legacy_columns = [field_name for field_name in LEGACY_EXTERNAL_COLUMNS if field_name in row]
    if legacy_columns:
        raise ValueError(
            f"row {row_number}: legacy columns are not allowed: {', '.join(legacy_columns)}"
        )
    missing = [header for header in STATIC_CANDIDATE_HEADERS if header not in row]
    if missing:
        raise ValueError(f"row {row_number}: missing required columns: {', '.join(missing)}")

    day_id = _require_non_blank_text(row, "day_id", row_number=row_number)
    candidate_id = _require_non_blank_text(row, "candidate_id", row_number=row_number)
    candidate_rule_name = _require_non_blank_text(
        row,
        "candidate_rule_name",
        row_number=row_number,
    )
    candidate_rule_version = _require_non_blank_text(
        row,
        "candidate_rule_version",
        row_number=row_number,
    )
    candidate_rule_params = _parse_json_object(
        row.get("candidate_rule_params_json"),
        field_name="candidate_rule_params_json",
        row_number=row_number,
    )
    _require_non_blank_text(row, "descriptor_schema_version", row_number=row_number)
    center_left_photo_id = _require_non_blank_text(
        row,
        "center_left_photo_id",
        row_number=row_number,
    )
    center_right_photo_id = _require_non_blank_text(
        row,
        "center_right_photo_id",
        row_number=row_number,
    )

    window_radius = _parse_positive_int(
        row.get("window_radius"),
        field_name=f"row {row_number} window_radius",
    )
    window_size = window_radius_to_window_size(window_radius)
    expected_headers = candidate_row_headers(window_radius=window_radius, include_thumbnail=True)
    missing_dynamic_headers = [header for header in expected_headers if header not in row]
    if missing_dynamic_headers:
        raise ValueError(
            f"row {row_number}: missing required columns: {', '.join(missing_dynamic_headers)}"
        )
    if candidate_rule_params.get("window_radius") != window_radius:
        raise ValueError(
            f"row {row_number}: candidate_rule_params_json window_radius must equal row window_radius"
        )

    frame_timestamp_fields = [
        f"frame_{index:02d}_timestamp"
        for index in range(1, window_size + 1)
    ]
    frame_photo_id_fields = [
        f"frame_{index:02d}_photo_id"
        for index in range(1, window_size + 1)
    ]
    frame_relpath_fields = [
        f"frame_{index:02d}_relpath"
        for index in range(1, window_size + 1)
    ]
    frame_thumb_fields = [
        f"frame_{index:02d}_thumb_path"
        for index in range(1, window_size + 1)
    ]
    frame_preview_fields = [
        f"frame_{index:02d}_preview_path"
        for index in range(1, window_size + 1)
    ]

    frame_photo_ids = [
        _require_non_blank_text(row, field_name, row_number=row_number)
        for field_name in frame_photo_id_fields
    ]
    frame_relpaths = [
        _require_non_blank_text(row, field_name, row_number=row_number)
        for field_name in frame_relpath_fields
    ]
    for field_name in frame_thumb_fields + frame_preview_fields:
        _require_string_value(row, field_name, row_number=row_number)

    window_photo_ids = _parse_json_array_of_strings(
        row.get("window_photo_ids"),
        field_name="window_photo_ids",
        row_number=row_number,
        expected_length=window_size,
    )
    window_relative_paths = _parse_json_array_of_strings(
        row.get("window_relative_paths"),
        field_name="window_relative_paths",
        row_number=row_number,
        expected_length=window_size,
    )
    if window_photo_ids != frame_photo_ids:
        raise ValueError(
            f"row {row_number}: window_photo_ids must match ordered frame photo_id columns"
        )
    if window_relative_paths != frame_relpaths:
        raise ValueError(
            f"row {row_number}: window_relative_paths must match ordered frame relpath columns"
        )

    if center_left_photo_id != frame_photo_ids[window_radius - 1]:
        raise ValueError(
            f"row {row_number}: center_left_photo_id must equal the left center frame photo_id"
        )
    if center_right_photo_id != frame_photo_ids[window_radius]:
        raise ValueError(
            f"row {row_number}: center_right_photo_id must equal the right center frame photo_id"
        )

    timestamps = [
        normalize_timestamp(_require_non_blank_text(row, field_name, row_number=row_number))
        for field_name in frame_timestamp_fields
    ]
    for index in range(1, len(timestamps)):
        if timestamps[index] < timestamps[index - 1]:
            raise ValueError(
                f"row {row_number}: frame timestamps must be ordered and non-decreasing"
            )

    expected_candidate_id = canonical_candidate_id(
        day_id=day_id,
        center_left_photo_id=center_left_photo_id,
        center_right_photo_id=center_right_photo_id,
        candidate_rule_version=candidate_rule_version,
    )
    if candidate_id != expected_candidate_id:
        raise ValueError(f"row {row_number}: candidate_id does not match canonical_candidate_id")
    if not candidate_rule_name:
        raise ValueError(f"row {row_number}: candidate_rule_name must not be blank")

    split_name_value = _require_string_value(row, "split_name", row_number=row_number).strip()
    if split_name_value and split_name_value not in VALID_SPLIT_NAMES:
        raise ValueError(
            f"row {row_number}: split_name must be one of train, validation, test when present"
        )

    left_segment_type = _require_non_blank_text(row, "left_segment_type", row_number=row_number)
    right_segment_type = _require_non_blank_text(row, "right_segment_type", row_number=row_number)
    segment_type = _require_non_blank_text(row, "segment_type", row_number=row_number)
    if left_segment_type not in VALID_SEGMENT_TYPES:
        raise ValueError(
            f"row {row_number}: left_segment_type must be one of {', '.join(sorted(VALID_SEGMENT_TYPES))}"
        )
    if right_segment_type not in VALID_SEGMENT_TYPES:
        raise ValueError(
            f"row {row_number}: right_segment_type must be one of {', '.join(sorted(VALID_SEGMENT_TYPES))}"
        )
    if segment_type != right_segment_type:
        raise ValueError(f"row {row_number}: segment_type must equal right_segment_type")

    left_segment_id = _require_non_blank_text(row, "left_segment_id", row_number=row_number)
    right_segment_id = _require_non_blank_text(row, "right_segment_id", row_number=row_number)
    boundary = _parse_bool(row["boundary"], field_name=f"row {row_number} boundary")
    expected_boundary = left_segment_id != right_segment_id
    if boundary != expected_boundary:
        raise ValueError(
            f"row {row_number}: boundary must match left/right segment-id equality"
        )


def validate_split_manifest(
    split_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    manifest_key: Optional[str] = None,
    required_classes: Optional[Sequence[str]] = None,
) -> None:
    if not split_rows:
        raise ValueError("split manifest must contain at least one row")
    if manifest_key is None:
        manifest_key = _detect_split_manifest_key(
            [field_name for row in split_rows for field_name in row.keys()]
        )
    if manifest_key == "candidate_id":
        seen_candidate_ids: set[str] = set()
        duplicate_candidate_ids: set[str] = set()
        for row_index, row in enumerate(candidate_rows, start=1):
            candidate_id = _require_non_blank_text(row, "candidate_id", row_number=row_index)
            if candidate_id in seen_candidate_ids:
                duplicate_candidate_ids.add(candidate_id)
            else:
                seen_candidate_ids.add(candidate_id)
        if duplicate_candidate_ids:
            raise ValueError(
                "candidate rows contain duplicate candidate_id values: "
                + ", ".join(sorted(duplicate_candidate_ids))
            )
    manifest_by_id: dict[str, str] = {}
    for row_index, row in enumerate(split_rows, start=1):
        manifest_id = _require_non_blank_text(row, manifest_key, row_number=row_index)
        split_name = _require_non_blank_text(row, "split_name", row_number=row_index)
        if split_name not in VALID_SPLIT_NAMES:
            raise ValueError(
                "split manifest split_name must be one of: train, validation, test"
            )
        previous_split_name = manifest_by_id.get(manifest_id)
        if previous_split_name is not None and previous_split_name != split_name:
            raise ValueError(
                f"split manifest assigns {manifest_key} {manifest_id} to multiple splits: "
                f"{previous_split_name}, {split_name}"
            )
        manifest_by_id[manifest_id] = split_name

    missing_ids = sorted(
        {
            _require_non_blank_text(row, manifest_key, row_number=index)
            for index, row in enumerate(candidate_rows, start=1)
            if _require_non_blank_text(row, manifest_key, row_number=index) not in manifest_by_id
        }
    )
    if missing_ids:
        missing_label = "candidate day_ids" if manifest_key == "day_id" else "candidate_ids"
        raise ValueError(
            f"split manifest is missing assignments for {missing_label}: "
            + ", ".join(missing_ids)
        )

    if not required_classes:
        return

    invalid_classes = sorted(set(required_classes) - VALID_SEGMENT_TYPES)
    if invalid_classes:
        raise ValueError(
            "required split classes must be a subset of: "
            + ", ".join(sorted(VALID_SEGMENT_TYPES))
        )

    classes_by_split: dict[str, set[str]] = {}
    for row_index, row in enumerate(candidate_rows, start=1):
        manifest_id = _require_non_blank_text(row, manifest_key, row_number=row_index)
        split_name = manifest_by_id[manifest_id]
        segment_type = _require_non_blank_text(row, "segment_type", row_number=row_index)
        classes_by_split.setdefault(split_name, set()).add(segment_type)

    for split_name in HELDOUT_SPLIT_NAMES:
        missing = sorted(set(required_classes) - classes_by_split.get(split_name, set()))
        if missing:
            raise ValueError(
                f"split {split_name} is missing required classes: {', '.join(missing)}"
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ML boundary candidate datasets, attrition reports, and optional split manifests."
    )
    parser.add_argument(
        "candidate_csv",
        help="Path to ml_boundary_candidates.csv or a single day directory like /data/20260323",
    )
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds ML boundary artifacts. Default: DAY/.vocatio WORKSPACE_DIR or DAY/_workspace",
    )
    parser.add_argument(
        "--attrition-json",
        help=(
            "Path to ml_boundary_attrition.json. "
            f"Default: WORKSPACE/{DEFAULT_ATTRITION_FILENAME} when candidate_csv is DAY"
        ),
    )
    parser.add_argument(
        "--split-manifest-csv",
        help=(
            "Optional split manifest CSV with candidate_id and split_name columns, "
            "or day_id and split_name columns. "
            "Relative paths resolve from WORKSPACE when candidate_csv is DAY."
        ),
    )
    parser.add_argument(
        "--required-heldout-classes",
        nargs="+",
        help="Optional held-out classes that must appear in both validation and test splits",
    )
    parser.add_argument(
        "--report-json",
        help=(
            "Optional path to write a validation report JSON. "
            f"Default: WORKSPACE/{DEFAULT_REPORT_FILENAME} when candidate_csv is DAY"
        ),
    )
    return parser.parse_args(argv)


def _resolve_workspace_path(workspace_dir: Path, value: Optional[str], default_name: str) -> Path:
    if not value:
        return workspace_dir / default_name
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def _looks_like_day_dir(path: Path) -> bool:
    if path.is_dir():
        return True
    return not path.exists() and path.suffix == "" and path.name.isdigit() and len(path.name) == 8


def _resolve_cli_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    candidate_input_path = Path(args.candidate_csv).expanduser()
    if candidate_input_path.is_dir() and (candidate_input_path / DEFAULT_CORPUS_CANDIDATES_FILENAME).is_file():
        corpus_workspace = candidate_input_path.resolve()
        candidate_csv_path = corpus_workspace / DEFAULT_CORPUS_CANDIDATES_FILENAME
        attrition_json_path = _resolve_workspace_path(
            corpus_workspace,
            args.attrition_json,
            DEFAULT_ATTRITION_FILENAME,
        )
        split_manifest_path = (
            _resolve_workspace_path(corpus_workspace, args.split_manifest_csv, "ml_boundary_splits.csv")
            if args.split_manifest_csv
            else None
        )
        report_json_path = _resolve_workspace_path(
            corpus_workspace,
            args.report_json,
            DEFAULT_REPORT_FILENAME,
        )
        return (
            candidate_csv_path.resolve(),
            attrition_json_path.resolve(),
            split_manifest_path.resolve() if split_manifest_path is not None else None,
            report_json_path.resolve(),
        )

    if _looks_like_day_dir(candidate_input_path):
        day_dir = candidate_input_path.resolve()
        workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
        candidate_csv_path = _resolve_workspace_path(
            workspace_dir,
            None,
            DEFAULT_OUTPUT_FILENAME,
        )
        attrition_json_path = _resolve_workspace_path(
            workspace_dir,
            args.attrition_json,
            DEFAULT_ATTRITION_FILENAME,
        )
        split_manifest_path = (
            _resolve_workspace_path(workspace_dir, args.split_manifest_csv, "ml_boundary_splits.csv")
            if args.split_manifest_csv
            else None
        )
        report_json_path = _resolve_workspace_path(
            workspace_dir,
            args.report_json,
            DEFAULT_REPORT_FILENAME,
        )
        return (
            candidate_csv_path.resolve(),
            attrition_json_path.resolve(),
            split_manifest_path.resolve() if split_manifest_path is not None else None,
            report_json_path.resolve(),
        )

    if not args.attrition_json:
        raise SystemExit("--attrition-json is required when candidate_csv is a CSV path")

    candidate_csv_path = candidate_input_path.resolve()
    attrition_json_path = Path(args.attrition_json).expanduser().resolve()
    split_manifest_path = (
        Path(args.split_manifest_csv).expanduser().resolve()
        if args.split_manifest_csv
        else None
    )
    report_json_path = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else None
    )
    return candidate_csv_path, attrition_json_path, split_manifest_path, report_json_path


def _validate_row_count_against_attrition(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    attrition_report: Mapping[str, object],
) -> None:
    retained = _parse_non_negative_int(
        attrition_report["candidate_count_retained"],
        field_name="candidate_count_retained",
    )
    if len(candidate_rows) != retained:
        raise ValueError(
            "candidate CSV row count must match attrition candidate_count_retained"
        )


def build_validation_report(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    attrition_report: Mapping[str, object],
) -> dict[str, object]:
    class_balance: dict[str, int] = {}
    for row_index, row in enumerate(candidate_rows, start=2):
        segment_type = _require_non_blank_text(row, "segment_type", row_number=row_index)
        class_balance[segment_type] = class_balance.get(segment_type, 0) + 1

    hard_negative_field_available = all("is_hard_negative" in row for row in candidate_rows)
    if hard_negative_field_available:
        hard_negative_true = 0
        hard_negative_false = 0
        for row_index, row in enumerate(candidate_rows, start=2):
            if _parse_bool(row.get("is_hard_negative"), field_name=f"row {row_index} is_hard_negative"):
                hard_negative_true += 1
            else:
                hard_negative_false += 1
        hard_negative_summary: dict[str, object] = {
            "available": True,
            "hard_negative_true_count": hard_negative_true,
            "hard_negative_false_count": hard_negative_false,
        }
    else:
        hard_negative_summary = {
            "available": False,
            "reason": "candidate CSV does not include is_hard_negative",
        }

    return {
        "candidate_row_count": len(candidate_rows),
        "class_balance_by_segment_type": class_balance,
        "attrition_exclusion_counts": {
            "candidate_count_excluded_missing_window": _parse_non_negative_int(
                attrition_report["candidate_count_excluded_missing_window"],
                field_name="candidate_count_excluded_missing_window",
            ),
            "candidate_count_excluded_missing_artifacts": _parse_non_negative_int(
                attrition_report["candidate_count_excluded_missing_artifacts"],
                field_name="candidate_count_excluded_missing_artifacts",
            ),
        },
        "hard_negative_coverage": hard_negative_summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    candidate_csv_path, attrition_json_path, split_manifest_path, report_json_path = _resolve_cli_paths(
        args
    )

    try:
        candidate_rows = _read_csv_rows(candidate_csv_path, required_headers=STATIC_CANDIDATE_HEADERS)
        attrition_report = _read_json_object(attrition_json_path)
        validate_attrition_report(attrition_report)
        _validate_row_count_against_attrition(
            candidate_rows=candidate_rows,
            attrition_report=attrition_report,
        )

        with _progress() as progress:
            validate_task = progress.add_task("Validate candidates".ljust(25), total=len(candidate_rows))
            for row_number, row in enumerate(candidate_rows, start=2):
                validate_candidate_row(row, row_number=row_number)
                progress.advance(validate_task)

        if split_manifest_path is not None:
            split_rows, manifest_key = _read_split_manifest_rows(split_manifest_path)
            validate_split_manifest(
                split_rows,
                candidate_rows,
                manifest_key=manifest_key,
                required_classes=args.required_heldout_classes,
            )

        validation_report = build_validation_report(
            candidate_rows=candidate_rows,
            attrition_report=attrition_report,
        )
        if report_json_path is not None:
            report_json_path.write_text(
                json.dumps(validation_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        console.print(
            "Validated ML boundary dataset: "
            f"{candidate_csv_path} ({len(candidate_rows)} row(s))"
        )
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
