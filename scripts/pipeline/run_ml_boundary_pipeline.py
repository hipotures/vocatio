#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from build_ml_boundary_candidate_dataset import (
    DEFAULT_WINDOW_RADIUS as DEFAULT_DATASET_WINDOW_RADIUS,
)
from build_ml_boundary_candidate_dataset import candidate_row_headers
from lib.ml_boundary_metrics import predictor_metric_spec
from lib.ml_boundary_truth import VALID_SEGMENT_TYPES
from lib.pipeline_io import atomic_write_csv, atomic_write_json
from lib.window_radius_contract import positive_window_radius_arg
from lib.workspace_dir import load_vocatio_config, resolve_workspace_dir


console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]

CORPUS_CANDIDATES_FILENAME = "ml_boundary_candidates.corpus.csv"
DAY_METADATA_FILENAME = "ml_boundary_day_metadata.csv"
SPLIT_MANIFEST_FILENAME = "ml_boundary_splits.csv"
CORPUS_ATTRITION_FILENAME = "ml_boundary_attrition.json"
DATASET_REPORT_FILENAME = "ml_boundary_dataset_report.json"
VALIDATION_REPORT_FILENAME = "ml_boundary_validation_report.json"
PIPELINE_SUMMARY_FILENAME = "ml_boundary_pipeline_summary.json"
EVALUATION_METRICS_FILENAME = "metrics.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"
SPLIT_MANIFEST_HEADERS = ["candidate_id", "split_name"]
DEFAULT_SPLIT_STRATEGY = "global_stratified"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SPLIT_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
HELDOUT_SPLIT_NAMES = ("validation", "test")
LEGACY_EXTERNAL_COLUMNS = ("window_size", "overlap")
DATASET_CONTRACT_KEYS = (
    "candidate_rule_name",
    "candidate_rule_version",
    "candidate_rule_params_json",
    "descriptor_schema_version",
    "window_radius",
)
FRAME_SCHEMA_COLUMN_RE = re.compile(
    r"^frame_\d{2}_(photo_id|relpath|timestamp|thumb_path|preview_path)$"
)


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    seed: int


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ML boundary verifier pipeline from reviewed GUI state to train/evaluate artifacts.",
    )
    parser.add_argument(
        "day_dirs",
        nargs="+",
        help="One or more day directories like /data/20260323",
    )
    parser.add_argument(
        "--corpus-workspace",
        help=(
            "Directory for merged corpus artifacts. "
            "Default: first DAY workspace/ml_boundary_corpus"
        ),
    )
    parser.add_argument(
        "--mode",
        default="tabular_only",
        choices=("tabular_only", "tabular_plus_thumbnail"),
        help="Training mode passed to train_ml_boundary_verifier.py",
    )
    parser.add_argument(
        "--model-run-id",
        default="run-001",
        help="Run directory name under ml_boundary_models/ and ml_boundary_eval/. Default: run-001",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("global_random", "global_stratified"),
        default=None,
        help="Corpus split strategy after merging candidate rows. Default: global_stratified.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        help="Fraction of merged candidate rows assigned to train. Default: 0.70",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Fraction of merged candidate rows assigned to validation. Default: 0.15",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        help="Fraction of merged candidate rows assigned to test. Default: 0.15",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Deterministic seed for corpus split shuffling. Default: 42",
    )
    parser.add_argument(
        "--required-heldout-classes",
        nargs="+",
        help=(
            "Required classes for validation/test split coverage. "
            "Default: all segment types present in merged corpus."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after dataset, split, and validation artifacts (skip train/evaluate).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rebuild candidate rows with --overwrite and replace train/eval artifacts.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="AutoGluon preset forwarded to train_ml_boundary_verifier.py.",
    )
    parser.add_argument(
        "--train-minutes",
        type=float,
        default=None,
        help="Optional per-predictor training time limit in minutes forwarded to the training step.",
    )
    return parser.parse_args(argv)


def _script_path(script_name: str) -> Path:
    return Path(__file__).resolve().parent / script_name


def _script_ref(script_name: str) -> str:
    return str(Path("scripts/pipeline") / script_name)


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, cwd=REPO_ROOT)


def _print_compact_command(label: str, command: Sequence[str]) -> None:
    console.print(f"Run {label}: {' '.join(str(value) for value in command)}")


def _existing_day_candidate_outputs(workspace_dir: Path) -> list[Path]:
    return [
        path
        for path in (
            workspace_dir / "ml_boundary_candidates.csv",
            workspace_dir / "ml_boundary_attrition.json",
            workspace_dir / DATASET_REPORT_FILENAME,
        )
        if path.exists()
    ]


def _resolve_days(day_values: Sequence[str]) -> list[Path]:
    day_dirs: list[Path] = []
    for value in day_values:
        day_dir = Path(value).expanduser().resolve()
        if not day_dir.is_dir():
            raise FileNotFoundError(f"Day directory does not exist: {day_dir}")
        day_dirs.append(day_dir)
    return day_dirs


def _prepare_single_day(day_dir: Path, *, restart: bool) -> Path:
    workspace_dir = resolve_workspace_dir(day_dir, None)
    day_window_radius = _resolve_day_window_radius(day_dir)
    export_command = [
        sys.executable,
        _script_ref("export_ml_boundary_reviewed_truth.py"),
        str(day_dir),
    ]
    _print_compact_command("Export reviewed truth", export_command)
    _run_command(export_command)

    existing_outputs = _existing_day_candidate_outputs(workspace_dir)
    if existing_outputs and not restart:
        raise ValueError(
            "Existing ML boundary day outputs detected. "
            f"Use --restart to rebuild them: {', '.join(str(path.name) for path in existing_outputs)}"
        )

    build_command = [
        sys.executable,
        _script_ref("build_ml_boundary_candidate_dataset.py"),
        str(day_dir),
        "--window-radius",
        str(day_window_radius),
    ]
    if restart:
        build_command.append("--overwrite")
    _print_compact_command("Build candidate dataset", build_command)
    _run_command(build_command)

    validate_command = [
        sys.executable,
        _script_ref("validate_ml_boundary_dataset.py"),
        str(day_dir),
    ]
    _print_compact_command("Validate day dataset", validate_command)
    _run_command(validate_command)

    return workspace_dir


def _resolve_day_window_radius(day_dir: Path) -> int:
    config = load_vocatio_config(day_dir)
    configured_value = str(config.get("VLM_WINDOW_RADIUS", "") or "").strip()
    if configured_value == "":
        return DEFAULT_DATASET_WINDOW_RADIUS
    return positive_window_radius_arg(configured_value)


def _resolve_optional_float(
    *,
    cli_value: float | None,
    config: dict[str, str],
    config_key: str,
    default_value: float,
) -> float:
    if cli_value is not None:
        return cli_value
    configured_value = str(config.get(config_key, "") or "").strip()
    if configured_value == "":
        return default_value
    try:
        return float(configured_value)
    except ValueError as exc:
        raise ValueError(f"{config_key} must be a float, got {configured_value!r}") from exc


def _resolve_optional_int(
    *,
    cli_value: int | None,
    config: dict[str, str],
    config_key: str,
    default_value: int,
) -> int:
    if cli_value is not None:
        return cli_value
    configured_value = str(config.get(config_key, "") or "").strip()
    if configured_value == "":
        return default_value
    try:
        return int(configured_value)
    except ValueError as exc:
        raise ValueError(f"{config_key} must be an integer, got {configured_value!r}") from exc


def resolve_split_config(args: argparse.Namespace, day_dirs: Sequence[Path]) -> SplitConfig:
    config = load_vocatio_config(day_dirs[0]) if day_dirs else {}
    strategy = args.split_strategy
    if strategy is None:
        configured_strategy = str(config.get("ML_SPLIT_STRATEGY", "") or "").strip()
        strategy = configured_strategy or DEFAULT_SPLIT_STRATEGY
    if strategy not in {"global_random", "global_stratified"}:
        raise ValueError("split strategy must be one of: global_random, global_stratified")

    train_fraction = _resolve_optional_float(
        cli_value=args.train_fraction,
        config=config,
        config_key="ML_SPLIT_TRAIN_FRACTION",
        default_value=DEFAULT_TRAIN_FRACTION,
    )
    validation_fraction = _resolve_optional_float(
        cli_value=args.validation_fraction,
        config=config,
        config_key="ML_SPLIT_VALIDATION_FRACTION",
        default_value=DEFAULT_VALIDATION_FRACTION,
    )
    test_fraction = _resolve_optional_float(
        cli_value=args.test_fraction,
        config=config,
        config_key="ML_SPLIT_TEST_FRACTION",
        default_value=DEFAULT_TEST_FRACTION,
    )
    for name, value in (
        ("train_fraction", train_fraction),
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    fraction_total = train_fraction + validation_fraction + test_fraction
    if not math.isclose(fraction_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/validation/test fractions must sum to 1.0")

    seed = _resolve_optional_int(
        cli_value=args.split_seed,
        config=config,
        config_key="ML_SPLIT_SEED",
        default_value=DEFAULT_SPLIT_SEED,
    )
    return SplitConfig(
        strategy=strategy,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )


def _parse_positive_int(value: object, *, field_name: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _require_non_blank_string(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _load_dataset_report(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    missing = [key for key in DATASET_CONTRACT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path.name} missing required keys: {', '.join(missing)}")
    payload["candidate_rule_name"] = _require_non_blank_string(
        payload.get("candidate_rule_name"),
        field_name=f"{path.name} candidate_rule_name",
    )
    payload["candidate_rule_version"] = _require_non_blank_string(
        payload.get("candidate_rule_version"),
        field_name=f"{path.name} candidate_rule_version",
    )
    payload["candidate_rule_params_json"] = _require_non_blank_string(
        payload.get("candidate_rule_params_json"),
        field_name=f"{path.name} candidate_rule_params_json",
    )
    payload["descriptor_schema_version"] = _require_non_blank_string(
        payload.get("descriptor_schema_version"),
        field_name=f"{path.name} descriptor_schema_version",
    )
    payload["window_radius"] = _parse_positive_int(
        payload.get("window_radius"),
        field_name=f"{path.name} window_radius",
    )
    return payload


def _load_dataset_contract(day_workspaces: Sequence[Path]) -> dict[str, object]:
    shared_contract: dict[str, object] | None = None
    shared_path: Path | None = None
    for workspace_dir in day_workspaces:
        report_path = workspace_dir / DATASET_REPORT_FILENAME
        report = _load_dataset_report(report_path)
        contract = {key: report[key] for key in DATASET_CONTRACT_KEYS}
        if shared_contract is None:
            shared_contract = contract
            shared_path = report_path
            continue
        mismatches = [
            key
            for key in DATASET_CONTRACT_KEYS
            if contract[key] != shared_contract[key]
        ]
        if mismatches:
            mismatch_details = ", ".join(
                f"{key}={contract[key]!r} (expected {shared_contract[key]!r})"
                for key in mismatches
            )
            raise ValueError(
                "ML boundary dataset contract mismatch between "
                f"{shared_path} and {report_path}: {mismatch_details}"
            )
    if shared_contract is None:
        raise ValueError("No ML boundary dataset reports were found")
    return shared_contract


def _extract_dataset_contract_from_candidate_rows(
    candidate_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    if not candidate_rows:
        raise ValueError("merged candidate rows must contain at least one row")

    contract: dict[str, object] = {}
    for key in DATASET_CONTRACT_KEYS:
        values = {str(row.get(key, "")).strip() for row in candidate_rows}
        if "" in values:
            raise ValueError(f"merged candidate rows must not contain blank {key} values")
        if len(values) != 1:
            raise ValueError(
                f"merged candidate rows must contain exactly one {key}, got {sorted(values)}"
            )
        value = next(iter(values))
        if key == "window_radius":
            contract[key] = _parse_positive_int(
                value,
                field_name="merged candidate rows window_radius",
            )
        else:
            contract[key] = value
    return contract


def _validate_dataset_contract_against_candidate_rows(
    report_contract: dict[str, object],
    candidate_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    row_contract = _extract_dataset_contract_from_candidate_rows(candidate_rows)
    mismatches = [
        key
        for key in DATASET_CONTRACT_KEYS
        if row_contract[key] != report_contract[key]
    ]
    if mismatches:
        mismatch_details = ", ".join(
            f"{key}={report_contract[key]!r} (expected {row_contract[key]!r})"
            for key in mismatches
        )
        raise ValueError(
            "ML boundary dataset contract mismatch between reports and candidate rows: "
            f"{mismatch_details}"
        )
    return row_contract


def _unexpected_schema_columns(
    columns: Sequence[str],
    *,
    expected_headers: Sequence[str],
) -> list[str]:
    expected = set(expected_headers)
    return sorted(
        column
        for column in set(columns)
        if column not in expected and FRAME_SCHEMA_COLUMN_RE.match(column)
    )


def _read_candidate_rows(path: Path, *, expected_headers: Sequence[str]) -> list[dict[str, str]]:
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
        missing = [header for header in expected_headers if header not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        unexpected_columns = _unexpected_schema_columns(
            reader.fieldnames or (),
            expected_headers=expected_headers,
        )
        if unexpected_columns:
            raise ValueError(
                f"{path.name} unexpected columns are not allowed: {', '.join(unexpected_columns)}"
            )
        rows = [dict(row) for row in reader]
    return rows


def _merge_candidate_rows(
    candidate_csv_paths: Sequence[Path],
    *,
    expected_headers: Sequence[str],
) -> list[dict[str, str]]:
    merged_rows: list[dict[str, str]] = []
    with _progress() as progress:
        task_id = progress.add_task("Merge candidate rows".ljust(25), total=len(candidate_csv_paths))
        for path in candidate_csv_paths:
            merged_rows.extend(_read_candidate_rows(path, expected_headers=expected_headers))
            progress.advance(task_id)
    if not merged_rows:
        raise ValueError(
            "No ML boundary candidates were retained. Check each workspace ml_boundary_attrition.json "
            "for excluded_missing_artifacts and excluded_missing_window counts."
        )
    merged_rows.sort(
        key=lambda row: (
            str(row.get("day_id", "")),
            str(row.get("candidate_id", "")),
        )
    )
    return merged_rows


def _build_day_metadata_rows(merged_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    segment_types_by_day: dict[str, set[str]] = {}
    for row in merged_rows:
        day_id = str(row.get("day_id", "")).strip()
        right_segment_type = str(row.get("right_segment_type", "")).strip()
        if day_id == "" or right_segment_type == "":
            raise ValueError("merged candidate rows must include non-blank day_id and right_segment_type")
        if right_segment_type not in VALID_SEGMENT_TYPES:
            choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
            raise ValueError(
                "merged candidate rows include unsupported right_segment_type "
                f"{right_segment_type!r}; expected one of {choices}"
            )
        segment_types_by_day.setdefault(day_id, set()).add(right_segment_type)

    metadata_rows: list[dict[str, str]] = []
    for day_id in sorted(segment_types_by_day):
        year = day_id[:4] if len(day_id) >= 4 and day_id[:4].isdigit() else ""
        metadata_rows.append(
            {
                "day_id": day_id,
                "segment_types": json.dumps(
                    sorted(segment_types_by_day[day_id]),
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                "year": year,
                "camera": "",
                "domain_shift_hint": "",
            }
        )
    return metadata_rows


def _load_attrition_report(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Attrition report must be a JSON object: {path}")

    required_keys = (
        "candidate_count_generated",
        "candidate_count_excluded_missing_window",
        "candidate_count_excluded_missing_artifacts",
        "candidate_count_retained",
        "true_boundary_coverage_before_exclusions",
        "true_boundary_coverage_after_exclusions",
    )
    report: dict[str, int] = {}
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Attrition report missing required key {key}: {path}")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Attrition report key {key} must be an integer: {path}")
        report[key] = value
    return report


def _build_corpus_attrition_report(day_workspaces: Sequence[Path]) -> dict[str, int]:
    totals = {
        "candidate_count_generated": 0,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 0,
        "true_boundary_coverage_before_exclusions": 0,
        "true_boundary_coverage_after_exclusions": 0,
    }
    for workspace_dir in day_workspaces:
        report = _load_attrition_report(workspace_dir / CORPUS_ATTRITION_FILENAME)
        for key in totals:
            totals[key] += report[key]
    return totals


def _validate_corpus_size(merged_rows: Sequence[dict[str, str]]) -> None:
    if len(merged_rows) < 3:
        raise ValueError(
            "ML boundary corpus split requires at least three candidate rows to produce train, validation, and test splits"
        )


def _split_counts_for_total(
    total_rows: int,
    split_config: SplitConfig,
    *,
    minimum_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    split_names = SPLIT_NAMES
    total_fraction = (
        split_config.train_fraction
        + split_config.validation_fraction
        + split_config.test_fraction
    )
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.0")
    if minimum_counts is None:
        minimum_counts = {split_name: 1 for split_name in split_names}
    if set(minimum_counts) != set(split_names):
        raise ValueError("minimum_counts must define train, validation, and test")
    if any(value < 0 for value in minimum_counts.values()):
        raise ValueError("minimum_counts values must be non-negative")
    if sum(minimum_counts.values()) > total_rows:
        raise ValueError("minimum split counts exceed available candidate rows")

    targets = {
        "train": total_rows * split_config.train_fraction,
        "validation": total_rows * split_config.validation_fraction,
        "test": total_rows * split_config.test_fraction,
    }
    counts = {
        split_name: max(minimum_counts[split_name], int(round(targets[split_name])))
        for split_name in split_names
    }

    while sum(counts.values()) > total_rows:
        candidates = [
            split_name
            for split_name in split_names
            if counts[split_name] > minimum_counts[split_name]
        ]
        if not candidates:
            break
        split_name = max(
            candidates,
            key=lambda name: (
                counts[name] - targets[name],
                counts[name],
                -split_names.index(name),
            ),
        )
        counts[split_name] -= 1

    while sum(counts.values()) < total_rows:
        split_name = max(
            split_names,
            key=lambda name: (
                targets[name] - counts[name],
                targets[name],
                -split_names.index(name),
            ),
        )
        counts[split_name] += 1

    return counts


def _validate_candidate_row(row: dict[str, str]) -> tuple[str, str, str]:
    candidate_id = str(row.get("candidate_id", "")).strip()
    if candidate_id == "":
        raise ValueError("merged candidate rows must include non-blank candidate_id values")
    boundary = str(row.get("boundary", "")).strip().lower()
    if boundary == "":
        raise ValueError("merged candidate rows must include non-blank boundary values")
    right_segment_type = str(row.get("right_segment_type", "")).strip()
    if right_segment_type == "":
        raise ValueError("merged candidate rows must include non-blank right_segment_type values")
    return candidate_id, boundary, right_segment_type


def _validate_unique_candidate_ids(candidate_rows: Sequence[dict[str, str]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in candidate_rows:
        candidate_id, _, _ = _validate_candidate_row(row)
        if candidate_id in seen:
            duplicates.add(candidate_id)
        else:
            seen.add(candidate_id)
    if duplicates:
        duplicate_values = ", ".join(sorted(duplicates))
        raise ValueError(
            "merged candidate rows contain duplicate candidate_id values: "
            f"{duplicate_values}"
        )


def _build_split_rows_from_assignments(assignments: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"candidate_id": candidate_id, "split_name": assignments[candidate_id]}
        for candidate_id in sorted(assignments)
    ]


def _build_global_random_split_rows(
    candidate_rows: Sequence[dict[str, str]],
    split_config: SplitConfig,
) -> list[dict[str, str]]:
    _validate_corpus_size(candidate_rows)
    _validate_unique_candidate_ids(candidate_rows)
    candidate_ids = [_validate_candidate_row(row)[0] for row in candidate_rows]
    random_generator = random.Random(split_config.seed)
    random_generator.shuffle(candidate_ids)
    counts = _split_counts_for_total(len(candidate_ids), split_config)

    assignments: dict[str, str] = {}
    start_index = 0
    for split_name in SPLIT_NAMES:
        end_index = start_index + counts[split_name]
        for candidate_id in candidate_ids[start_index:end_index]:
            assignments[candidate_id] = split_name
        start_index = end_index
    return _build_split_rows_from_assignments(assignments)


def _stratified_minimum_split_counts(required_heldout_classes: Sequence[str]) -> dict[str, int]:
    heldout_minimum = len(dict.fromkeys(required_heldout_classes))
    return {
        "train": 1,
        "validation": max(1, heldout_minimum),
        "test": max(1, heldout_minimum),
    }


def _shuffle_strata(
    candidate_rows: Sequence[dict[str, str]],
    *,
    split_config: SplitConfig,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, str]]:
    random_generator = random.Random(split_config.seed)
    strata: dict[tuple[str, str], list[str]] = {}
    right_segment_type_by_candidate_id: dict[str, str] = {}
    for row in candidate_rows:
        candidate_id, boundary, right_segment_type = _validate_candidate_row(row)
        strata.setdefault((boundary, right_segment_type), []).append(candidate_id)
        right_segment_type_by_candidate_id[candidate_id] = right_segment_type
    for stratum_key in sorted(strata):
        random_generator.shuffle(strata[stratum_key])
    return strata, right_segment_type_by_candidate_id


def _reserve_required_heldout_assignments(
    strata: dict[tuple[str, str], list[str]],
    *,
    required_heldout_classes: Sequence[str],
) -> dict[str, str]:
    ordered_classes = list(dict.fromkeys(required_heldout_classes))
    if not ordered_classes:
        return {}

    requirements = [
        (split_name, segment_type)
        for split_name in HELDOUT_SPLIT_NAMES
        for segment_type in ordered_classes
    ]
    original_sizes = {
        stratum_key: len(candidate_ids)
        for stratum_key, candidate_ids in strata.items()
    }
    remaining_counts = dict(original_sizes)
    chosen_strata: dict[tuple[str, str], tuple[str, str]] = {}
    best_plan: dict[tuple[str, str], tuple[str, str]] | None = None
    best_score: tuple[int, int, int, int] | None = None

    def _search(requirement_index: int) -> None:
        nonlocal best_plan, best_score
        if requirement_index >= len(requirements):
            boundaries_by_split = {split_name: set() for split_name in HELDOUT_SPLIT_NAMES}
            used_strata = set()
            selected_size_total = 0
            for split_name, segment_type in requirements:
                stratum_key = chosen_strata[(split_name, segment_type)]
                boundaries_by_split[split_name].add(stratum_key[0])
                used_strata.add(stratum_key)
                selected_size_total += original_sizes[stratum_key]
            coverage_counts = tuple(
                len(boundaries_by_split[split_name])
                for split_name in HELDOUT_SPLIT_NAMES
            )
            score = (
                min(coverage_counts),
                sum(coverage_counts),
                len(used_strata),
                selected_size_total,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_plan = dict(chosen_strata)
            return

        split_name, segment_type = requirements[requirement_index]
        reserved_boundaries = {
            stratum_key[0]
            for (reserved_split_name, _), stratum_key in chosen_strata.items()
            if reserved_split_name == split_name
        }
        eligible_strata = [
            stratum_key
            for stratum_key in sorted(strata)
            if stratum_key[1] == segment_type and remaining_counts[stratum_key] > 0
        ]
        if not eligible_strata:
            return
        eligible_strata.sort(
            key=lambda stratum_key: (
                stratum_key[0] not in reserved_boundaries,
                remaining_counts[stratum_key],
                stratum_key,
            ),
            reverse=True,
        )
        for stratum_key in eligible_strata:
            remaining_counts[stratum_key] -= 1
            chosen_strata[(split_name, segment_type)] = stratum_key
            _search(requirement_index + 1)
            del chosen_strata[(split_name, segment_type)]
            remaining_counts[stratum_key] += 1

    _search(0)
    if best_plan is None:
        missing_classes = ", ".join(repr(value) for value in ordered_classes)
        raise ValueError(
            f"Unable to reserve held-out coverage for required segment types: {missing_classes}"
        )

    assignments: dict[str, str] = {}
    for split_name, segment_type in requirements:
        stratum_key = best_plan[(split_name, segment_type)]
        candidate_id = strata[stratum_key].pop(0)
        assignments[candidate_id] = split_name
    return assignments


def _can_preserve_full_heldout_strata(
    strata: dict[tuple[str, str], list[str]],
    *,
    target_counts: dict[str, int],
) -> bool:
    populated_strata = [
        stratum_key
        for stratum_key in sorted(strata)
        if strata[stratum_key]
    ]
    if not populated_strata:
        return True
    if any(target_counts[split_name] < len(populated_strata) for split_name in HELDOUT_SPLIT_NAMES):
        return False
    return all(len(strata[stratum_key]) >= len(HELDOUT_SPLIT_NAMES) for stratum_key in populated_strata)


def _reserve_full_heldout_strata_assignments(
    strata: dict[tuple[str, str], list[str]],
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split_name in HELDOUT_SPLIT_NAMES:
        for stratum_key in sorted(strata):
            if not strata[stratum_key]:
                raise ValueError("Unable to preserve full held-out strata coverage")
            candidate_id = strata[stratum_key].pop(0)
            assignments[candidate_id] = split_name
    return assignments


def _allocate_remaining_stratified_counts(
    strata: dict[tuple[str, str], list[str]],
    *,
    target_counts: dict[str, int],
) -> dict[tuple[str, str], dict[str, int]]:
    remaining_total = sum(len(candidate_ids) for candidate_ids in strata.values())
    if remaining_total != sum(target_counts.values()):
        raise ValueError("remaining stratified allocation rows do not match target counts")

    counts_by_stratum = {
        stratum_key: {split_name: 0 for split_name in SPLIT_NAMES}
        for stratum_key in strata
    }
    if remaining_total == 0:
        return counts_by_stratum

    stratum_sizes = {
        stratum_key: len(candidate_ids)
        for stratum_key, candidate_ids in strata.items()
    }
    target_by_stratum = {
        stratum_key: {
            split_name: stratum_sizes[stratum_key] * target_counts[split_name] / remaining_total
            for split_name in SPLIT_NAMES
        }
        for stratum_key in strata
    }
    stratum_remaining: dict[tuple[str, str], int] = {}
    split_remaining = dict(target_counts)
    for stratum_key in sorted(strata):
        for split_name in SPLIT_NAMES:
            count = int(math.floor(target_by_stratum[stratum_key][split_name]))
            counts_by_stratum[stratum_key][split_name] = count
            split_remaining[split_name] -= count
        stratum_remaining[stratum_key] = stratum_sizes[stratum_key] - sum(
            counts_by_stratum[stratum_key].values()
        )

    while sum(stratum_remaining.values()) > 0:
        best_choice: tuple[object, ...] | None = None
        best_stratum_key: tuple[str, str] | None = None
        best_split_name: str | None = None
        for stratum_key in sorted(strata):
            if stratum_remaining[stratum_key] <= 0:
                continue
            for split_name in SPLIT_NAMES:
                if split_remaining[split_name] <= 0:
                    continue
                remainder = (
                    target_by_stratum[stratum_key][split_name]
                    - counts_by_stratum[stratum_key][split_name]
                )
                choice = (
                    remainder,
                    split_remaining[split_name],
                    stratum_sizes[stratum_key],
                    -SPLIT_NAMES.index(split_name),
                    stratum_key,
                )
                if best_choice is None or choice > best_choice:
                    best_choice = choice
                    best_stratum_key = stratum_key
                    best_split_name = split_name
        if best_stratum_key is None or best_split_name is None:
            raise ValueError("Unable to complete stratified allocation")
        counts_by_stratum[best_stratum_key][best_split_name] += 1
        stratum_remaining[best_stratum_key] -= 1
        split_remaining[best_split_name] -= 1

    if any(value != 0 for value in split_remaining.values()):
        raise ValueError("Stratified allocation left unmatched split counts")
    return counts_by_stratum


def _has_required_heldout_coverage(
    assignments: dict[str, str],
    *,
    right_segment_type_by_candidate_id: dict[str, str],
    required_heldout_classes: Sequence[str],
) -> bool:
    if not required_heldout_classes:
        return True
    classes_by_split: dict[str, set[str]] = {}
    for candidate_id, split_name in assignments.items():
        classes_by_split.setdefault(split_name, set()).add(right_segment_type_by_candidate_id[candidate_id])
    for split_name in HELDOUT_SPLIT_NAMES:
        if not set(required_heldout_classes).issubset(classes_by_split.get(split_name, set())):
            return False
    return True


def _required_heldout_coverage_error(
    required_heldout_classes: Sequence[str],
    *,
    reason: str | None = None,
) -> ValueError:
    required_values = ", ".join(sorted(dict.fromkeys(required_heldout_classes)))
    message = (
        "global_stratified cannot satisfy required held-out classes under the requested "
        f"split configuration: {required_values}"
    )
    if reason:
        message = f"{message} ({reason})"
    return ValueError(message)


def _raise_required_heldout_coverage_error(
    required_heldout_classes: Sequence[str],
    *,
    reason: str | None = None,
    cause: Exception | None = None,
) -> None:
    error = _required_heldout_coverage_error(required_heldout_classes, reason=reason)
    if cause is not None:
        raise error from cause
    raise error


def _build_global_stratified_split_rows(
    candidate_rows: Sequence[dict[str, str]],
    split_config: SplitConfig,
    *,
    required_heldout_classes: Sequence[str] = (),
) -> tuple[list[dict[str, str]], str]:
    _validate_corpus_size(candidate_rows)
    _validate_unique_candidate_ids(candidate_rows)
    strata, right_segment_type_by_candidate_id = _shuffle_strata(
        candidate_rows,
        split_config=split_config,
    )
    try:
        target_counts = _split_counts_for_total(
            len(candidate_rows),
            split_config,
            minimum_counts=_stratified_minimum_split_counts(required_heldout_classes),
        )
    except ValueError as exc:
        if required_heldout_classes and str(exc) == "minimum split counts exceed available candidate rows":
            _raise_required_heldout_coverage_error(
                required_heldout_classes,
                reason=str(exc),
                cause=exc,
            )
        raise

    if _can_preserve_full_heldout_strata(strata, target_counts=target_counts):
        assignments = _reserve_full_heldout_strata_assignments(strata)
    else:
        try:
            assignments = _reserve_required_heldout_assignments(
                strata,
                required_heldout_classes=required_heldout_classes,
            )
        except ValueError as exc:
            if required_heldout_classes and str(exc).startswith(
                "Unable to reserve held-out coverage for required segment types:"
            ):
                _raise_required_heldout_coverage_error(
                    required_heldout_classes,
                    reason=str(exc),
                    cause=exc,
                )
            raise

    preassigned_counts = {
        split_name: sum(1 for value in assignments.values() if value == split_name)
        for split_name in SPLIT_NAMES
    }
    remaining_target_counts = {
        split_name: target_counts[split_name] - preassigned_counts[split_name]
        for split_name in SPLIT_NAMES
    }
    if any(value < 0 for value in remaining_target_counts.values()):
        if required_heldout_classes:
            _raise_required_heldout_coverage_error(
                required_heldout_classes,
                reason="Reserved held-out assignments exceed target split counts",
            )
        return _build_global_random_split_rows(candidate_rows, split_config), "global_random"
    counts_by_stratum = _allocate_remaining_stratified_counts(
        strata,
        target_counts=remaining_target_counts,
    )

    for stratum_key in sorted(strata):
        candidate_ids = list(strata[stratum_key])
        start_index = 0
        for split_name in SPLIT_NAMES:
            end_index = start_index + counts_by_stratum[stratum_key][split_name]
            for candidate_id in candidate_ids[start_index:end_index]:
                assignments[candidate_id] = split_name
            start_index = end_index

    if not _has_required_heldout_coverage(
        assignments,
        right_segment_type_by_candidate_id=right_segment_type_by_candidate_id,
        required_heldout_classes=required_heldout_classes,
    ):
        if required_heldout_classes:
            _raise_required_heldout_coverage_error(
                required_heldout_classes,
                reason="coverage check failed after stratified allocation",
            )
        return _build_global_random_split_rows(candidate_rows, split_config), "global_random"

    return _build_split_rows_from_assignments(assignments), "global_stratified"


def _build_corpus_split_rows(
    merged_rows: Sequence[dict[str, str]],
    split_config: SplitConfig,
    *,
    required_heldout_classes: Sequence[str] = (),
) -> tuple[list[dict[str, str]], str]:
    if split_config.strategy == "global_random":
        return _build_global_random_split_rows(merged_rows, split_config), "global_random"
    if split_config.strategy == "global_stratified":
        return _build_global_stratified_split_rows(
            merged_rows,
            split_config,
            required_heldout_classes=required_heldout_classes,
        )
    raise ValueError("split strategy must be one of: global_random, global_stratified")


def _required_classes(
    metadata_rows: Sequence[dict[str, str]],
    explicit_required: Sequence[str] | None,
) -> list[str]:
    if explicit_required:
        values = []
        for value in explicit_required:
            normalized = value.strip()
            if normalized == "":
                raise ValueError("required held-out classes must not include blank values")
            if normalized not in VALID_SEGMENT_TYPES:
                choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
                raise ValueError(f"required held-out classes must be one of: {choices}")
            values.append(normalized)
        return values

    discovered: set[str] = set()
    for row in metadata_rows:
        parsed = json.loads(row["segment_types"])
        if not isinstance(parsed, list):
            raise ValueError("segment_types must be a JSON array in day metadata rows")
        for value in parsed:
            if not isinstance(value, str):
                raise ValueError("segment_types values must be strings")
            discovered.add(value)
    return sorted(discovered)


def _load_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _training_split_count(training_metadata: dict[str, object] | None, split_name: str) -> int:
    if not isinstance(training_metadata, dict):
        return 0
    split_counts_by_name = training_metadata.get("split_counts_by_name")
    if not isinstance(split_counts_by_name, dict):
        return 0
    raw_value = split_counts_by_name.get(split_name, 0)
    try:
        return int(raw_value or 0)
    except (TypeError, ValueError):
        return 0


def _training_metadata_int(training_metadata: dict[str, object] | None, key: str) -> int | None:
    if not isinstance(training_metadata, dict):
        return None
    raw_value = training_metadata.get(key)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _heuristic_coverage_summary_line(training_metadata: dict[str, object] | None) -> str | None:
    coverage = _heuristic_coverage_summary_payload(training_metadata)
    if coverage is None:
        return None
    if coverage["source_available"] is False:
        return (
            "Heuristics (training corpus): unavailable "
            f"(photo_boundary_scores.csv not found; expected_pairs={coverage['total_pair_count']}, "
            f"expected_candidates={coverage['total_candidate_count']})"
        )
    if (
        coverage["total_pair_count"] is not None
        and coverage["missing_pair_count"] is not None
        and coverage["total_candidate_count"] is not None
        and coverage["missing_candidate_count"] is not None
    ):
        return (
            "Heuristics (training corpus): "
            f"pairs={coverage['covered_pair_count']}/{coverage['total_pair_count']}, "
            f"complete_candidates={coverage['complete_candidate_count']}/{coverage['total_candidate_count']}"
        )
    return (
        "Heuristics (training corpus): "
        f"missing_pairs={coverage['missing_pair_count']}, "
        f"missing_candidates={coverage['missing_candidate_count']}"
    )


def _heuristic_coverage_summary_payload(
    training_metadata: dict[str, object] | None,
) -> dict[str, int | bool | None] | None:
    if not isinstance(training_metadata, dict):
        return None
    total_pair_count = _training_metadata_int(training_metadata, "total_heuristic_pair_count")
    missing_pair_count = _training_metadata_int(training_metadata, "missing_heuristic_pair_count")
    total_candidate_count = _training_metadata_int(training_metadata, "total_heuristic_candidate_count")
    missing_candidate_count = _training_metadata_int(training_metadata, "missing_heuristic_candidate_count")
    source_available = training_metadata.get("heuristic_scores_source_available")
    if (
        total_pair_count is not None
        and missing_pair_count is not None
        and total_candidate_count is not None
        and missing_candidate_count is not None
    ):
        covered_pair_count = total_pair_count - missing_pair_count
        complete_candidate_count = total_candidate_count - missing_candidate_count
        return {
            "source_available": bool(source_available) if source_available is not None else None,
            "total_pair_count": total_pair_count,
            "covered_pair_count": covered_pair_count,
            "missing_pair_count": missing_pair_count,
            "total_candidate_count": total_candidate_count,
            "complete_candidate_count": complete_candidate_count,
            "missing_candidate_count": missing_candidate_count,
        }
    if missing_pair_count is not None and missing_candidate_count is not None:
        return {
            "source_available": bool(source_available) if source_available is not None else None,
            "total_pair_count": None,
            "covered_pair_count": None,
            "missing_pair_count": missing_pair_count,
            "total_candidate_count": None,
            "complete_candidate_count": None,
            "missing_candidate_count": missing_candidate_count,
        }
    return None


def _build_predictor_confusion_table(
    metrics_payload: dict[str, object],
    predictor_name: str,
) -> Table:
    confusion_key = f"{predictor_name}_confusion_matrix"
    confusion_payload = metrics_payload.get(confusion_key)
    if not isinstance(confusion_payload, dict):
        raise ValueError(f"evaluation metrics missing {confusion_key} object")
    row_count = int(metrics_payload.get("row_count", 0) or 0)
    predictor_spec = predictor_metric_spec(predictor_name)

    labels = sorted(VALID_SEGMENT_TYPES)
    table = Table(
        title=f"{predictor_spec.console_label} Confusion Matrix (test split, n={row_count})",
        expand=False,
    )
    table.add_column("truth\\pred", justify="left")
    for label in labels:
        table.add_column(label, justify="right")

    for truth_label in labels:
        row_payload = confusion_payload.get(truth_label, {})
        if not isinstance(row_payload, dict):
            row_payload = {}
        table.add_row(
            truth_label,
            *[
                str(int(row_payload.get(predicted_label, 0) or 0))
                for predicted_label in labels
            ],
        )
    return table


def _primary_eval_metric_value(metrics_payload: dict[str, object], predictor_name: str) -> float:
    metric_spec = predictor_metric_spec(predictor_name)
    return float(metrics_payload.get(metric_spec.evaluation_metric_key, 0.0) or 0.0)


def _render_eval_metrics_summary(
    metrics_payload: dict[str, object],
    training_metadata: dict[str, object] | None = None,
) -> Group:
    review_cost_metrics = metrics_payload.get("review_cost_metrics")
    if not isinstance(review_cost_metrics, dict):
        raise ValueError("evaluation metrics missing review_cost_metrics object")
    left_segment_type_metric_spec = predictor_metric_spec("left_segment_type")
    right_segment_type_metric_spec = predictor_metric_spec("right_segment_type")
    boundary_metric_spec = predictor_metric_spec("boundary")
    train_row_count = _training_split_count(training_metadata, "train")
    validation_row_count = _training_split_count(training_metadata, "validation")
    test_row_count = _training_split_count(training_metadata, "test")
    left_segment_type_primary_value = _primary_eval_metric_value(metrics_payload, "left_segment_type")
    left_segment_type_accuracy = float(metrics_payload.get("left_segment_type_accuracy", 0.0) or 0.0)
    left_segment_type_correct_count = int(
        metrics_payload.get("left_segment_type_correct_count", 0) or 0
    )
    left_segment_type_incorrect_count = int(
        metrics_payload.get("left_segment_type_incorrect_count", 0) or 0
    )
    right_segment_type_primary_value = _primary_eval_metric_value(metrics_payload, "right_segment_type")
    right_segment_type_accuracy = float(metrics_payload.get("right_segment_type_accuracy", 0.0) or 0.0)
    right_segment_type_correct_count = int(
        metrics_payload.get("right_segment_type_correct_count", 0) or 0
    )
    right_segment_type_incorrect_count = int(
        metrics_payload.get("right_segment_type_incorrect_count", 0) or 0
    )
    boundary_primary_value = _primary_eval_metric_value(metrics_payload, "boundary")
    boundary_correct_count = int(metrics_payload.get("boundary_correct_count", 0) or 0)
    boundary_incorrect_count = int(metrics_payload.get("boundary_incorrect_count", 0) or 0)
    boundary_true_positive_count = int(metrics_payload.get("boundary_true_positive_count", 0) or 0)
    boundary_false_positive_count = int(metrics_payload.get("boundary_false_positive_count", 0) or 0)
    boundary_false_negative_count = int(metrics_payload.get("boundary_false_negative_count", 0) or 0)
    boundary_true_negative_count = int(metrics_payload.get("boundary_true_negative_count", 0) or 0)
    estimated_correction_actions = int(review_cost_metrics.get("estimated_correction_actions", 0) or 0)
    merge_run_count = int(review_cost_metrics.get("merge_run_count", 0) or 0)
    split_run_count = int(review_cost_metrics.get("split_run_count", 0) or 0)
    lines = [
        "",
        "Final ML summary:",
        f"Rows: train={train_row_count}, validation={validation_row_count}, test={test_row_count}",
    ]
    heuristic_coverage_line = _heuristic_coverage_summary_line(training_metadata)
    if heuristic_coverage_line is not None:
        lines.append(heuristic_coverage_line)
    lines.extend(
        [
            (
                "Left segment type: "
                f"{left_segment_type_metric_spec.validation_metric_name}={left_segment_type_primary_value:.4f}, "
                f"accuracy={left_segment_type_accuracy:.4f}, "
                f"correct={left_segment_type_correct_count}, incorrect={left_segment_type_incorrect_count}"
            ),
            (
                "Right segment type: "
                f"{right_segment_type_metric_spec.validation_metric_name}={right_segment_type_primary_value:.4f}, "
                f"accuracy={right_segment_type_accuracy:.4f}, "
                f"correct={right_segment_type_correct_count}, incorrect={right_segment_type_incorrect_count}"
            ),
            (
                "Boundary: "
                f"{boundary_metric_spec.validation_metric_name}={boundary_primary_value:.4f}, "
                f"correct={boundary_correct_count}, incorrect={boundary_incorrect_count}, "
                f"tp={boundary_true_positive_count}, fp={boundary_false_positive_count}, "
                f"fn={boundary_false_negative_count}, tn={boundary_true_negative_count}"
            ),
            f"Review cost: merge_runs={merge_run_count}, split_runs={split_run_count}, estimated_actions={estimated_correction_actions}",
        ]
    )
    return Group(
        *(Text(line, no_wrap=False, overflow="fold") for line in lines),
        _build_predictor_confusion_table(metrics_payload, "left_segment_type"),
        _build_predictor_confusion_table(metrics_payload, "right_segment_type"),
    )


def render_final_summary(
    *,
    evaluation_metrics: dict[str, object],
    training_metadata: dict[str, object] | None = None,
) -> None:
    console.print(_render_eval_metrics_summary(evaluation_metrics, training_metadata))


def _run_training_and_evaluation(
    *,
    corpus_candidates_path: Path,
    split_manifest_path: Path,
    model_dir: Path,
    eval_dir: Path,
    mode: str,
    restart: bool,
    preset: str | None,
    train_minutes: float | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    train_command = [
        "uv",
        "run",
        "--no-default-groups",
        "--group",
        "autogluon",
        "python3",
        _script_ref("train_ml_boundary_verifier.py"),
        str(corpus_candidates_path.parent),
        "--model-run-id",
        model_dir.name,
        "--mode",
        mode,
    ]
    if preset:
        train_command.extend(["--preset", preset])
    if train_minutes is not None:
        train_command.extend(["--train-minutes", str(train_minutes)])
    if restart:
        train_command.append("--overwrite")
    _print_compact_command("Train verifier", train_command)
    _run_command(train_command)
    training_metadata = None
    training_metadata_path = model_dir / TRAINING_METADATA_FILENAME
    if training_metadata_path.is_file():
        training_metadata = _load_json_object(training_metadata_path)

    evaluate_command = [
        "uv",
        "run",
        "--no-default-groups",
        "--group",
        "autogluon",
        "python3",
        _script_ref("evaluate_ml_boundary_verifier.py"),
        str(corpus_candidates_path.parent),
        "--model-run-id",
        model_dir.name,
    ]
    if restart:
        evaluate_command.append("--overwrite")
    _print_compact_command("Evaluate verifier", evaluate_command)
    _run_command(evaluate_command)
    metrics_path = eval_dir / EVALUATION_METRICS_FILENAME
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Expected evaluation metrics artifact: {metrics_path}")
    metrics_payload = _load_json_object(metrics_path)
    render_final_summary(evaluation_metrics=metrics_payload, training_metadata=training_metadata)
    return metrics_payload, training_metadata


def _write_pipeline_summary(
    *,
    summary_path: Path,
    day_dirs: Sequence[Path],
    day_workspaces: Sequence[Path],
    corpus_candidates_path: Path,
    day_metadata_path: Path,
    split_manifest_path: Path | None,
    validation_report_path: Path | None,
    dataset_contract: dict[str, object],
    model_dir: Path | None,
    eval_dir: Path | None,
    evaluation_metrics: dict[str, object] | None,
    training_metadata: dict[str, object] | None,
    required_heldout_classes: Sequence[str],
    requested_split_strategy: str,
    effective_split_strategy: str,
    split_config: SplitConfig,
    mode: str,
    prepare_only: bool,
    note: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "day_dirs": [str(path) for path in day_dirs],
        "day_workspaces": [str(path) for path in day_workspaces],
        "corpus_candidates_csv": str(corpus_candidates_path),
        "day_metadata_csv": str(day_metadata_path),
        "required_heldout_classes": list(required_heldout_classes),
        "requested_split_strategy": requested_split_strategy,
        "effective_split_strategy": effective_split_strategy,
        "train_fraction": split_config.train_fraction,
        "validation_fraction": split_config.validation_fraction,
        "test_fraction": split_config.test_fraction,
        "split_seed": split_config.seed,
        "mode": mode,
        "prepare_only": prepare_only,
        "window_radius": int(dataset_contract["window_radius"]),
        "candidate_rule_name": dataset_contract["candidate_rule_name"],
        "candidate_rule_version": dataset_contract["candidate_rule_version"],
        "candidate_rule_params_json": dataset_contract["candidate_rule_params_json"],
        "descriptor_schema_version": dataset_contract["descriptor_schema_version"],
    }
    if split_manifest_path is not None:
        payload["split_manifest_csv"] = str(split_manifest_path)
    if validation_report_path is not None:
        payload["validation_report_json"] = str(validation_report_path)
    if model_dir is not None:
        payload["model_dir"] = str(model_dir)
    if eval_dir is not None:
        payload["eval_dir"] = str(eval_dir)
    if evaluation_metrics is not None:
        payload["evaluation_metrics"] = evaluation_metrics
    if training_metadata is not None:
        payload["training_metadata"] = training_metadata
        payload["training_preset"] = training_metadata.get("training_preset")
        payload["train_minutes"] = training_metadata.get("train_minutes")
        payload["time_limit_seconds"] = training_metadata.get("time_limit_seconds")
        heuristic_coverage = _heuristic_coverage_summary_payload(training_metadata)
        if heuristic_coverage is not None:
            payload["heuristic_boundary_coverage"] = heuristic_coverage
    if note:
        payload["note"] = note
    atomic_write_json(summary_path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    day_dirs = _resolve_days(args.day_dirs)
    split_config = resolve_split_config(args, day_dirs)
    day_workspaces: list[Path] = []
    for day_dir in day_dirs:
        workspace_dir = _prepare_single_day(day_dir, restart=args.restart)
        day_workspaces.append(workspace_dir)

    if args.corpus_workspace:
        corpus_workspace = Path(args.corpus_workspace).expanduser().resolve()
    else:
        corpus_workspace = (day_workspaces[0] / "ml_boundary_corpus").resolve()
    corpus_workspace.mkdir(parents=True, exist_ok=True)

    candidate_csv_paths = [workspace / "ml_boundary_candidates.csv" for workspace in day_workspaces]
    try:
        dataset_contract = _load_dataset_contract(day_workspaces)
        merged_headers = candidate_row_headers(
            window_radius=int(dataset_contract["window_radius"]),
            include_thumbnail=True,
        )
        merged_rows = _merge_candidate_rows(candidate_csv_paths, expected_headers=merged_headers)
        dataset_contract = _validate_dataset_contract_against_candidate_rows(
            dataset_contract,
            merged_rows,
        )
        _validate_corpus_size(merged_rows)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Error: {error}[/red]")
        return 1
    corpus_candidates_path = corpus_workspace / CORPUS_CANDIDATES_FILENAME
    atomic_write_csv(corpus_candidates_path, merged_headers, merged_rows)
    console.print(f"Wrote merged candidate dataset: {corpus_candidates_path}")

    day_metadata_rows = _build_day_metadata_rows(merged_rows)
    day_metadata_headers = ["day_id", "segment_types", "year", "camera", "domain_shift_hint"]
    day_metadata_path = corpus_workspace / DAY_METADATA_FILENAME
    atomic_write_csv(day_metadata_path, day_metadata_headers, day_metadata_rows)
    console.print(f"Wrote day metadata: {day_metadata_path}")

    corpus_attrition_path = corpus_workspace / CORPUS_ATTRITION_FILENAME
    atomic_write_json(
        corpus_attrition_path,
        _build_corpus_attrition_report(day_workspaces),
    )
    console.print(f"Wrote corpus attrition: {corpus_attrition_path}")

    required_heldout_classes = _required_classes(
        day_metadata_rows,
        explicit_required=args.required_heldout_classes,
    )
    split_rows, effective_split_strategy = _build_corpus_split_rows(
        merged_rows,
        split_config,
        required_heldout_classes=required_heldout_classes,
    )
    split_manifest_path = corpus_workspace / SPLIT_MANIFEST_FILENAME
    atomic_write_csv(split_manifest_path, SPLIT_MANIFEST_HEADERS, split_rows)
    console.print(f"Wrote split manifest: {split_manifest_path}")

    validate_command = [
        sys.executable,
        _script_ref("validate_ml_boundary_dataset.py"),
        str(corpus_workspace),
        "--attrition-json",
        CORPUS_ATTRITION_FILENAME,
        "--split-manifest-csv",
        SPLIT_MANIFEST_FILENAME,
        "--report-json",
        VALIDATION_REPORT_FILENAME,
    ]
    if required_heldout_classes:
        validate_command.append("--required-heldout-classes")
        validate_command.extend(required_heldout_classes)
    _print_compact_command("Validate corpus dataset", validate_command)
    _run_command(validate_command)

    model_dir: Path | None = None
    eval_dir: Path | None = None
    evaluation_metrics: dict[str, object] | None = None
    training_metadata: dict[str, object] | None = None
    if not args.prepare_only:
        model_dir = corpus_workspace / "ml_boundary_models" / args.model_run_id
        eval_dir = corpus_workspace / "ml_boundary_eval" / args.model_run_id
        evaluation_metrics, training_metadata = _run_training_and_evaluation(
            corpus_candidates_path=corpus_candidates_path,
            split_manifest_path=split_manifest_path,
            model_dir=model_dir,
            eval_dir=eval_dir,
            mode=args.mode,
            restart=args.restart,
            preset=args.preset,
            train_minutes=args.train_minutes,
        )

    summary_path = corpus_workspace / PIPELINE_SUMMARY_FILENAME
    _write_pipeline_summary(
        summary_path=summary_path,
        day_dirs=day_dirs,
        day_workspaces=day_workspaces,
        corpus_candidates_path=corpus_candidates_path,
        day_metadata_path=day_metadata_path,
        split_manifest_path=split_manifest_path,
        validation_report_path=corpus_workspace / VALIDATION_REPORT_FILENAME,
        dataset_contract=dataset_contract,
        model_dir=model_dir,
        eval_dir=eval_dir,
        evaluation_metrics=evaluation_metrics,
        training_metadata=training_metadata,
        required_heldout_classes=required_heldout_classes,
        requested_split_strategy=split_config.strategy,
        effective_split_strategy=effective_split_strategy,
        split_config=split_config,
        mode=args.mode,
        prepare_only=args.prepare_only,
    )
    console.print(f"Wrote pipeline summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
