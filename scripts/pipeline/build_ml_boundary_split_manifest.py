#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
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

from lib.ml_boundary_truth import VALID_SEGMENT_TYPES
from lib.pipeline_io import atomic_write_csv


console = Console(stderr=True)

DEFAULT_OUTPUT_FILENAME = "ml_boundary_splits.csv"
OUTPUT_HEADERS = ["day_id", "split_name"]
HELDOUT_SPLIT_NAMES = ("validation", "test")
REQUIRED_DAY_METADATA_HEADERS = ("day_id", "segment_types")


@dataclass(frozen=True)
class DayMetadata:
    day_id: str
    segment_types: frozenset[str]
    domain_key: tuple[str, ...] | None


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit ML boundary split manifest from day-level metadata CSV files."
    )
    parser.add_argument(
        "day_metadata_csv",
        nargs="?",
        help="Path to one day-metadata CSV file.",
    )
    parser.add_argument(
        "--day-metadata-csv",
        action="append",
        dest="day_metadata_csvs",
        default=[],
        help="Additional day-metadata CSV file. Repeat as needed.",
    )
    parser.add_argument(
        "--required-heldout-classes",
        nargs="+",
        default=[],
        help="Classes that validation and test must each cover under day-level isolation.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output split-manifest CSV path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    return parser.parse_args(argv)


def _require_non_blank_text(row: Mapping[str, object], field_name: str, *, row_number: int) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"row {row_number}: {field_name} must not be blank")
    return str(value).strip()


def _parse_segment_types(value: str, *, row_number: int) -> frozenset[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row_number}: segment_types must be valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"row {row_number}: segment_types must be a non-empty JSON array")

    segment_types: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"row {row_number}: segment_types must contain only non-blank strings"
            )
        normalized = item.strip()
        if normalized not in VALID_SEGMENT_TYPES:
            choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
            raise ValueError(
                f"row {row_number}: segment_types entries must be one of: {choices}"
            )
        segment_types.append(normalized)
    return frozenset(segment_types)


def _domain_key_for_row(row: Mapping[str, str]) -> tuple[str, ...] | None:
    components: list[str] = []
    for field_name in ("domain_shift_hint", "year", "camera"):
        value = str(row.get(field_name, "") or "").strip()
        if value:
            components.append(f"{field_name}={value}")
    if not components:
        return None
    return tuple(components)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            header for header in REQUIRED_DAY_METADATA_HEADERS if header not in set(fieldnames)
        ]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _load_day_metadata(paths: Sequence[Path]) -> list[DayMetadata]:
    metadata_by_day: dict[str, DayMetadata] = {}
    task_description = "read metadata files".ljust(25)

    with _progress() as progress:
        task_id = progress.add_task(task_description, total=len(paths))
        for path in paths:
            for row_number, row in enumerate(_read_csv_rows(path), start=2):
                day_id = _require_non_blank_text(row, "day_id", row_number=row_number)
                if day_id in metadata_by_day:
                    raise ValueError(f"Duplicate day_id across metadata inputs: {day_id}")
                metadata_by_day[day_id] = DayMetadata(
                    day_id=day_id,
                    segment_types=_parse_segment_types(
                        _require_non_blank_text(row, "segment_types", row_number=row_number),
                        row_number=row_number,
                    ),
                    domain_key=_domain_key_for_row(row),
                )
            progress.advance(task_id)

    if len(metadata_by_day) < 3:
        raise ValueError("At least three day_id rows are required to build train, validation, and test splits")
    return sorted(metadata_by_day.values(), key=lambda item: item.day_id)


def _normalize_required_classes(required_classes: Sequence[str]) -> frozenset[str]:
    normalized: list[str] = []
    for value in required_classes:
        text = value.strip()
        if not text:
            raise ValueError("required held-out classes must not contain blank values")
        if text not in VALID_SEGMENT_TYPES:
            choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
            raise ValueError(f"required held-out classes must be drawn from: {choices}")
        normalized.append(text)
    return frozenset(normalized)


def _covered_classes(days: Sequence[DayMetadata]) -> frozenset[str]:
    covered: set[str] = set()
    for day in days:
        covered.update(day.segment_types)
    return frozenset(covered)


def _baseline_domain_key(days: Sequence[DayMetadata]) -> tuple[str, ...] | None:
    counts = Counter(day.domain_key for day in days)
    return min(
        counts,
        key=lambda key: (-counts[key], key is None, key or ()),
    )


def _supports_domain_shift(days: Sequence[DayMetadata], baseline_domain_key: tuple[str, ...] | None) -> bool:
    return any(
        day.domain_key is not None and day.domain_key != baseline_domain_key
        for day in days
    )


def _heldout_represents_domain_shift(
    heldout_days: Sequence[DayMetadata],
    baseline_domain_key: tuple[str, ...] | None,
) -> bool:
    for day in heldout_days:
        if day.domain_key is not None and day.domain_key != baseline_domain_key:
            return True
    return False


def _assignment_score(
    validation_days: Sequence[DayMetadata],
    test_days: Sequence[DayMetadata],
    baseline_domain_key: tuple[str, ...] | None,
) -> tuple[object, ...]:
    return (
        len(validation_days) + len(test_days),
        0 if _heldout_represents_domain_shift(test_days, baseline_domain_key) else 1,
        len(validation_days),
        tuple(day.day_id for day in validation_days),
        tuple(day.day_id for day in test_days),
    )


def _build_split_manifest_from_days(
    days: Sequence[DayMetadata],
    *,
    required_heldout_classes: Sequence[str],
) -> list[dict[str, str]]:
    if len(days) < 3:
        raise ValueError("At least three day_id rows are required to build train, validation, and test splits")

    required_classes = _normalize_required_classes(required_heldout_classes)
    available_classes = _covered_classes(days)
    if required_classes - available_classes:
        missing = ", ".join(sorted(required_classes - available_classes))
        raise ValueError(
            "Unable to satisfy required held-out class coverage because the available corpus is missing: "
            f"{missing}"
        )

    baseline_domain_key = _baseline_domain_key(days)
    require_domain_shift = _supports_domain_shift(days, baseline_domain_key)
    best_assignment: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_score: tuple[object, ...] | None = None
    day_indexes = tuple(range(len(days)))

    for validation_size in range(1, len(days) - 1):
        for validation_indexes in combinations(day_indexes, validation_size):
            validation_days = [days[index] for index in validation_indexes]
            if required_classes and not required_classes.issubset(_covered_classes(validation_days)):
                continue

            remaining_indexes = tuple(
                index for index in day_indexes if index not in validation_indexes
            )
            for test_size in range(1, len(remaining_indexes)):
                for test_indexes in combinations(remaining_indexes, test_size):
                    train_indexes = tuple(
                        index
                        for index in day_indexes
                        if index not in validation_indexes and index not in test_indexes
                    )
                    if not train_indexes:
                        continue

                    test_days = [days[index] for index in test_indexes]
                    if required_classes and not required_classes.issubset(_covered_classes(test_days)):
                        continue

                    if require_domain_shift and not (
                        _heldout_represents_domain_shift(validation_days, baseline_domain_key)
                        or _heldout_represents_domain_shift(test_days, baseline_domain_key)
                    ):
                        continue

                    score = _assignment_score(
                        validation_days,
                        test_days,
                        baseline_domain_key,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_assignment = (validation_indexes, test_indexes)

    if best_assignment is None:
        classes_text = ", ".join(sorted(required_classes)) if required_classes else "none requested"
        raise ValueError(
            "Unable to satisfy required held-out class coverage under day-level isolation "
            f"without leakage. Required held-out classes: {classes_text}."
        )

    validation_indexes, test_indexes = best_assignment
    split_by_day: dict[str, str] = {}
    for index, day in enumerate(days):
        split_name = "train"
        if index in validation_indexes:
            split_name = "validation"
        elif index in test_indexes:
            split_name = "test"
        split_by_day[day.day_id] = split_name

    return [
        {"day_id": day.day_id, "split_name": split_by_day[day.day_id]}
        for day in days
    ]


def build_split_manifest_rows(
    day_metadata_rows: Sequence[Mapping[str, str]],
    *,
    required_heldout_classes: Sequence[str] = (),
) -> list[dict[str, str]]:
    metadata_by_day: dict[str, DayMetadata] = {}
    for row_number, row in enumerate(day_metadata_rows, start=2):
        day_id = _require_non_blank_text(row, "day_id", row_number=row_number)
        if day_id in metadata_by_day:
            raise ValueError(f"Duplicate day_id in metadata rows: {day_id}")
        metadata_by_day[day_id] = DayMetadata(
            day_id=day_id,
            segment_types=_parse_segment_types(
                _require_non_blank_text(row, "segment_types", row_number=row_number),
                row_number=row_number,
            ),
            domain_key=_domain_key_for_row(row),
        )

    return _build_split_manifest_from_days(
        sorted(metadata_by_day.values(), key=lambda item: item.day_id),
        required_heldout_classes=required_heldout_classes,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_paths: list[Path] = []
    if args.day_metadata_csv:
        input_paths.append(Path(args.day_metadata_csv).expanduser())
    input_paths.extend(Path(value).expanduser() for value in args.day_metadata_csvs)
    if not input_paths:
        raise ValueError("Provide one day-metadata CSV path or repeat --day-metadata-csv")

    day_metadata = _load_day_metadata(input_paths)
    output_rows = _build_split_manifest_from_days(
        day_metadata,
        required_heldout_classes=args.required_heldout_classes,
    )

    output_path = Path(args.output_csv).expanduser()
    atomic_write_csv(output_path, OUTPUT_HEADERS, output_rows)
    console.print(f"Wrote split manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
