#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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
REQUIRED_DAY_METADATA_HEADERS = ("day_id", "segment_types")


@dataclass(frozen=True)
class DayMetadata:
    day_id: str
    segment_types: frozenset[str]
    domain_key: tuple[str, ...] | None
    has_domain_shift_hint: bool


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


def _has_domain_shift_hint(row: Mapping[str, str]) -> bool:
    return bool(str(row.get("domain_shift_hint", "") or "").strip())


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
                try:
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
                        has_domain_shift_hint=_has_domain_shift_hint(row),
                    )
                except ValueError as exc:
                    raise ValueError(f"{path.name}: {exc}") from exc
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


def _is_domain_shift_day(day: DayMetadata, train_days: Sequence[DayMetadata]) -> bool:
    if day.domain_key is None:
        return False
    train_domain_keys = {item.domain_key for item in train_days if item.domain_key is not None}
    return day.domain_key not in train_domain_keys


def _split_has_domain_shift(split_days: Sequence[DayMetadata], train_days: Sequence[DayMetadata]) -> bool:
    return any(_is_domain_shift_day(day, train_days) for day in split_days)


def _split_has_hint_domain_shift(split_days: Sequence[DayMetadata], train_days: Sequence[DayMetadata]) -> bool:
    return any(
        day.has_domain_shift_hint and _is_domain_shift_day(day, train_days)
        for day in split_days
    )


def _missing_required_classes(
    selected_days: Sequence[DayMetadata],
    required_classes: frozenset[str],
) -> frozenset[str]:
    return frozenset(required_classes - _covered_classes(selected_days))


def _greedy_expand_split(
    selected_days: list[DayMetadata],
    available_days: list[DayMetadata],
    required_classes: frozenset[str],
) -> tuple[list[DayMetadata], list[DayMetadata]] | None:
    expanded_days = list(selected_days)
    remaining_days = list(available_days)

    while True:
        missing_classes = _missing_required_classes(expanded_days, required_classes)
        if not missing_classes:
            return expanded_days, remaining_days

        best_index: int | None = None
        best_score: tuple[object, ...] | None = None
        for index, day in enumerate(remaining_days):
            newly_covered = day.segment_types & missing_classes
            if not newly_covered:
                continue

            projected_split = expanded_days + [day]
            projected_train = [
                candidate for candidate_index, candidate in enumerate(remaining_days)
                if candidate_index != index
            ]
            score = (
                -len(newly_covered),
                0 if _split_has_hint_domain_shift(projected_split, projected_train) else 1,
                0 if _split_has_domain_shift(projected_split, projected_train) else 1,
                day.day_id,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        if best_index is None:
            return None

        expanded_days.append(remaining_days.pop(best_index))


def _assignment_score(
    validation_days: Sequence[DayMetadata],
    test_days: Sequence[DayMetadata],
    train_days: Sequence[DayMetadata],
) -> tuple[object, ...]:
    validation_is_domain_shift = _split_has_domain_shift(validation_days, train_days)
    test_is_domain_shift = _split_has_domain_shift(test_days, train_days)
    validation_is_hint_shift = _split_has_hint_domain_shift(validation_days, train_days)
    test_is_hint_shift = _split_has_hint_domain_shift(test_days, train_days)
    return (
        len(validation_days) + len(test_days),
        -int(validation_is_hint_shift) - int(test_is_hint_shift),
        0 if test_is_hint_shift else 1,
        -int(validation_is_domain_shift) - int(test_is_domain_shift),
        0 if test_is_domain_shift else 1,
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

    best_assignment: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    best_score: tuple[object, ...] | None = None
    day_indexes = tuple(range(len(days)))

    for validation_index in day_indexes:
        validation_day = days[validation_index]
        for test_index in day_indexes:
            if test_index == validation_index:
                continue
            test_day = days[test_index]
            base_remaining_days = [
                day
                for index, day in enumerate(days)
                if index not in {validation_index, test_index}
            ]

            for first_split_name in ("validation", "test"):
                validation_days = [validation_day]
                test_days = [test_day]
                remaining_days = list(base_remaining_days)

                split_order = (first_split_name, "test" if first_split_name == "validation" else "validation")
                construction_failed = False
                for split_name in split_order:
                    if split_name == "validation":
                        expanded = _greedy_expand_split(
                            validation_days,
                            remaining_days,
                            required_classes,
                        )
                        if expanded is None:
                            construction_failed = True
                            break
                        validation_days, remaining_days = expanded
                    else:
                        expanded = _greedy_expand_split(
                            test_days,
                            remaining_days,
                            required_classes,
                        )
                        if expanded is None:
                            construction_failed = True
                            break
                        test_days, remaining_days = expanded

                if construction_failed:
                    continue

                train_days = list(remaining_days)
                score = _assignment_score(validation_days, test_days, train_days)
                if best_score is None or score < best_score:
                    best_score = score
                    best_assignment = (
                        tuple(day.day_id for day in validation_days),
                        tuple(day.day_id for day in test_days),
                    )

    if best_assignment is None:
        classes_text = ", ".join(sorted(required_classes)) if required_classes else "none requested"
        raise ValueError(
            "Unable to satisfy required held-out class coverage under day-level isolation "
            "without leakage. "
            f"Required held-out classes: {classes_text}."
        )

    validation_day_ids, test_day_ids = best_assignment
    split_by_day: dict[str, str] = {}
    for index, day in enumerate(days):
        split_name = "train"
        if day.day_id in validation_day_ids:
            split_name = "validation"
        elif day.day_id in test_day_ids:
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
            has_domain_shift_hint=_has_domain_shift_hint(row),
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
