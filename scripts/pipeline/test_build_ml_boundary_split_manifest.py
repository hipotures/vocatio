import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_split_manifest import build_split_manifest_rows, main


DAY_METADATA_HEADERS = [
    "day_id",
    "year",
    "camera",
    "domain_shift_hint",
    "segment_types",
]


def _write_day_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAY_METADATA_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_build_split_manifest_rows_assigns_whole_days_and_prefers_domain_shift_in_test() -> None:
    rows = build_split_manifest_rows(
        [
            {
                "day_id": "20250323",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20250324",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20260323",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
        ],
        required_heldout_classes=["performance", "ceremony", "warmup"],
    )

    assert rows == [
        {"day_id": "20250323", "split_name": "validation"},
        {"day_id": "20250324", "split_name": "train"},
        {"day_id": "20260323", "split_name": "test"},
    ]


def test_build_split_manifest_rows_prefers_domain_shift_absent_from_train() -> None:
    rows = build_split_manifest_rows(
        [
            {
                "day_id": "20250323",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20250324",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20260323",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20260324",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
        ],
        required_heldout_classes=["performance", "ceremony", "warmup"],
    )

    assert rows == [
        {"day_id": "20250323", "split_name": "train"},
        {"day_id": "20250324", "split_name": "train"},
        {"day_id": "20260323", "split_name": "validation"},
        {"day_id": "20260324", "split_name": "test"},
    ]


def test_build_split_manifest_rows_fails_when_requested_coverage_requires_leakage() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_split_manifest_rows(
            [
                {
                    "day_id": "20250323",
                    "year": "2025",
                    "camera": "cam-a",
                    "domain_shift_hint": "",
                    "segment_types": '["performance","ceremony","warmup"]',
                },
                {
                    "day_id": "20250324",
                    "year": "2025",
                    "camera": "cam-a",
                    "domain_shift_hint": "",
                    "segment_types": '["performance"]',
                },
                {
                    "day_id": "20250325",
                    "year": "2025",
                    "camera": "cam-a",
                    "domain_shift_hint": "",
                    "segment_types": '["ceremony"]',
                },
            ],
            required_heldout_classes=["performance", "ceremony", "warmup"],
        )

    assert "Unable to satisfy required held-out class coverage" in str(exc_info.value)
    assert "day-level isolation" in str(exc_info.value)


def test_build_split_manifest_rows_fails_when_no_single_day_can_cover_heldout_classes() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_split_manifest_rows(
            [
                {
                    "day_id": "20250323",
                    "year": "2025",
                    "camera": "cam-a",
                    "domain_shift_hint": "",
                    "segment_types": '["performance"]',
                },
                {
                    "day_id": "20250324",
                    "year": "2025",
                    "camera": "cam-a",
                    "domain_shift_hint": "",
                    "segment_types": '["ceremony","warmup"]',
                },
                {
                    "day_id": "20260323",
                    "year": "2026",
                    "camera": "cam-b",
                    "domain_shift_hint": "new-camera",
                    "segment_types": '["performance"]',
                },
                {
                    "day_id": "20260324",
                    "year": "2026",
                    "camera": "cam-b",
                    "domain_shift_hint": "new-camera",
                    "segment_types": '["ceremony","warmup"]',
                },
            ],
            required_heldout_classes=["performance", "ceremony", "warmup"],
        )

    assert "single-day validation/test policy" in str(exc_info.value)
    assert "day-level isolation" in str(exc_info.value)


def test_main_merges_repeated_day_metadata_csv_inputs_and_writes_manifest(tmp_path: Path) -> None:
    first_csv = tmp_path / "days_a.csv"
    second_csv = tmp_path / "days_b.csv"
    output_csv = tmp_path / "ml_boundary_splits.csv"

    _write_day_metadata_csv(
        first_csv,
        [
            {
                "day_id": "20250323",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20250324",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
        ],
    )
    _write_day_metadata_csv(
        second_csv,
        [
            {
                "day_id": "20250325",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20260323",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
        ],
    )

    exit_code = main(
        [
            "--day-metadata-csv",
            str(first_csv),
            "--day-metadata-csv",
            str(second_csv),
            "--required-heldout-classes",
            "performance",
            "ceremony",
            "warmup",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    assert _read_csv_rows(output_csv) == [
        {"day_id": "20250323", "split_name": "validation"},
        {"day_id": "20250324", "split_name": "train"},
        {"day_id": "20250325", "split_name": "train"},
        {"day_id": "20260323", "split_name": "test"},
    ]


def test_main_includes_source_filename_in_row_level_metadata_errors(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_days.csv"
    _write_day_metadata_csv(
        bad_csv,
        [
            {
                "day_id": "20250323",
                "year": "2025",
                "camera": "cam-a",
                "domain_shift_hint": "",
                "segment_types": '["performance","unknown"]',
            },
            {
                "day_id": "20260323",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
            {
                "day_id": "20260324",
                "year": "2026",
                "camera": "cam-b",
                "domain_shift_hint": "new-camera",
                "segment_types": '["performance","ceremony","warmup"]',
            },
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        main(["--day-metadata-csv", str(bad_csv)])

    assert "bad_days.csv" in str(exc_info.value)
    assert "row 2" in str(exc_info.value)
