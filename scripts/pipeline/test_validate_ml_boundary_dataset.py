import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import _serialize_candidate_row
from build_ml_boundary_candidate_dataset import candidate_row_headers
from build_ml_boundary_candidate_dataset import build_candidate_rows
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.ml_boundary_truth import build_final_photo_truth
from lib.ml_boundary_truth import load_truth_row
from validate_ml_boundary_dataset import (
    build_validation_report,
    main,
    validate_attrition_report,
    validate_candidate_row,
    validate_split_manifest,
)


def _candidate_row(
    *,
    day_id: str = "20250325",
    window_radius: int = 2,
    candidate_rule_version: str = "gap-v1",
    left_segment_id: str = "seg-a",
    right_segment_id: str = "seg-b",
    left_segment_type: str = "performance",
    right_segment_type: str = "ceremony",
    frame_timestamps: tuple[str, ...] = ("1.0", "2.0", "3.0", "25.0"),
) -> dict[str, str]:
    headers = candidate_row_headers(window_radius=window_radius, include_thumbnail=True)
    row = {header: "" for header in headers}
    frame_count = window_radius * 2
    frame_photo_ids = {
        f"frame_{frame_index:02d}_photo_id": f"p{frame_index}"
        for frame_index in range(1, frame_count + 1)
    }
    frame_relpaths = {
        f"frame_{frame_index:02d}_relpath": f"cam/p{frame_index}.jpg"
        for frame_index in range(1, frame_count + 1)
    }
    window_photo_ids = [f"p{frame_index}" for frame_index in range(1, frame_count + 1)]
    window_relative_paths = [f"cam/p{frame_index}.jpg" for frame_index in range(1, frame_count + 1)]
    center_left_index = window_radius
    center_right_index = window_radius + 1
    row.update(frame_photo_ids)
    row.update(
        {
            "day_id": day_id,
            "window_radius": str(window_radius),
            "center_left_photo_id": f"p{center_left_index}",
            "center_right_photo_id": f"p{center_right_index}",
            "left_segment_id": left_segment_id,
            "right_segment_id": right_segment_id,
            "left_segment_type": left_segment_type,
            "right_segment_type": right_segment_type,
            "segment_type": right_segment_type,
            "boundary": "true" if left_segment_id != right_segment_id else "false",
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": candidate_rule_version,
            "candidate_rule_params_json": (
                f'{{"gap_threshold_seconds":20.0,"window_radius":{window_radius}}}'
            ),
            "descriptor_schema_version": "not_included_v1",
            "split_name": "",
            "window_photo_ids": json.dumps(window_photo_ids),
            "window_relative_paths": json.dumps(window_relative_paths),
        }
    )
    row.update(frame_relpaths)
    for frame_index in range(1, frame_count + 1):
        suffix = f"{frame_index:02d}"
        row[f"frame_{suffix}_timestamp"] = frame_timestamps[frame_index - 1]
        row[f"frame_{suffix}_thumb_path"] = f"thumb/p{frame_index}.jpg"
        row[f"frame_{suffix}_preview_path"] = f"preview/p{frame_index}.jpg"
    candidate_identity_rule_version = (
        f'{candidate_rule_version}|{row["candidate_rule_params_json"]}'
    )
    row["candidate_id"] = canonical_candidate_id(
        day_id=day_id,
        center_left_photo_id=f"p{center_left_index}",
        center_right_photo_id=f"p{center_right_index}",
        candidate_rule_version=candidate_identity_rule_version,
    )
    return row


def _write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else candidate_row_headers(window_radius=2, include_thumbnail=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["day_id", "split_name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_attrition_report_accepts_sane_report() -> None:
    validate_attrition_report(
        {
            "candidate_count_generated": 3,
            "candidate_count_excluded_missing_window": 1,
            "candidate_count_excluded_missing_artifacts": 0,
            "candidate_count_retained": 2,
            "true_boundary_coverage_before_exclusions": 2,
            "true_boundary_coverage_after_exclusions": 1,
        }
    )


def test_validate_attrition_report_rejects_inconsistent_counts() -> None:
    try:
        validate_attrition_report(
            {
                "candidate_count_generated": 3,
                "candidate_count_excluded_missing_window": 1,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 1,
                "true_boundary_coverage_before_exclusions": 2,
                "true_boundary_coverage_after_exclusions": 1,
            }
        )
    except ValueError as exc:
        assert "candidate_count_generated must equal" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_attrition_report_rejects_coverage_count_above_generated_or_retained() -> None:
    try:
        validate_attrition_report(
            {
                "candidate_count_generated": 3,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 3,
                "true_boundary_coverage_before_exclusions": 4,
                "true_boundary_coverage_after_exclusions": 3,
            }
        )
    except ValueError as exc:
        assert "true_boundary_coverage_before_exclusions must not exceed candidate_count_generated" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_candidate_row_accepts_valid_row() -> None:
    validate_candidate_row(_candidate_row(), row_number=2)


def test_validator_accepts_distinct_left_and_right_segment_types() -> None:
    row = _candidate_row(left_segment_type="performance", right_segment_type="ceremony")
    validate_candidate_row(row, row_number=2)


def test_validate_candidate_row_rejects_blank_left_segment_type() -> None:
    row = _candidate_row()
    row["left_segment_type"] = ""

    with pytest.raises(ValueError, match="left_segment_type must not be blank"):
        validate_candidate_row(row, row_number=2)


def test_truth_loader_reads_left_and_right_segment_types() -> None:
    truth = load_truth_row(
        {
            "left_segment_type": "performance",
            "right_segment_type": "ceremony",
            "boundary": "1",
        }
    )
    assert truth.left_segment_type == "performance"
    assert truth.right_segment_type == "ceremony"


def test_validate_candidate_row_accepts_freshly_generated_candidate_row() -> None:
    photos = [
        {"photo_id": "p1", "order_idx": 1, "timestamp": 0.0, "relative_path": "cam/p1.jpg"},
        {"photo_id": "p2", "order_idx": 2, "timestamp": 1.0, "relative_path": "cam/p2.jpg"},
        {"photo_id": "p3", "order_idx": 3, "timestamp": 2.0, "relative_path": "cam/p3.jpg"},
        {"photo_id": "p4", "order_idx": 4, "timestamp": 30.0, "relative_path": "cam/p4.jpg"},
        {"photo_id": "p5", "order_idx": 5, "timestamp": 31.0, "relative_path": "cam/p5.jpg"},
    ]
    truth = build_final_photo_truth(
        [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p3", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p4", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p5", "segment_id": "s2", "segment_type": "ceremony"},
        ]
    )

    rows, _ = build_candidate_rows(
        photos=photos,
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    validate_candidate_row(_serialize_candidate_row(rows[0]), row_number=2)


def test_validate_candidate_row_rejects_boundary_segment_mismatch() -> None:
    row = _candidate_row(left_segment_id="seg-a", right_segment_id="seg-a")
    row["boundary"] = "true"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "boundary must match left/right segment-id equality" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_candidate_row_rejects_noncanonical_candidate_id() -> None:
    row = _candidate_row()
    row["candidate_id"] = "badcafef00d1234"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "candidate_id does not match canonical_candidate_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_candidate_row_rejects_corrupted_window_arrays() -> None:
    row = _candidate_row()
    row["window_photo_ids"] = "[\"p1\",\"p2\",\"p3\"]"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "window_photo_ids must contain exactly 4 items" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_candidate_row_accepts_dynamic_window_radius_schema() -> None:
    row = _candidate_row(
        window_radius=3,
        frame_timestamps=("1.0", "2.0", "3.0", "25.0", "26.0", "27.0"),
    )

    validate_candidate_row(row, row_number=2)


def test_validate_candidate_row_rejects_legacy_window_size_column() -> None:
    row = _candidate_row()
    row["window_size"] = "4"

    with pytest.raises(ValueError, match="legacy columns are not allowed: window_size"):
        validate_candidate_row(row, row_number=2)


def test_validate_candidate_row_rejects_extra_frame_columns_for_declared_radius() -> None:
    row = _candidate_row()
    row["frame_05_photo_id"] = "p5"
    row["frame_05_relpath"] = "cam/p5.jpg"
    row["frame_05_timestamp"] = "26.0"
    row["frame_05_thumb_path"] = "thumb/p5.jpg"
    row["frame_05_preview_path"] = "preview/p5.jpg"

    with pytest.raises(
        ValueError,
        match="unexpected columns are not allowed: frame_05_photo_id, frame_05_preview_path, frame_05_relpath, frame_05_thumb_path, frame_05_timestamp",
    ):
        validate_candidate_row(row, row_number=2)


def test_validate_candidate_row_rejects_invalid_split_name() -> None:
    row = _candidate_row()
    row["split_name"] = "dev"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "split_name must be one of train, validation, test when present" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_candidate_row_rejects_blank_or_invalid_provenance_fields() -> None:
    row = _candidate_row()
    row["candidate_rule_name"] = ""

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "candidate_rule_name must not be blank" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    row = _candidate_row()
    row["candidate_rule_params_json"] = "[]"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "candidate_rule_params_json must decode to a JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    row = _candidate_row()
    row["descriptor_schema_version"] = ""

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "descriptor_schema_version must not be blank" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_split_manifest_requires_heldout_classes_when_requested() -> None:
    candidate_rows = [
        _candidate_row(day_id="20250324", right_segment_type="performance"),
        _candidate_row(day_id="20250325", right_segment_type="ceremony"),
        _candidate_row(day_id="20250326", right_segment_type="performance"),
    ]
    split_rows = [
        {"day_id": "20250324", "split_name": "train"},
        {"day_id": "20250325", "split_name": "validation"},
        {"day_id": "20250326", "split_name": "test"},
    ]

    try:
        validate_split_manifest(
            split_rows,
            candidate_rows,
            required_classes=["performance", "ceremony"],
        )
    except ValueError as exc:
        assert "split validation is missing required classes" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_split_manifest_rejects_duplicate_candidate_ids_for_candidate_keyed_manifests() -> None:
    first_row = _candidate_row(day_id="20250324", candidate_rule_version="gap-v1")
    duplicate_row = _candidate_row(day_id="20250325", candidate_rule_version="gap-v2")
    duplicate_row["candidate_id"] = first_row["candidate_id"]

    try:
        validate_split_manifest(
            [{"candidate_id": first_row["candidate_id"], "split_name": "train"}],
            [first_row, duplicate_row],
            manifest_key="candidate_id",
        )
    except ValueError as exc:
        assert "candidate rows contain duplicate candidate_id values" in str(exc)
        assert first_row["candidate_id"] in str(exc)
    else:
        raise AssertionError("expected duplicate candidate_id values to be rejected")


def test_validate_split_manifest_accepts_candidate_level_assignments() -> None:
    candidate_rows = [
        _candidate_row(day_id="20250325", candidate_rule_version="gap-v1"),
        _candidate_row(day_id="20250325", candidate_rule_version="gap-v2"),
    ]
    split_rows = [
        {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
        {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "validation"},
    ]

    validate_split_manifest(split_rows, candidate_rows)


def test_validate_split_manifest_rejects_ambiguous_manifest_keys() -> None:
    candidate_rows = [_candidate_row(day_id="20250325", candidate_rule_version="gap-v1")]
    split_rows = [
        {
            "day_id": "20250325",
            "candidate_id": candidate_rows[0]["candidate_id"],
            "split_name": "train",
        }
    ]

    with pytest.raises(
        ValueError,
        match="split manifest must not contain both day_id and candidate_id columns",
    ):
        validate_split_manifest(split_rows, candidate_rows)


def test_validate_split_manifest_rejects_empty_split_rows_direct_call() -> None:
    candidate_rows = [_candidate_row(day_id="20250325", candidate_rule_version="gap-v1")]

    with pytest.raises(ValueError, match="split manifest must contain at least one row"):
        validate_split_manifest([], candidate_rows)


def test_validate_split_manifest_requires_heldout_classes_for_candidate_level_assignments() -> None:
    candidate_rows = [
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v1",
            right_segment_type="performance",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v2",
            right_segment_type="performance",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v3",
            right_segment_type="ceremony",
        ),
    ]
    split_rows = [
        {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
        {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "validation"},
        {"candidate_id": candidate_rows[2]["candidate_id"], "split_name": "test"},
    ]

    with pytest.raises(ValueError, match="split validation is missing required classes: ceremony"):
        validate_split_manifest(
            split_rows,
            candidate_rows,
            required_classes=["performance", "ceremony"],
        )


def test_validate_split_manifest_rejects_invalid_split_name() -> None:
    candidate_rows = [_candidate_row(day_id="20250324", right_segment_type="performance")]
    split_rows = [{"day_id": "20250324", "split_name": "dev"}]

    try:
        validate_split_manifest(split_rows, candidate_rows)
    except ValueError as exc:
        assert "split manifest split_name must be one of: train, validation, test" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_validation_report_marks_hard_negative_coverage_unavailable_when_missing() -> None:
    report = build_validation_report(
        candidate_rows=[_candidate_row(right_segment_type="performance")],
        attrition_report={
            "candidate_count_generated": 1,
            "candidate_count_excluded_missing_window": 0,
            "candidate_count_excluded_missing_artifacts": 0,
            "candidate_count_retained": 1,
            "true_boundary_coverage_before_exclusions": 1,
            "true_boundary_coverage_after_exclusions": 1,
        },
    )

    assert report["candidate_row_count"] == 1
    assert report["class_balance_by_segment_type"] == {"performance": 1}
    assert report["attrition_exclusion_counts"] == {
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
    }
    assert report["hard_negative_coverage"] == {
        "available": False,
        "reason": "candidate CSV does not include is_hard_negative",
    }


def test_cli_main_validates_candidate_csv_attrition_and_split_manifest(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "ml_boundary_candidates.csv"
    attrition_json = tmp_path / "ml_boundary_attrition.json"
    split_manifest_csv = tmp_path / "ml_boundary_splits.csv"
    report_json = tmp_path / "ml_boundary_validation_report.json"

    candidate_rows = [
        _candidate_row(day_id="20250323", right_segment_type="performance"),
        _candidate_row(
            day_id="20250324",
            left_segment_type="ceremony",
            right_segment_type="ceremony",
            left_segment_id="seg-c",
            right_segment_id="seg-c",
        ),
        _candidate_row(
            day_id="20250325",
            left_segment_type="warmup",
            right_segment_type="warmup",
            left_segment_id="seg-w",
            right_segment_id="seg-w",
        ),
        _candidate_row(day_id="20250326", right_segment_type="performance"),
        _candidate_row(
            day_id="20250327",
            left_segment_type="ceremony",
            right_segment_type="ceremony",
            left_segment_id="seg-c2",
            right_segment_id="seg-c2",
        ),
        _candidate_row(
            day_id="20250328",
            left_segment_type="warmup",
            right_segment_type="warmup",
            left_segment_id="seg-w2",
            right_segment_id="seg-w2",
        ),
    ]
    _write_candidate_csv(candidate_csv, candidate_rows)
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 6,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 6,
                "true_boundary_coverage_before_exclusions": 2,
                "true_boundary_coverage_after_exclusions": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_split_manifest(
        split_manifest_csv,
        [
            {"day_id": "20250323", "split_name": "train"},
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "train"},
            {"day_id": "20250326", "split_name": "validation"},
            {"day_id": "20250327", "split_name": "validation"},
            {"day_id": "20250328", "split_name": "validation"},
            {"day_id": "20250329", "split_name": "test"},
            {"day_id": "20250330", "split_name": "test"},
            {"day_id": "20250331", "split_name": "test"},
        ],
    )
    candidate_rows.extend(
        [
            _candidate_row(day_id="20250329", right_segment_type="performance"),
            _candidate_row(
                day_id="20250330",
                left_segment_type="ceremony",
                right_segment_type="ceremony",
                left_segment_id="seg-c3",
                right_segment_id="seg-c3",
            ),
            _candidate_row(
                day_id="20250331",
                left_segment_type="warmup",
                right_segment_type="warmup",
                left_segment_id="seg-w3",
                right_segment_id="seg-w3",
            ),
        ]
    )
    _write_candidate_csv(candidate_csv, candidate_rows)
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 9,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 9,
                "true_boundary_coverage_before_exclusions": 3,
                "true_boundary_coverage_after_exclusions": 3,
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            str(candidate_csv),
            "--attrition-json",
            str(attrition_json),
            "--split-manifest-csv",
            str(split_manifest_csv),
            "--required-heldout-classes",
            "performance",
            "ceremony",
            "warmup",
            "--report-json",
            str(report_json),
        ]
    )

    assert result == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["candidate_row_count"] == 9
    assert payload["class_balance_by_segment_type"] == {
        "ceremony": 3,
        "performance": 3,
        "warmup": 3,
    }
    assert payload["attrition_exclusion_counts"] == {
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
    }
    assert payload["hard_negative_coverage"] == {
        "available": False,
        "reason": "candidate CSV does not include is_hard_negative",
    }


def test_cli_main_accepts_candidate_keyed_split_manifest(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "ml_boundary_candidates.csv"
    attrition_json = tmp_path / "ml_boundary_attrition.json"
    split_manifest_csv = tmp_path / "ml_boundary_splits.csv"
    report_json = tmp_path / "ml_boundary_validation_report.json"

    candidate_rows = [
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v1",
            right_segment_type="performance",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v2",
            left_segment_type="ceremony",
            right_segment_type="ceremony",
            left_segment_id="seg-c1",
            right_segment_id="seg-c1",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v3",
            left_segment_type="warmup",
            right_segment_type="warmup",
            left_segment_id="seg-w1",
            right_segment_id="seg-w1",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v4",
            right_segment_type="performance",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v5",
            left_segment_type="ceremony",
            right_segment_type="ceremony",
            left_segment_id="seg-c2",
            right_segment_id="seg-c2",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v6",
            left_segment_type="warmup",
            right_segment_type="warmup",
            left_segment_id="seg-w2",
            right_segment_id="seg-w2",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v7",
            right_segment_type="performance",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v8",
            left_segment_type="ceremony",
            right_segment_type="ceremony",
            left_segment_id="seg-c3",
            right_segment_id="seg-c3",
        ),
        _candidate_row(
            day_id="20250325",
            candidate_rule_version="gap-v9",
            left_segment_type="warmup",
            right_segment_type="warmup",
            left_segment_id="seg-w3",
            right_segment_id="seg-w3",
        ),
    ]
    _write_candidate_csv(candidate_csv, candidate_rows)
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 9,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 9,
                "true_boundary_coverage_before_exclusions": 3,
                "true_boundary_coverage_after_exclusions": 3,
            }
        ),
        encoding="utf-8",
    )
    _write_split_manifest(
        split_manifest_csv,
        [
            {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
            {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "train"},
            {"candidate_id": candidate_rows[2]["candidate_id"], "split_name": "train"},
            {"candidate_id": candidate_rows[3]["candidate_id"], "split_name": "validation"},
            {"candidate_id": candidate_rows[4]["candidate_id"], "split_name": "validation"},
            {"candidate_id": candidate_rows[5]["candidate_id"], "split_name": "validation"},
            {"candidate_id": candidate_rows[6]["candidate_id"], "split_name": "test"},
            {"candidate_id": candidate_rows[7]["candidate_id"], "split_name": "test"},
            {"candidate_id": candidate_rows[8]["candidate_id"], "split_name": "test"},
        ],
    )

    result = main(
        [
            str(candidate_csv),
            "--attrition-json",
            str(attrition_json),
            "--split-manifest-csv",
            str(split_manifest_csv),
            "--required-heldout-classes",
            "performance",
            "ceremony",
            "warmup",
            "--report-json",
            str(report_json),
        ]
    )

    assert result == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["candidate_row_count"] == 9


def test_cli_main_resolves_day_workspace_defaults_from_vocatio(tmp_path: Path) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "external-workspace"
    candidate_csv = workspace_dir / "ml_boundary_candidates.csv"
    attrition_json = workspace_dir / "ml_boundary_attrition.json"
    report_json = workspace_dir / "ml_boundary_validation_report.json"

    day_dir.mkdir()
    workspace_dir.mkdir()
    (day_dir / ".vocatio").write_text(
        f"WORKSPACE_DIR={workspace_dir}\n",
        encoding="utf-8",
    )

    _write_candidate_csv(candidate_csv, [_candidate_row(day_id="20250325")])
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 1,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 1,
                "true_boundary_coverage_before_exclusions": 1,
                "true_boundary_coverage_after_exclusions": 1,
            }
        ),
        encoding="utf-8",
    )

    result = main([str(day_dir)])

    assert result == 0
    assert report_json.is_file()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["candidate_row_count"] == 1
    assert payload["class_balance_by_segment_type"] == {"ceremony": 1}
    assert payload["attrition_exclusion_counts"] == {
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
    }


def test_cli_main_expands_user_home_in_day_mode_workspace_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "external-workspace"
    home_dir = tmp_path / "home"
    candidate_csv = workspace_dir / "ml_boundary_candidates.csv"
    attrition_json = workspace_dir / "ml_boundary_attrition.json"
    report_json = home_dir / "ml_boundary_validation_report.json"

    day_dir.mkdir()
    workspace_dir.mkdir()
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    (day_dir / ".vocatio").write_text(
        f"WORKSPACE_DIR={workspace_dir}\n",
        encoding="utf-8",
    )

    _write_candidate_csv(candidate_csv, [_candidate_row(day_id="20250325")])
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 1,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 1,
                "true_boundary_coverage_before_exclusions": 1,
                "true_boundary_coverage_after_exclusions": 1,
            }
        ),
        encoding="utf-8",
    )

    result = main([str(day_dir), "--report-json", "~/ml_boundary_validation_report.json"])

    assert result == 0
    assert report_json.is_file()
    assert not (workspace_dir / "~" / "ml_boundary_validation_report.json").exists()


def test_cli_main_treats_mistyped_non_csv_candidate_path_as_explicit_file(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "ml_boundary_candidates.cvs"

    with pytest.raises(SystemExit, match="--attrition-json is required when candidate_csv is a CSV path"):
        main([str(candidate_csv)])


def test_cli_main_rejects_header_only_candidate_csv_with_extra_frame_columns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_csv = tmp_path / "ml_boundary_candidates.csv"
    attrition_json = tmp_path / "ml_boundary_attrition.json"
    headers = candidate_row_headers(window_radius=2, include_thumbnail=True) + [
        "frame_05_photo_id",
        "frame_05_relpath",
        "frame_05_timestamp",
        "frame_05_thumb_path",
        "frame_05_preview_path",
    ]
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
    attrition_json.write_text(
        json.dumps(
            {
                "candidate_count_generated": 0,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": 0,
                "candidate_count_retained": 0,
                "true_boundary_coverage_before_exclusions": 0,
                "true_boundary_coverage_after_exclusions": 0,
            }
        ),
        encoding="utf-8",
    )

    result = main([str(candidate_csv), "--attrition-json", str(attrition_json)])

    assert result == 1
    captured = capsys.readouterr()
    assert "unexpected columns are not allowed" in captured.err
    assert "frame_05_photo_id" in captured.err
