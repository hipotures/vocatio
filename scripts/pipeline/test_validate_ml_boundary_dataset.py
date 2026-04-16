import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
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
    candidate_rule_version: str = "gap-v1",
    left_segment_id: str = "seg-a",
    right_segment_id: str = "seg-b",
    left_segment_type: str = "performance",
    right_segment_type: str = "ceremony",
    frame_timestamps: tuple[str, str, str, str, str] = ("1.0", "2.0", "3.0", "25.0", "26.0"),
) -> dict[str, str]:
    row = {header: "" for header in CANDIDATE_ROW_HEADERS}
    frame_photo_ids = {
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    row.update(frame_photo_ids)
    row.update(
        {
            "day_id": day_id,
            "window_size": "5",
            "center_left_photo_id": "p3",
            "center_right_photo_id": "p4",
            "left_segment_id": left_segment_id,
            "right_segment_id": right_segment_id,
            "left_segment_type": left_segment_type,
            "right_segment_type": right_segment_type,
            "segment_type": right_segment_type,
            "boundary": "true" if left_segment_id != right_segment_id else "false",
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": candidate_rule_version,
            "candidate_rule_params_json": "{\"gap_threshold_seconds\":20.0}",
            "descriptor_schema_version": "not_included_v1",
            "split_name": "",
            "window_photo_ids": "[\"p1\",\"p2\",\"p3\",\"p4\",\"p5\"]",
            "window_relative_paths": "[\"cam/p1.jpg\",\"cam/p2.jpg\",\"cam/p3.jpg\",\"cam/p4.jpg\",\"cam/p5.jpg\"]",
            "frame_01_relpath": "cam/p1.jpg",
            "frame_02_relpath": "cam/p2.jpg",
            "frame_03_relpath": "cam/p3.jpg",
            "frame_04_relpath": "cam/p4.jpg",
            "frame_05_relpath": "cam/p5.jpg",
            "frame_01_timestamp": frame_timestamps[0],
            "frame_02_timestamp": frame_timestamps[1],
            "frame_03_timestamp": frame_timestamps[2],
            "frame_04_timestamp": frame_timestamps[3],
            "frame_05_timestamp": frame_timestamps[4],
            "frame_01_thumb_path": "thumb/p1.jpg",
            "frame_02_thumb_path": "thumb/p2.jpg",
            "frame_03_thumb_path": "thumb/p3.jpg",
            "frame_04_thumb_path": "thumb/p4.jpg",
            "frame_05_thumb_path": "thumb/p5.jpg",
            "frame_01_preview_path": "preview/p1.jpg",
            "frame_02_preview_path": "preview/p2.jpg",
            "frame_03_preview_path": "preview/p3.jpg",
            "frame_04_preview_path": "preview/p4.jpg",
            "frame_05_preview_path": "preview/p5.jpg",
        }
    )
    row["candidate_id"] = canonical_candidate_id(
        day_id=day_id,
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version=candidate_rule_version,
    )
    return row


def _write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_ROW_HEADERS)
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
    row["window_photo_ids"] = "[\"p1\",\"p2\",\"p3\",\"p4\"]"

    try:
        validate_candidate_row(row, row_number=2)
    except ValueError as exc:
        assert "window_photo_ids must contain exactly 5 items" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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
