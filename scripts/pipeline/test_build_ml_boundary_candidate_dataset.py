import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import (
    DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1,
    build_candidate_rows,
    main,
)
from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS
from lib.ml_boundary_truth import build_final_photo_truth


def _write_media_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(MEDIA_MANIFEST_HEADERS)
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in fieldnames
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_truth_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["photo_id", "segment_id", "segment_type"])
        writer.writeheader()
        writer.writerows(rows)


def _build_manifest_photo_row(
    *,
    day: str,
    photo_id: str,
    relative_path: str,
    photo_order_index: int,
    start_epoch_ms: int,
) -> dict[str, str]:
    row = {header: "" for header in MEDIA_MANIFEST_HEADERS}
    row.update(
        {
            "day": day,
            "stream_id": "p-a7r5",
            "device": "a7r5",
            "media_type": "photo",
            "path": f"/data/{day}/{relative_path}",
            "relative_path": relative_path,
            "media_id": photo_id,
            "photo_id": photo_id,
            "filename": Path(relative_path).name,
            "extension": ".jpg",
            "capture_time_local": "2025-03-25T08:00:00",
            "capture_subsec": "000",
            "photo_order_index": str(photo_order_index),
            "start_local": "2025-03-25T08:00:00",
            "start_epoch_ms": str(start_epoch_ms),
        }
    )
    return row


def test_build_candidate_rows_excludes_candidates_without_full_window() -> None:
    photos = [
        {"photo_id": "p1", "order_idx": 1, "timestamp": 0.0},
        {"photo_id": "p2", "order_idx": 2, "timestamp": 1.0},
        {"photo_id": "p3", "order_idx": 3, "timestamp": 20.0},
    ]
    truth = build_final_photo_truth(
        [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p3", "segment_id": "s2", "segment_type": "ceremony"},
        ]
    )

    rows, report = build_candidate_rows(
        photos=photos,
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    assert rows == []
    assert report == {
        "candidate_count_generated": 1,
        "candidate_count_excluded_missing_window": 1,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 0,
        "true_boundary_coverage_before_exclusions": 1,
        "true_boundary_coverage_after_exclusions": 0,
    }


def test_build_candidate_rows_preserves_ordered_frame_fields_and_labels() -> None:
    photos = [
        {"photo_id": "p4", "order_idx": 4, "timestamp": 30.0, "relative_path": "cam/p4.jpg"},
        {"photo_id": "p1", "order_idx": 1, "timestamp": 0.0, "relative_path": "cam/p1.jpg"},
        {"photo_id": "p5", "order_idx": 5, "timestamp": 31.0, "relative_path": "cam/p5.jpg"},
        {"photo_id": "p3", "order_idx": 3, "timestamp": 2.0, "relative_path": "cam/p3.jpg"},
        {"photo_id": "p2", "order_idx": 2, "timestamp": 1.0, "relative_path": "cam/p2.jpg"},
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

    rows, report = build_candidate_rows(
        photos=photos,
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    assert report == {
        "candidate_count_generated": 1,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 1,
        "true_boundary_coverage_before_exclusions": 1,
        "true_boundary_coverage_after_exclusions": 1,
    }

    row = rows[0]
    assert row["frame_01_photo_id"] == "p1"
    assert row["frame_02_photo_id"] == "p2"
    assert row["frame_03_photo_id"] == "p3"
    assert row["frame_04_photo_id"] == "p4"
    assert row["frame_05_photo_id"] == "p5"
    assert row["frame_03_relpath"] == "cam/p3.jpg"
    assert row["frame_01_timestamp"] == 0.0
    assert row["frame_05_timestamp"] == 31.0
    assert row["window_photo_ids"] == ["p1", "p2", "p3", "p4", "p5"]
    assert row["window_relative_paths"] == [
        "cam/p1.jpg",
        "cam/p2.jpg",
        "cam/p3.jpg",
        "cam/p4.jpg",
        "cam/p5.jpg",
    ]
    assert row["center_left_photo_id"] == "p3"
    assert row["center_right_photo_id"] == "p4"
    assert row["left_segment_id"] == "s1"
    assert row["right_segment_id"] == "s2"
    assert row["left_segment_type"] == "performance"
    assert row["right_segment_type"] == "ceremony"
    assert row["segment_type"] == "ceremony"
    assert row["boundary"] is True
    assert row["day_id"] == "20250325"
    assert row["window_size"] == 5
    assert row["candidate_rule_name"] == "gap_threshold"
    assert row["candidate_rule_version"] == "gap-v1"
    assert row["candidate_rule_params_json"] == "{\"gap_threshold_seconds\":10.0}"
    assert row["descriptor_schema_version"] == DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1
    assert row["split_name"] == ""
    assert row["frame_01_thumb_path"] == ""
    assert row["frame_02_thumb_path"] == ""
    assert row["frame_03_thumb_path"] == ""
    assert row["frame_04_thumb_path"] == ""
    assert row["frame_05_thumb_path"] == ""
    assert row["frame_01_preview_path"] == ""
    assert row["frame_02_preview_path"] == ""
    assert row["frame_03_preview_path"] == ""
    assert row["frame_04_preview_path"] == ""
    assert row["frame_05_preview_path"] == ""


def test_build_candidate_rows_produces_deterministic_candidate_ids() -> None:
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

    first_rows, first_report = build_candidate_rows(
        photos=list(photos),
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )
    second_rows, second_report = build_candidate_rows(
        photos=list(reversed(photos)),
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    assert first_report == second_report
    assert [row["candidate_id"] for row in first_rows] == [
        row["candidate_id"] for row in second_rows
    ]
    assert first_rows[0]["candidate_id"] == second_rows[0]["candidate_id"]


def test_build_candidate_rows_uses_right_side_segment_type_and_boundary_labels() -> None:
    photos = [
        {"photo_id": "p1", "order_idx": 1, "timestamp": 0.0, "relative_path": "cam/p1.jpg"},
        {"photo_id": "p2", "order_idx": 2, "timestamp": 1.0, "relative_path": "cam/p2.jpg"},
        {"photo_id": "p3", "order_idx": 3, "timestamp": 2.0, "relative_path": "cam/p3.jpg"},
        {"photo_id": "p4", "order_idx": 4, "timestamp": 3.0, "relative_path": "cam/p4.jpg"},
        {"photo_id": "p5", "order_idx": 5, "timestamp": 31.0, "relative_path": "cam/p5.jpg"},
        {"photo_id": "p6", "order_idx": 6, "timestamp": 32.0, "relative_path": "cam/p6.jpg"},
        {"photo_id": "p7", "order_idx": 7, "timestamp": 60.0, "relative_path": "cam/p7.jpg"},
    ]
    truth = build_final_photo_truth(
        [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p3", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p4", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p5", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p6", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p7", "segment_id": "s3", "segment_type": "warmup"},
        ]
    )

    rows, report = build_candidate_rows(
        photos=photos,
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    assert report == {
        "candidate_count_generated": 2,
        "candidate_count_excluded_missing_window": 1,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 1,
        "true_boundary_coverage_before_exclusions": 1,
        "true_boundary_coverage_after_exclusions": 0,
    }
    assert len(rows) == 1
    assert rows[0]["center_left_photo_id"] == "p4"
    assert rows[0]["center_right_photo_id"] == "p5"
    assert rows[0]["segment_type"] == "ceremony"
    assert rows[0]["boundary"] is False


def test_build_candidate_rows_emits_schema_stable_proxy_columns_when_missing() -> None:
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

    row = rows[0]
    expected_empty_fields = [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
        "frame_05_thumb_path",
        "frame_01_preview_path",
        "frame_02_preview_path",
        "frame_03_preview_path",
        "frame_04_preview_path",
        "frame_05_preview_path",
        "split_name",
    ]

    for field_name in expected_empty_fields:
        assert field_name in row
        assert row[field_name] == ""
    assert row["descriptor_schema_version"] == DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1


def test_build_candidate_rows_counts_missing_artifacts_for_generated_true_boundary() -> None:
    photos = [
        {"photo_id": "p1", "order_idx": 1, "timestamp": 0.0, "relative_path": "cam/p1.jpg"},
        {"photo_id": "p2", "order_idx": 2, "timestamp": 1.0, "relative_path": "cam/p2.jpg"},
        {"photo_id": "p3", "order_idx": 3, "timestamp": 2.0},
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

    rows, report = build_candidate_rows(
        photos=photos,
        truth=truth,
        gap_threshold_seconds=10.0,
        day_id="20250325",
        candidate_rule_version="gap-v1",
    )

    assert rows == []
    assert report == {
        "candidate_count_generated": 1,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 1,
        "candidate_count_retained": 0,
        "true_boundary_coverage_before_exclusions": 1,
        "true_boundary_coverage_after_exclusions": 0,
    }


def test_main_writes_candidate_csv_and_reports_from_manifest_and_truth() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        output_csv = workspace_dir / "ml_boundary_candidates.csv"
        attrition_json = workspace_dir / "ml_boundary_attrition.json"
        report_json = workspace_dir / "ml_boundary_dataset_report.json"

        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms=0,
            )
            | {
                "thumb_path": "embedded_jpg/thumb/p1.jpg",
                "preview_path": "embedded_jpg/preview/p1.jpg",
            },
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p2",
                relative_path="p-a7r5/p2.jpg",
                photo_order_index=1,
                start_epoch_ms=1_000,
            )
            | {
                "thumb_path": "embedded_jpg/thumb/p2.jpg",
                "preview_path": "embedded_jpg/preview/p2.jpg",
            },
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p3",
                relative_path="p-a7r5/p3.jpg",
                photo_order_index=2,
                start_epoch_ms=2_000,
            )
            | {
                "thumb_path": "embedded_jpg/thumb/p3.jpg",
                "preview_path": "embedded_jpg/preview/p3.jpg",
            },
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p4",
                relative_path="p-a7r5/p4.jpg",
                photo_order_index=3,
                start_epoch_ms=30_000,
            )
            | {
                "thumb_path": "embedded_jpg/thumb/p4.jpg",
                "preview_path": "embedded_jpg/preview/p4.jpg",
            },
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p5",
                relative_path="p-a7r5/p5.jpg",
                photo_order_index=4,
                start_epoch_ms=31_000,
            )
            | {
                "thumb_path": "embedded_jpg/thumb/p5.jpg",
                "preview_path": "embedded_jpg/preview/p5.jpg",
            },
        ]
        truth_rows = [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p3", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p4", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p5", "segment_id": "s2", "segment_type": "ceremony"},
        ]

        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, truth_rows)

        exit_code = main(
            [
                str(day_dir),
                "--workspace-dir",
                str(workspace_dir),
                "--gap-threshold-seconds",
                "10.0",
            ]
        )

        assert exit_code == 0
        assert output_csv.is_file()
        assert attrition_json.is_file()
        assert report_json.is_file()

        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["day_id"] == "20250325"
        assert rows[0]["center_left_photo_id"] == "p3"
        assert rows[0]["center_right_photo_id"] == "p4"
        assert rows[0]["segment_type"] == "ceremony"
        assert rows[0]["boundary"] == "True"
        assert json.loads(rows[0]["window_photo_ids"]) == ["p1", "p2", "p3", "p4", "p5"]
        assert json.loads(rows[0]["window_relative_paths"]) == [
            "p-a7r5/p1.jpg",
            "p-a7r5/p2.jpg",
            "p-a7r5/p3.jpg",
            "p-a7r5/p4.jpg",
            "p-a7r5/p5.jpg",
        ]
        assert rows[0]["frame_01_thumb_path"] == "embedded_jpg/thumb/p1.jpg"
        assert rows[0]["frame_03_thumb_path"] == "embedded_jpg/thumb/p3.jpg"
        assert rows[0]["frame_05_thumb_path"] == "embedded_jpg/thumb/p5.jpg"
        assert rows[0]["frame_01_preview_path"] == "embedded_jpg/preview/p1.jpg"
        assert rows[0]["frame_03_preview_path"] == "embedded_jpg/preview/p3.jpg"
        assert rows[0]["frame_05_preview_path"] == "embedded_jpg/preview/p5.jpg"
        assert rows[0]["descriptor_schema_version"] == DESCRIPTOR_SCHEMA_VERSION_NOT_INCLUDED_V1

        attrition_payload = json.loads(attrition_json.read_text(encoding="utf-8"))
        assert attrition_payload == {
            "candidate_count_generated": 1,
            "candidate_count_excluded_missing_window": 0,
            "candidate_count_excluded_missing_artifacts": 0,
            "candidate_count_retained": 1,
            "true_boundary_coverage_before_exclusions": 1,
            "true_boundary_coverage_after_exclusions": 1,
        }

        report_payload = json.loads(report_json.read_text(encoding="utf-8"))
        assert report_payload["day_id"] == "20250325"
        assert report_payload["candidate_rule_name"] == "gap_threshold"
        assert report_payload["candidate_rule_version"] == "gap-v1"
        assert report_payload["gap_threshold_seconds"] == 10.0
        assert report_payload["candidate_count_generated"] == 1
        assert report_payload["candidate_count_excluded_missing_window"] == 0
        assert report_payload["candidate_count_excluded_missing_artifacts"] == 0
        assert report_payload["candidate_count_retained"] == 1
        assert report_payload["true_boundary_coverage_before_exclusions"] == 1
        assert report_payload["true_boundary_coverage_after_exclusions"] == 1


def test_main_uses_workspace_dir_from_vocatio_defaults() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = Path(tmp) / "external-workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()
        (day_dir / ".vocatio").write_text(
            f"WORKSPACE_DIR={workspace_dir}\n",
            encoding="utf-8",
        )

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        output_csv = workspace_dir / "ml_boundary_candidates.csv"
        attrition_json = workspace_dir / "ml_boundary_attrition.json"
        report_json = workspace_dir / "ml_boundary_dataset_report.json"

        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms=0,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p2",
                relative_path="p-a7r5/p2.jpg",
                photo_order_index=1,
                start_epoch_ms=1_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p3",
                relative_path="p-a7r5/p3.jpg",
                photo_order_index=2,
                start_epoch_ms=2_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p4",
                relative_path="p-a7r5/p4.jpg",
                photo_order_index=3,
                start_epoch_ms=30_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p5",
                relative_path="p-a7r5/p5.jpg",
                photo_order_index=4,
                start_epoch_ms=31_000,
            ),
        ]
        truth_rows = [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p3", "segment_id": "s1", "segment_type": "performance"},
            {"photo_id": "p4", "segment_id": "s2", "segment_type": "ceremony"},
            {"photo_id": "p5", "segment_id": "s2", "segment_type": "ceremony"},
        ]

        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, truth_rows)

        exit_code = main([str(day_dir)])

        assert exit_code == 0
        assert output_csv.is_file()
        assert attrition_json.is_file()
        assert report_json.is_file()


def test_main_expands_user_home_in_day_mode_workspace_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "external-workspace"
    home_dir = tmp_path / "home"
    report_json = home_dir / "ml_boundary_dataset_report.json"

    day_dir.mkdir()
    workspace_dir.mkdir()
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    (day_dir / ".vocatio").write_text(
        f"WORKSPACE_DIR={workspace_dir}\n",
        encoding="utf-8",
    )

    manifest_path = workspace_dir / "media_manifest.csv"
    truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
    output_csv = workspace_dir / "ml_boundary_candidates.csv"
    attrition_json = workspace_dir / "ml_boundary_attrition.json"

    manifest_rows = [
        _build_manifest_photo_row(
            day="20250325",
            photo_id="p1",
            relative_path="p-a7r5/p1.jpg",
            photo_order_index=0,
            start_epoch_ms=0,
        ),
        _build_manifest_photo_row(
            day="20250325",
            photo_id="p2",
            relative_path="p-a7r5/p2.jpg",
            photo_order_index=1,
            start_epoch_ms=1_000,
        ),
        _build_manifest_photo_row(
            day="20250325",
            photo_id="p3",
            relative_path="p-a7r5/p3.jpg",
            photo_order_index=2,
            start_epoch_ms=2_000,
        ),
        _build_manifest_photo_row(
            day="20250325",
            photo_id="p4",
            relative_path="p-a7r5/p4.jpg",
            photo_order_index=3,
            start_epoch_ms=30_000,
        ),
        _build_manifest_photo_row(
            day="20250325",
            photo_id="p5",
            relative_path="p-a7r5/p5.jpg",
            photo_order_index=4,
            start_epoch_ms=31_000,
        ),
    ]
    truth_rows = [
        {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p3", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p4", "segment_id": "s2", "segment_type": "ceremony"},
        {"photo_id": "p5", "segment_id": "s2", "segment_type": "ceremony"},
    ]

    _write_media_manifest(manifest_path, manifest_rows)
    _write_truth_csv(truth_path, truth_rows)

    exit_code = main([str(day_dir), "--report-json", "~/ml_boundary_dataset_report.json"])

    assert exit_code == 0
    assert output_csv.is_file()
    assert attrition_json.is_file()
    assert report_json.is_file()
    assert not (workspace_dir / "~" / "ml_boundary_dataset_report.json").exists()


@pytest.mark.parametrize("threshold_text", ["nan", "inf", "-inf", "0", "-1"])
def test_parse_args_rejects_non_finite_gap_thresholds(threshold_text: str) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "/tmp/20250325",
                "--gap-threshold-seconds",
                threshold_text,
            ]
        )


def test_main_rejects_empty_reviewed_truth_csv() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"

        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms=0,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p2",
                relative_path="p-a7r5/p2.jpg",
                photo_order_index=1,
                start_epoch_ms=1_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p3",
                relative_path="p-a7r5/p3.jpg",
                photo_order_index=2,
                start_epoch_ms=2_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p4",
                relative_path="p-a7r5/p4.jpg",
                photo_order_index=3,
                start_epoch_ms=30_000,
            ),
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p5",
                relative_path="p-a7r5/p5.jpg",
                photo_order_index=4,
                start_epoch_ms=31_000,
            ),
        ]

        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, [])

        with pytest.raises(ValueError, match="is empty"):
            main(
                [
                    str(day_dir),
                    "--workspace-dir",
                    str(workspace_dir),
                    "--gap-threshold-seconds",
                    "10.0",
                ]
            )


def test_main_rejects_path_collision_between_input_and_output() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms=0,
            ),
        ]
        truth_rows = [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        ]
        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, truth_rows)

        with pytest.raises(ValueError, match="Path collision"):
            main(
                [
                    str(day_dir),
                    "--workspace-dir",
                    str(workspace_dir),
                    "--output-csv",
                    "media_manifest.csv",
                    "--overwrite",
                ]
            )


def test_main_rejects_output_path_collision() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms=0,
            ),
        ]
        truth_rows = [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        ]
        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, truth_rows)

        with pytest.raises(ValueError, match="Path collision"):
            main(
                [
                    str(day_dir),
                    "--workspace-dir",
                    str(workspace_dir),
                    "--attrition-json",
                    "same.json",
                    "--report-json",
                    "same.json",
                    "--overwrite",
                ]
            )


def test_main_rejects_invalid_manifest_start_epoch_ms_with_context() -> None:
    with TemporaryDirectory() as tmp:
        day_dir = Path(tmp) / "20250325"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        manifest_path = workspace_dir / "media_manifest.csv"
        truth_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        manifest_rows = [
            _build_manifest_photo_row(
                day="20250325",
                photo_id="p1",
                relative_path="p-a7r5/p1.jpg",
                photo_order_index=0,
                start_epoch_ms="bad-ms",  # type: ignore[arg-type]
            ),
        ]
        truth_rows = [
            {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        ]
        _write_media_manifest(manifest_path, manifest_rows)
        _write_truth_csv(truth_path, truth_rows)

        with pytest.raises(ValueError, match=r"Invalid start_epoch_ms for photo_id=p1: bad-ms"):
            main(
                [
                    str(day_dir),
                    "--workspace-dir",
                    str(workspace_dir),
                    "--overwrite",
                ]
            )
