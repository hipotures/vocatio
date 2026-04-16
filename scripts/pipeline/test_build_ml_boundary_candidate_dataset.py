import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import build_candidate_rows
from lib.ml_boundary_truth import build_final_photo_truth


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
            {"photo_id": "p7", "segment_id": "s2", "segment_type": "ceremony"},
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
    }
    assert len(rows) == 1
    assert rows[0]["center_left_photo_id"] == "p4"
    assert rows[0]["center_right_photo_id"] == "p5"
    assert rows[0]["segment_type"] == "ceremony"
    assert rows[0]["boundary"] is False
