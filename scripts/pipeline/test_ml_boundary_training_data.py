import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.ml_boundary_training_data import (
    load_candidate_training_frame,
    load_training_data_bundle,
)


def _candidate_row(
    *,
    day_id: str,
    segment_type: str = "ceremony",
    boundary: str = "1",
) -> dict[str, str]:
    row = {header: "" for header in CANDIDATE_ROW_HEADERS}
    row.update(
        {
            "candidate_id": canonical_candidate_id(
                day_id=day_id,
                center_left_photo_id=f"{day_id}-p3",
                center_right_photo_id=f"{day_id}-p4",
                candidate_rule_version="gap-v1",
            ),
            "day_id": day_id,
            "window_size": "5",
            "center_left_photo_id": f"{day_id}-p3",
            "center_right_photo_id": f"{day_id}-p4",
            "left_segment_id": f"{day_id}-seg-a",
            "right_segment_id": f"{day_id}-seg-b",
            "left_segment_type": "performance",
            "right_segment_type": segment_type or "ceremony",
            "segment_type": segment_type,
            "boundary": boundary,
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": "gap-v1",
            "candidate_rule_params_json": "{\"gap_threshold_seconds\":20.0}",
            "descriptor_schema_version": "not_included_v1",
            "split_name": "",
            "window_photo_ids": "[\"p1\",\"p2\",\"p3\",\"p4\",\"p5\"]",
            "window_relative_paths": "[\"cam/p1.jpg\",\"cam/p2.jpg\",\"cam/p3.jpg\",\"cam/p4.jpg\",\"cam/p5.jpg\"]",
        }
    )
    for frame_index in range(1, 6):
        suffix = f"{frame_index:02d}"
        row[f"frame_{suffix}_photo_id"] = f"{day_id}-p{frame_index}"
        row[f"frame_{suffix}_relpath"] = f"cam/{day_id}-p{frame_index}.jpg"
        row[f"frame_{suffix}_timestamp"] = str(float(frame_index))
        row[f"frame_{suffix}_thumb_path"] = f"thumb/{day_id}-p{frame_index}.jpg"
        row[f"frame_{suffix}_preview_path"] = f"preview/{day_id}-p{frame_index}.jpg"
    return row


def _write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_ROW_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["day_id", "split_name"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_candidate_training_frame_reads_csv_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )

    frame = load_candidate_training_frame(dataset_path)

    assert frame.shape == (2, len(CANDIDATE_ROW_HEADERS))
    assert frame["day_id"].tolist() == ["20250324", "20250325"]
    assert frame["segment_type"].tolist() == ["performance", "ceremony"]


def test_load_training_data_bundle_joins_split_manifest_and_selects_train_validation_rows(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
            _candidate_row(day_id="20250326", segment_type="warmup", boundary="0"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
            {"day_id": "20250326", "split_name": "test"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_plus_thumbnail",
    )

    assert bundle.train_rows["day_id"].tolist() == ["20250324"]
    assert bundle.validation_rows["day_id"].tolist() == ["20250325"]
    assert bundle.test_rows["day_id"].tolist() == ["20250326"]
    assert bundle.split_counts_by_name == {"train": 1, "validation": 1, "test": 1}
    assert bundle.segment_type.train_data["segment_type"].tolist() == ["performance"]
    assert bundle.boundary.validation_data["boundary"].tolist() == [1]
    assert bundle.boundary.test_data["boundary"].tolist() == [0]
    assert bundle.image_feature_columns == [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
        "frame_05_thumb_path",
    ]
    assert "gap_34" in bundle.shared_feature_columns
    assert "candidate_id" not in bundle.shared_feature_columns
    assert "candidate_rule_name" not in bundle.shared_feature_columns
    assert "frame_01_relpath" not in bundle.shared_feature_columns
    assert "frame_01_timestamp" not in bundle.shared_feature_columns
    assert "frame_01_preview_path" not in bundle.shared_feature_columns


def test_load_training_data_bundle_requires_base_columns_for_tabular_mode(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    dataset_path.write_text("day_id,segment_type\n20250324,performance\n", encoding="utf-8")
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="missing required columns: boundary"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
        )


def test_load_training_data_bundle_requires_thumbnail_columns_for_thumbnail_mode(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    row = _candidate_row(day_id="20250324", segment_type="performance", boundary="0")
    row.pop("frame_05_thumb_path")
    headers = [header for header in CANDIDATE_ROW_HEADERS if header != "frame_05_thumb_path"]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="missing required columns: frame_05_thumb_path"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_plus_thumbnail",
        )


def test_load_training_data_bundle_rejects_missing_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    with pytest.raises(ValueError, match="segment_type label must not be blank"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
        )


def test_load_training_data_bundle_requires_split_manifest_coverage(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="split manifest is missing day_id entries: 20250325"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
        )
