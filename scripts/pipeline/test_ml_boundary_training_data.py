import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from bootstrap_photo_boundaries import PHOTO_BOUNDARY_SCORE_HEADERS
from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.ml_boundary_features import CANONICAL_MISSING
from lib.ml_boundary_training_data import (
    load_candidate_training_frame,
    load_training_data_bundle,
)


def _candidate_row(
    *,
    day_id: str,
    right_segment_type: str = "ceremony",
    boundary: str = "1",
    candidate_rule_version: str = "gap-v1",
) -> dict[str, str]:
    row = {header: "" for header in CANDIDATE_ROW_HEADERS}
    row.update(
        {
            "candidate_id": canonical_candidate_id(
                day_id=day_id,
                center_left_photo_id=f"{day_id}-p2",
                center_right_photo_id=f"{day_id}-p3",
                candidate_rule_version=candidate_rule_version,
            ),
            "day_id": day_id,
            "window_radius": "2",
            "center_left_photo_id": f"{day_id}-p2",
            "center_right_photo_id": f"{day_id}-p3",
            "left_segment_id": f"{day_id}-seg-a",
            "right_segment_id": f"{day_id}-seg-b",
            "left_segment_type": "performance",
            "right_segment_type": right_segment_type or "ceremony",
            "boundary": boundary,
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": candidate_rule_version,
            "candidate_rule_params_json": "{\"gap_threshold_seconds\":20.0}",
            "descriptor_schema_version": "not_included_v1",
            "split_name": "",
            "window_photo_ids": "[\"p1\",\"p2\",\"p3\",\"p4\"]",
            "window_relative_paths": "[\"cam/p1.jpg\",\"cam/p2.jpg\",\"cam/p3.jpg\",\"cam/p4.jpg\"]",
        }
    )
    for frame_index in range(1, 5):
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
    fieldnames = list(rows[0].keys()) if rows else ["day_id", "split_name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_boundary_scores_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHOTO_BOUNDARY_SCORE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _shift_candidate_window(row: dict[str, str], *, day_id: str, first_photo_index: int) -> dict[str, str]:
    shifted_row = dict(row)
    window_photo_ids = [f"{day_id}-p{photo_index}" for photo_index in range(first_photo_index, first_photo_index + 4)]
    window_relative_paths = [f"cam/{photo_id}.jpg" for photo_id in window_photo_ids]
    for frame_index, photo_index in enumerate(range(first_photo_index, first_photo_index + 4), start=1):
        suffix = f"{frame_index:02d}"
        photo_id = f"{day_id}-p{photo_index}"
        shifted_row[f"frame_{suffix}_photo_id"] = photo_id
        shifted_row[f"frame_{suffix}_relpath"] = f"cam/{photo_id}.jpg"
        shifted_row[f"frame_{suffix}_timestamp"] = str(float(photo_index))
        shifted_row[f"frame_{suffix}_thumb_path"] = f"thumb/{photo_id}.jpg"
        shifted_row[f"frame_{suffix}_preview_path"] = f"preview/{photo_id}.jpg"
    shifted_row["center_left_photo_id"] = window_photo_ids[1]
    shifted_row["center_right_photo_id"] = window_photo_ids[2]
    shifted_row["candidate_id"] = canonical_candidate_id(
        day_id=day_id,
        center_left_photo_id=window_photo_ids[1],
        center_right_photo_id=window_photo_ids[2],
        candidate_rule_version=str(shifted_row["candidate_rule_version"]),
    )
    shifted_row["window_photo_ids"] = json.dumps([photo_id.split("-", 1)[1] for photo_id in window_photo_ids])
    shifted_row["window_relative_paths"] = json.dumps(window_relative_paths)
    return shifted_row


def _write_annotation_payload(
    annotation_dir: Path,
    *,
    relative_path: str,
    data: dict[str, object],
) -> None:
    annotation_path = annotation_dir / f"{relative_path}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "photo_pre_model_v1",
                "relative_path": relative_path,
                "generated_at": "2026-04-15T12:00:00+02:00",
                "model": "test-model",
                "data": data,
            }
        ),
        encoding="utf-8",
    )


def test_load_candidate_training_frame_reads_csv_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
        ],
    )

    frame = load_candidate_training_frame(dataset_path)

    assert frame.shape == (2, len(CANDIDATE_ROW_HEADERS))
    assert frame["day_id"].tolist() == ["20250324", "20250325"]
    assert frame["right_segment_type"].tolist() == ["performance", "ceremony"]


def test_load_training_data_bundle_joins_split_manifest_and_selects_train_validation_rows(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
            _candidate_row(day_id="20250326", right_segment_type="warmup", boundary="0"),
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

    assert bundle.window_radius == 2
    assert bundle.train_rows["day_id"].tolist() == ["20250324"]
    assert bundle.validation_rows["day_id"].tolist() == ["20250325"]
    assert bundle.test_rows["day_id"].tolist() == ["20250326"]
    assert bundle.split_counts_by_name == {"train": 1, "validation": 1, "test": 1}
    assert bundle.right_segment_type.train_data["right_segment_type"].tolist() == ["performance"]
    assert bundle.boundary.validation_data["boundary"].tolist() == [1]
    assert bundle.boundary.test_data["boundary"].tolist() == [0]
    assert bundle.image_feature_columns == [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
    ]
    assert "gap_23" in bundle.shared_feature_columns
    assert "candidate_id" not in bundle.shared_feature_columns
    assert "candidate_rule_name" not in bundle.shared_feature_columns
    assert "frame_01_relpath" not in bundle.shared_feature_columns
    assert "frame_01_timestamp" not in bundle.shared_feature_columns
    assert "frame_01_preview_path" not in bundle.shared_feature_columns


def test_load_training_data_bundle_exposes_left_right_and_boundary_predictors(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    assert bundle.left_segment_type.label_column == "left_segment_type"
    assert bundle.right_segment_type.label_column == "right_segment_type"
    assert bundle.boundary.label_column == "boundary"


def test_feature_columns_manifest_uses_left_and_right_keys(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    columns = bundle.feature_columns_by_mode["tabular_only"]
    assert "left_segment_type_feature_columns" in columns
    assert "right_segment_type_feature_columns" in columns
    assert "segment_type_feature_columns" not in columns


def test_predictor_feature_sets_match_for_left_and_right(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    assert bundle.left_segment_type.feature_columns == bundle.right_segment_type.feature_columns


def test_load_training_data_bundle_accepts_candidate_level_split_manifest(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    candidate_rows = [
        _candidate_row(
            day_id="20250325",
            right_segment_type="performance",
            boundary="0",
            candidate_rule_version="gap-v1",
        ),
        _candidate_row(
            day_id="20250325",
            right_segment_type="ceremony",
            boundary="1",
            candidate_rule_version="gap-v2",
        ),
        _candidate_row(
            day_id="20250325",
            right_segment_type="warmup",
            boundary="0",
            candidate_rule_version="gap-v3",
        ),
    ]
    _write_candidate_csv(dataset_path, candidate_rows)
    _write_split_manifest(
        split_manifest_path,
        [
            {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
            {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "validation"},
            {"candidate_id": candidate_rows[2]["candidate_id"], "split_name": "test"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    assert bundle.train_rows["right_segment_type"].tolist() == ["performance"]
    assert bundle.validation_rows["right_segment_type"].tolist() == ["ceremony"]
    assert bundle.test_rows["right_segment_type"].tolist() == ["warmup"]
    assert bundle.split_counts_by_name == {"train": 1, "validation": 1, "test": 1}


def test_load_training_data_bundle_rejects_ambiguous_split_manifest_keys(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    candidate_rows = [
        _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
        _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
    ]
    _write_candidate_csv(dataset_path, candidate_rows)
    _write_split_manifest(
        split_manifest_path,
        [
            {
                "day_id": "20250324",
                "candidate_id": candidate_rows[0]["candidate_id"],
                "split_name": "train",
            },
            {
                "day_id": "20250325",
                "candidate_id": candidate_rows[1]["candidate_id"],
                "split_name": "validation",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="split manifest must not contain both day_id and candidate_id columns",
    ):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
        )


def test_load_training_data_bundle_requires_base_columns_for_tabular_mode(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    dataset_path.write_text(
        "day_id,left_segment_type,right_segment_type\n20250324,performance,ceremony\n",
        encoding="utf-8",
    )
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="missing required columns: boundary, window_radius"):
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
    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0")
    row.pop("frame_04_thumb_path")
    headers = [header for header in CANDIDATE_ROW_HEADERS if header != "frame_04_thumb_path"]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="missing required columns: frame_04_thumb_path"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_plus_thumbnail",
        )


def test_load_training_data_bundle_rejects_extra_frame_columns_for_declared_radius(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0")
    row["frame_05_photo_id"] = "20250324-p5"
    row["frame_05_relpath"] = "cam/20250324-p5.jpg"
    row["frame_05_timestamp"] = "5.0"
    row["frame_05_thumb_path"] = "thumb/20250324-p5.jpg"
    row["frame_05_preview_path"] = "preview/20250324-p5.jpg"
    fieldnames = list(CANDIDATE_ROW_HEADERS) + [
        "frame_05_photo_id",
        "frame_05_relpath",
        "frame_05_timestamp",
        "frame_05_thumb_path",
        "frame_05_preview_path",
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match="unexpected columns are not allowed: frame_05_"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_plus_thumbnail",
        )


def test_load_training_data_bundle_requires_relpath_columns_for_heuristic_joins(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0")
    missing_column = "frame_04_relpath"
    row.pop(missing_column)
    headers = [header for header in CANDIDATE_ROW_HEADERS if header != missing_column]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )

    with pytest.raises(ValueError, match=f"missing required columns: {missing_column}"):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
            require_train_validation=False,
        )


def test_training_bundle_reads_window_radius_from_candidates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    assert bundle.window_radius == 2


def test_load_training_data_bundle_rejects_missing_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    missing_label_row = _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1")
    missing_label_row["right_segment_type"] = ""
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            missing_label_row,
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )

    with pytest.raises(ValueError, match="right_segment_type label must not be blank"):
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
            _candidate_row(day_id="20250324", right_segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", right_segment_type="ceremony", boundary="1"),
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


def test_load_training_data_bundle_reports_missing_annotation_counts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    annotation_dir = tmp_path / "photo_pre_model_annotations"
    annotation_dir.mkdir()

    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="1")
    _write_candidate_csv(dataset_path, [row])
    _write_split_manifest(
        split_manifest_path,
        [{"candidate_id": row["candidate_id"], "split_name": "train"}],
    )

    _write_annotation_payload(
        annotation_dir,
        relative_path="cam/20250324-p1.jpg",
        data={
            "people_count": "solo",
            "performer_view": "solo",
            "upper_garment": "top",
            "lower_garment": "skirt",
            "sleeves": "short",
            "leg_coverage": "bare",
            "dominant_colors": ["white", "purple"],
            "headwear": "none",
            "footwear": "ballet_shoes",
            "props": ["none"],
            "dance_style_hint": "ballet",
        },
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
        require_train_validation=False,
        annotation_dir=annotation_dir,
    )

    assert bundle.missing_annotation_photo_count == 3
    assert bundle.missing_annotation_candidate_count == 1


def test_load_training_data_bundle_joins_heuristic_boundary_scores_and_counts_missing_pairs(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    corpus_dir = workspace_dir / "ml_boundary_corpus"
    dataset_path = corpus_dir / "ml_boundary_candidates.corpus.csv"
    split_manifest_path = corpus_dir / "ml_boundary_splits.csv"
    boundary_scores_path = workspace_dir / "photo_boundary_scores.csv"
    corpus_dir.mkdir(parents=True)

    train_row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="1")
    validation_row = _shift_candidate_window(
        _candidate_row(
            day_id="20250324",
            right_segment_type="ceremony",
            boundary="0",
            candidate_rule_version="gap-v2",
        ),
        day_id="20250324",
        first_photo_index=3,
    )
    _write_candidate_csv(dataset_path, [train_row, validation_row])
    _write_split_manifest(
        split_manifest_path,
        [
            {"candidate_id": train_row["candidate_id"], "split_name": "train"},
            {"candidate_id": validation_row["candidate_id"], "split_name": "validation"},
        ],
    )
    _write_boundary_scores_csv(
        boundary_scores_path,
        [
            {
                "left_relative_path": "cam/20250324-p3.jpg",
                "right_relative_path": "cam/20250324-p4.jpg",
                "left_start_local": "2025-03-24T10:00:03",
                "right_start_local": "2025-03-24T10:00:04",
                "left_start_epoch_ms": "3000",
                "right_start_epoch_ms": "4000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.103000",
                "distance_zscore": "0.203000",
                "smoothed_distance_zscore": "0.303000",
                "time_gap_boost": "0.200000",
                "boundary_score": "0.123000",
                "boundary_label": "hard",
                "boundary_reason": "center-cut",
                "model_source": "bootstrap_heuristic",
            },
            {
                "left_relative_path": "cam/20250324-p1.jpg",
                "right_relative_path": "cam/20250324-p2.jpg",
                "left_start_local": "2025-03-24T10:00:01",
                "right_start_local": "2025-03-24T10:00:02",
                "left_start_epoch_ms": "1000",
                "right_start_epoch_ms": "2000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.101000",
                "distance_zscore": "0.201000",
                "smoothed_distance_zscore": "0.301000",
                "time_gap_boost": "0.000000",
                "boundary_score": "0.401000",
                "boundary_label": "none",
                "boundary_reason": "baseline",
                "model_source": "bootstrap_heuristic",
            },
            {
                "left_relative_path": "cam/20250324-p4.jpg",
                "right_relative_path": "cam/20250324-p5.jpg",
                "left_start_local": "2025-03-24T10:00:04",
                "right_start_local": "2025-03-24T10:00:05",
                "left_start_epoch_ms": "4000",
                "right_start_epoch_ms": "5000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.104000",
                "distance_zscore": "0.204000",
                "smoothed_distance_zscore": "0.304000",
                "time_gap_boost": "0.300000",
                "boundary_score": "0.404000",
                "boundary_label": "none",
                "boundary_reason": "settled",
                "model_source": "bootstrap_heuristic",
            },
            {
                "left_relative_path": "cam/20250324-p2.jpg",
                "right_relative_path": "cam/20250324-p3.jpg",
                "left_start_local": "2025-03-24T10:00:02",
                "right_start_local": "2025-03-24T10:00:03",
                "left_start_epoch_ms": "2000",
                "right_start_epoch_ms": "3000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.102000",
                "distance_zscore": "0.202000",
                "smoothed_distance_zscore": "0.302000",
                "time_gap_boost": "0.100000",
                "boundary_score": "0.222000",
                "boundary_label": "soft",
                "boundary_reason": "lifted",
                "model_source": "bootstrap_heuristic",
            },
        ],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
    )

    assert "heuristic_dino_dist_12" in bundle.shared_feature_columns
    assert "heuristic_boundary_score_23" in bundle.shared_feature_columns
    assert "heuristic_boundary_label_34" in bundle.shared_feature_columns
    assert "heuristic_dino_dist_12" in bundle.left_segment_type.feature_columns
    assert "heuristic_boundary_score_23" in bundle.boundary.feature_columns
    assert bundle.train_rows["heuristic_dino_dist_12"].tolist() == [0.101]
    assert bundle.train_rows["heuristic_boundary_score_23"].tolist() == [0.222]
    assert bundle.train_rows["heuristic_smoothed_distance_zscore_34"].tolist() == [0.303]
    assert bundle.train_rows["heuristic_time_gap_boost_34"].tolist() == [0.2]
    assert bundle.train_rows["heuristic_boundary_label_34"].tolist() == ["hard"]
    assert math.isnan(bundle.validation_rows["heuristic_dino_dist_34"].tolist()[0])
    assert math.isnan(bundle.validation_rows["heuristic_boundary_score_34"].tolist()[0])
    assert bundle.validation_rows["heuristic_boundary_label_34"].tolist() == [CANONICAL_MISSING]
    assert bundle.heuristic_scores_path == boundary_scores_path
    assert bundle.total_heuristic_pair_count == 6
    assert bundle.missing_heuristic_pair_count == 1
    assert bundle.total_heuristic_candidate_count == 2
    assert bundle.missing_heuristic_candidate_count == 1


def test_load_training_data_bundle_rejects_duplicate_heuristic_pair_rows(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    boundary_scores_path = tmp_path / "photo_boundary_scores.csv"

    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="1")
    _write_candidate_csv(dataset_path, [row])
    _write_split_manifest(
        split_manifest_path,
        [{"candidate_id": row["candidate_id"], "split_name": "train"}],
    )
    _write_boundary_scores_csv(
        boundary_scores_path,
        [
            {
                "left_relative_path": "cam/20250324-p1.jpg",
                "right_relative_path": "cam/20250324-p2.jpg",
                "left_start_local": "2025-03-24T10:00:01",
                "right_start_local": "2025-03-24T10:00:02",
                "left_start_epoch_ms": "1000",
                "right_start_epoch_ms": "2000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.101000",
                "distance_zscore": "0.201000",
                "smoothed_distance_zscore": "0.301000",
                "time_gap_boost": "0.000000",
                "boundary_score": "0.401000",
                "boundary_label": "none",
                "boundary_reason": "baseline",
                "model_source": "bootstrap_heuristic",
            },
            {
                "left_relative_path": "cam/20250324-p1.jpg",
                "right_relative_path": "cam/20250324-p2.jpg",
                "left_start_local": "2025-03-24T10:00:01",
                "right_start_local": "2025-03-24T10:00:02",
                "left_start_epoch_ms": "1000",
                "right_start_epoch_ms": "2000",
                "time_gap_seconds": "1.000000",
                "dino_cosine_distance": "0.999000",
                "distance_zscore": "0.999000",
                "smoothed_distance_zscore": "0.999000",
                "time_gap_boost": "0.999000",
                "boundary_score": "0.999000",
                "boundary_label": "soft",
                "boundary_reason": "duplicate",
                "model_source": "bootstrap_heuristic",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="photo_boundary_scores.csv contains duplicate adjacent pair rows for cam/20250324-p1.jpg -> cam/20250324-p2.jpg",
    ):
        load_training_data_bundle(
            dataset_path,
            split_manifest_path=split_manifest_path,
            mode="tabular_only",
            require_train_validation=False,
        )


def test_load_training_data_bundle_extends_descriptor_registry_from_dataset_annotations(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    annotation_dir = tmp_path / "photo_pre_model_annotations"

    train_row = _candidate_row(
        day_id="20250324",
        right_segment_type="performance",
        boundary="0",
        candidate_rule_version="gap-v1",
    )
    validation_row = _candidate_row(
        day_id="20250325",
        right_segment_type="ceremony",
        boundary="1",
        candidate_rule_version="gap-v2",
    )
    _write_candidate_csv(dataset_path, [train_row, validation_row])
    _write_split_manifest(
        split_manifest_path,
        [
            {"candidate_id": train_row["candidate_id"], "split_name": "train"},
            {"candidate_id": validation_row["candidate_id"], "split_name": "validation"},
        ],
    )

    default_payload = {
        "people_count": "solo",
        "performer_view": "solo",
        "upper_garment": "top",
        "lower_garment": "skirt",
        "sleeves": "short",
        "leg_coverage": "bare",
        "dominant_colors": ["white"],
        "headwear": "none",
        "footwear": "ballet_shoes",
        "props": ["none"],
        "dance_style_hint": "ballet",
    }
    for day_id in ("20250324", "20250325"):
        for frame_index in range(1, 5):
            relative_path = f"cam/{day_id}-p{frame_index}.jpg"
            payload = dict(default_payload)
            if day_id == "20250324":
                payload["appearance"] = {"silhouette": "clean-line"}
            if day_id == "20250325" and frame_index == 4:
                payload["appearance"] = {"accents": ["Gold", "Silver"]}
            elif day_id == "20250325" and frame_index == 5:
                payload["appearance"] = {"accents": "blue/white"}
            _write_annotation_payload(
                annotation_dir,
                relative_path=relative_path,
                data=payload,
            )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
        annotation_dir=annotation_dir,
    )

    assert "left_upper_garment" in bundle.shared_feature_columns
    assert "left_appearance_silhouette" in bundle.shared_feature_columns
    assert "right_appearance_accents_01" in bundle.shared_feature_columns
    assert "right_appearance_accents_02" in bundle.shared_feature_columns
    assert "right_appearance_accents_03" in bundle.shared_feature_columns
    assert bundle.train_rows["left_appearance_silhouette"].tolist() == ["clean-line"]
    assert bundle.train_rows["right_appearance_accents_01"].tolist() == [CANONICAL_MISSING]
    assert bundle.validation_rows["left_appearance_silhouette"].tolist() == [CANONICAL_MISSING]
    assert bundle.validation_rows["right_appearance_accents_01"].tolist() == ["gold"]
    assert bundle.validation_rows["right_appearance_accents_02"].tolist() == ["silver"]
    assert bundle.validation_rows["right_appearance_accents_03"].tolist() == [CANONICAL_MISSING]


def test_load_training_data_bundle_resolves_default_annotation_dir_for_corpus_dataset(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    corpus_dir = workspace_dir / "ml_boundary_corpus"
    dataset_path = corpus_dir / "ml_boundary_candidates.corpus.csv"
    split_manifest_path = corpus_dir / "ml_boundary_splits.csv"
    annotation_dir = workspace_dir / "photo_pre_model_annotations"
    corpus_dir.mkdir(parents=True)

    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="1")
    _write_candidate_csv(dataset_path, [row])
    _write_split_manifest(
        split_manifest_path,
        [{"candidate_id": row["candidate_id"], "split_name": "train"}],
    )
    default_payload = {
        "people_count": "solo",
        "performer_view": "solo",
        "upper_garment": "top",
        "lower_garment": "skirt",
        "sleeves": "short",
        "leg_coverage": "bare",
        "dominant_colors": ["white"],
        "headwear": "none",
        "footwear": "ballet_shoes",
        "props": ["none"],
        "dance_style_hint": "ballet",
    }
    for frame_index in range(1, 4):
        _write_annotation_payload(
            annotation_dir,
            relative_path=f"cam/20250324-p{frame_index}.jpg",
            data=default_payload,
        )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
        require_train_validation=False,
    )

    assert bundle.annotation_dir == annotation_dir
    assert bundle.train_rows["left_upper_garment"].tolist() == ["top"]
    assert bundle.missing_annotation_photo_count == 1


def test_load_training_data_bundle_skips_missing_counts_when_default_annotation_dir_is_absent(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    corpus_dir = workspace_dir / "ml_boundary_corpus"
    dataset_path = corpus_dir / "ml_boundary_candidates.corpus.csv"
    split_manifest_path = corpus_dir / "ml_boundary_splits.csv"
    corpus_dir.mkdir(parents=True)

    row = _candidate_row(day_id="20250324", right_segment_type="performance", boundary="1")
    _write_candidate_csv(dataset_path, [row])
    _write_split_manifest(
        split_manifest_path,
        [{"candidate_id": row["candidate_id"], "split_name": "train"}],
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
        require_train_validation=False,
    )

    assert bundle.annotation_dir is None
    assert bundle.heuristic_scores_path is None
    assert bundle.missing_annotation_photo_count == 0
    assert bundle.missing_annotation_candidate_count == 0
    assert bundle.total_heuristic_pair_count == 3
    assert bundle.missing_heuristic_pair_count == 3
    assert bundle.total_heuristic_candidate_count == 1
    assert bundle.missing_heuristic_candidate_count == 1
