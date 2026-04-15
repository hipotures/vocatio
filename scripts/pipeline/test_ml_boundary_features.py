import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_features import build_candidate_feature_row, cosine_distance


def test_cosine_distance_uses_l2_normalized_embeddings() -> None:
    a = [3.0, 0.0]
    b = [0.0, 5.0]
    c = [10.0, 0.0]

    assert cosine_distance(a, b) == 1.0
    assert cosine_distance(a, c) == 0.0


def test_build_candidate_feature_row_computes_ordered_gap_features() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 12.0,
        "frame_05_timestamp": 13.0,
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=None)

    assert row["gap_12"] == 1.0
    assert row["gap_23"] == 1.0
    assert row["gap_34"] == 10.0
    assert row["gap_45"] == 1.0
    assert row["center_gap_seconds"] == 10.0
    assert row["left_internal_gap_mean"] == 1.0
    assert row["right_internal_gap_mean"] == 1.0
    assert row["local_gap_median"] == 1.0
    assert row["gap_ratio"] == 10.0
    assert row["gap_is_local_outlier"] == 1
    assert row["max_gap_in_window"] == 10.0
    assert row["gap_variance"] == 15.1875


def test_build_candidate_feature_row_computes_embedding_features() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 3.0,
        "frame_05_timestamp": 4.0,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    embeddings = {
        "p1": [1.0, 0.0],
        "p2": [0.0, 1.0],
        "p3": [0.0, 1.0],
        "p4": [1.0, 0.0],
        "p5": [0.0, 1.0],
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=embeddings)

    assert row["embed_dist_12"] == 1.0
    assert row["embed_dist_23"] == 0.0
    assert row["embed_dist_34"] == 1.0
    assert row["embed_dist_45"] == 1.0
    assert row["left_consistency_score"] == 0.5
    assert row["right_consistency_score"] == 1.0
    assert row["cross_boundary_outlier_score"] == 1.0


def test_build_candidate_feature_row_handles_zero_non_central_median() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.0,
        "frame_03_timestamp": 0.0,
        "frame_04_timestamp": 5.0,
        "frame_05_timestamp": 5.0,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    embeddings = {
        "p1": [1.0, 0.0],
        "p2": [1.0, 0.0],
        "p3": [1.0, 0.0],
        "p4": [0.0, 1.0],
        "p5": [0.0, 1.0],
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=embeddings)

    assert row["gap_ratio"] == math.inf
    assert row["gap_is_local_outlier"] == 1
    assert row["cross_boundary_outlier_score"] == math.inf
