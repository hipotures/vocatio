import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_features import (
    CANONICAL_MISSING,
    CANONICAL_COSTUME_TYPE_VOCABULARY,
    aggregate_window_descriptors,
    build_candidate_feature_row,
    cosine_distance,
    normalize_descriptor_value,
)


def test_cosine_distance_uses_l2_normalized_embeddings() -> None:
    a = [3.0, 0.0]
    b = [0.0, 5.0]
    c = [10.0, 0.0]

    assert cosine_distance(a, b) == 1.0
    assert cosine_distance(a, c) == 0.0


def test_cosine_distance_rejects_invalid_components_and_shapes() -> None:
    invalid_cases = [
        ([True, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [False, 0.0]),
        ([math.nan, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [math.inf, 0.0]),
    ]

    for left, right in invalid_cases:
        with pytest.raises(ValueError):
            cosine_distance(left, right)

    with pytest.raises(ValueError, match="non-zero norm"):
        cosine_distance([0.0, 0.0], [1.0, 0.0])

    with pytest.raises(ValueError, match="shapes must match"):
        cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0])


def test_normalize_descriptor_value_preserves_explicit_missing() -> None:
    assert normalize_descriptor_value(None) == CANONICAL_MISSING
    assert normalize_descriptor_value("") == CANONICAL_MISSING
    assert normalize_descriptor_value("   ") == CANONICAL_MISSING
    assert normalize_descriptor_value("TuTu", allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY) == "tutu"


def test_normalize_descriptor_value_rejects_invalid_types_and_out_of_vocabulary_values() -> None:
    with pytest.raises(ValueError, match="strings or null"):
        normalize_descriptor_value(True, allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY)

    with pytest.raises(ValueError, match="strings or null"):
        normalize_descriptor_value(1, allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY)

    with pytest.raises(ValueError, match="canonical vocabulary"):
        normalize_descriptor_value("cape", allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY)


def test_aggregate_window_descriptors_uses_majority_vote_and_tie_break() -> None:
    assert (
        aggregate_window_descriptors(
            ["dress", "tutu", "tutu"],
            tie_break_value="dress",
            allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
        )
        == "tutu"
    )
    assert (
        aggregate_window_descriptors(
            ["dress", "tutu"],
            tie_break_value="dress",
            allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
        )
        == "dress"
    )
    assert aggregate_window_descriptors([], tie_break_value="dress") == CANONICAL_MISSING


def test_build_candidate_feature_row_computes_ordered_gap_features() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 12.0,
        "frame_05_timestamp": 13.0,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
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
    assert row["costume_type_left_value"] == CANONICAL_MISSING
    assert row["costume_type_right_value"] == CANONICAL_MISSING
    assert row["costume_type_changed"] == 0
    assert row["costume_type_left_missing"] == 1
    assert row["costume_type_right_missing"] == 1
    assert row["costume_type_left_consistency"] == 1.0
    assert row["costume_type_right_consistency"] == 1.0


def test_build_candidate_feature_row_uses_non_central_gap_median_for_outlier_flag() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 6.0,
        "frame_05_timestamp": 16.0,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=None)

    assert row["gap_12"] == 1.0
    assert row["gap_23"] == 1.0
    assert row["gap_34"] == 4.0
    assert row["gap_45"] == 10.0
    assert row["gap_is_local_outlier"] == 1


def test_build_candidate_feature_row_computes_descriptor_features() -> None:
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
    descriptors = {
        "p1": {"costume_type": "Dress"},
        "p2": {"costume_type": "TUTU"},
        "p3": {"costume_type": "dress"},
        "p4": {"costume_type": "jacket"},
        "p5": {"costume_type": "jacket"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["costume_type_left_value"] == "dress"
    assert row["costume_type_right_value"] == "jacket"
    assert row["costume_type_changed"] == 1
    assert row["costume_type_left_missing"] == 0
    assert row["costume_type_right_missing"] == 0
    assert row["costume_type_left_consistency"] == pytest.approx(2.0 / 3.0)
    assert row["costume_type_right_consistency"] == 1.0


def test_build_candidate_feature_row_uses_frame_position_tie_break_for_descriptors() -> None:
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
    descriptors = {
        "p1": {"costume_type": "dress"},
        "p2": {"costume_type": "tutu"},
        "p3": {"costume_type": "coat"},
        "p4": {"costume_type": "jacket"},
        "p5": {"costume_type": "skirt"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["costume_type_left_value"] == "coat"
    assert row["costume_type_right_value"] == "jacket"
    assert row["costume_type_left_consistency"] == pytest.approx(1.0 / 3.0)
    assert row["costume_type_right_consistency"] == pytest.approx(0.5)


def test_build_candidate_feature_row_preserves_missingness_in_descriptor_features() -> None:
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
    descriptors = {
        "p1": {"costume_type": "dress"},
        "p2": {},
        "p3": {"costume_type": "   "},
        "p4": {"costume_type": None},
        "p5": {"costume_type": "coat"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["costume_type_left_value"] == CANONICAL_MISSING
    assert row["costume_type_right_value"] == CANONICAL_MISSING
    assert row["costume_type_changed"] == 0
    assert row["costume_type_left_missing"] == 1
    assert row["costume_type_right_missing"] == 1
    assert row["costume_type_left_consistency"] == pytest.approx(2.0 / 3.0)
    assert row["costume_type_right_consistency"] == pytest.approx(0.5)


def test_build_candidate_feature_row_rejects_malformed_descriptor_record_shape() -> None:
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
    descriptors = {
        "p1": {"costume_type": "dress"},
        "p2": "not-a-mapping",
    }

    with pytest.raises(ValueError, match="must be a mapping"):
        build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)


def test_build_candidate_feature_row_rejects_out_of_vocabulary_descriptor_values() -> None:
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
    descriptors = {
        "p1": {"costume_type": "dress"},
        "p2": {"costume_type": "cape"},
    }

    with pytest.raises(ValueError, match="canonical vocabulary"):
        build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)


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


def test_build_candidate_feature_row_uses_non_central_embedding_median_for_cross_boundary_score() -> None:
    angle_36 = math.radians(36.86989764584401)
    angle_42 = math.radians(41.5930222125875)
    angle_132 = math.radians(131.5930222125875)
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
        "p1": [math.cos(angle_36), math.sin(angle_36)],
        "p2": [1.0, 0.0],
        "p3": [math.cos(-angle_36), math.sin(-angle_36)],
        "p4": [math.cos(angle_42), math.sin(angle_42)],
        "p5": [math.cos(angle_132), math.sin(angle_132)],
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=embeddings)

    assert row["embed_dist_12"] == pytest.approx(0.2, abs=1e-5)
    assert row["embed_dist_23"] == pytest.approx(0.2, abs=1e-5)
    assert row["embed_dist_34"] == pytest.approx(0.8, abs=1e-5)
    assert row["embed_dist_45"] == pytest.approx(1.0, abs=1e-5)
    assert row["cross_boundary_outlier_score"] == pytest.approx(4.0, abs=2e-5)


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


def test_build_candidate_feature_row_rejects_non_finite_timestamps() -> None:
    base_candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 3.0,
        "frame_05_timestamp": 4.0,
    }

    for bad_value in (math.nan, math.inf, -math.inf):
        candidate = dict(base_candidate)
        candidate["frame_03_timestamp"] = bad_value
        with pytest.raises(ValueError, match="frame_03_timestamp"):
            build_candidate_feature_row(candidate, descriptors={}, embeddings=None)


def test_build_candidate_feature_row_rejects_unordered_timestamps() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 1.5,
        "frame_05_timestamp": 4.0,
    }

    with pytest.raises(ValueError, match="non-decreasing"):
        build_candidate_feature_row(candidate, descriptors={}, embeddings=None)
