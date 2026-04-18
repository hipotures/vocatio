import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_features import (
    CANONICAL_MISSING,
    build_candidate_feature_row,
    cosine_distance,
    normalize_descriptor_value,
)
from lib.photo_pre_model_annotations import (
    build_photo_pre_model_descriptor_field_registry,
    load_photo_pre_model_data_by_relative_path,
)


def _default_descriptor_feature_columns() -> set[str]:
    registry = build_photo_pre_model_descriptor_field_registry()
    columns: set[str] = set()
    for side_name in ("left", "right"):
        for field_name, field_kind in registry.items():
            if field_kind == "multivalue":
                columns.update(
                    f"{side_name}_{field_name}_{index:02d}"
                    for index in range(1, 6)
                )
                continue
            columns.add(f"{side_name}_{field_name}")
    return columns


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


def test_normalize_descriptor_value_enforces_allowed_values() -> None:
    allowed_values = frozenset({"top", "jacket"})

    assert normalize_descriptor_value(None, allowed_values=allowed_values) == CANONICAL_MISSING
    assert normalize_descriptor_value("  TOP  ", allowed_values=allowed_values) == "top"

    with pytest.raises(ValueError, match="strings or null"):
        normalize_descriptor_value(True, allowed_values=allowed_values)

    with pytest.raises(ValueError, match="canonical vocabulary"):
        normalize_descriptor_value("cape", allowed_values=allowed_values)


def test_load_photo_pre_model_data_by_relative_path_reads_payload_data_only(tmp_path: Path) -> None:
    output_path = tmp_path / "cam" / "a.hif.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "photo_pre_model_v1",
                "relative_path": "cam/a.hif",
                "generated_at": "2026-04-17T12:00:00Z",
                "model": "test-model",
                "data": {
                    "upper_garment": "top",
                    "dominant_colors": ["white", "purple"],
                },
            }
        ),
        encoding="utf-8",
    )

    annotations = load_photo_pre_model_data_by_relative_path(
        tmp_path,
        ["cam/a.hif", "cam/missing.hif"],
    )

    assert annotations == {
        "cam/a.hif": {
            "people_count": "no_visible_people",
            "upper_garment": "top",
            "dominant_colors": ["white", "purple"],
        }
    }


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


def test_build_candidate_feature_row_includes_pairwise_heuristic_features() -> None:
    candidate = {
        "frame_01_timestamp": "0",
        "frame_02_timestamp": "5",
        "frame_03_timestamp": "10",
        "frame_04_timestamp": "40",
        "frame_05_timestamp": "45",
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    heuristic_features = {
        "12": {
            "dino_cosine_distance": 0.101,
            "boundary_score": 0.202,
            "distance_zscore": 0.303,
            "smoothed_distance_zscore": 0.404,
            "time_gap_boost": 0.0,
            "boundary_label": "none",
        },
        "23": {
            "dino_cosine_distance": 0.111,
            "boundary_score": 0.222,
            "distance_zscore": 0.333,
            "smoothed_distance_zscore": 0.444,
            "time_gap_boost": 0.5,
            "boundary_label": "soft",
        },
        "45": {
            "dino_cosine_distance": 0.123,
            "boundary_score": 0.234,
            "distance_zscore": -1.0,
            "smoothed_distance_zscore": 0.456,
            "time_gap_boost": 0.0,
            "boundary_label": "none",
        },
    }

    row = build_candidate_feature_row(
        candidate,
        descriptors={},
        embeddings=None,
        heuristic_features=heuristic_features,
    )

    assert row["heuristic_dino_dist_12"] == 0.101
    assert row["heuristic_dino_dist_23"] == 0.111
    assert row["heuristic_boundary_score_23"] == 0.222
    assert row["heuristic_distance_zscore_23"] == 0.333
    assert row["heuristic_smoothed_distance_zscore_23"] == 0.444
    assert row["heuristic_time_gap_boost_23"] == 0.5
    assert row["heuristic_boundary_label_23"] == "soft"
    assert row["heuristic_dino_dist_45"] == 0.123
    assert row["heuristic_boundary_score_45"] == 0.234
    assert row["heuristic_distance_zscore_45"] == -1.0
    assert row["heuristic_smoothed_distance_zscore_45"] == 0.456
    assert row["heuristic_time_gap_boost_45"] == 0.0
    assert row["heuristic_boundary_label_45"] == "none"
    assert isinstance(row["heuristic_dino_dist_34"], float)
    assert not math.isfinite(row["heuristic_dino_dist_34"])
    assert isinstance(row["heuristic_boundary_score_34"], float)
    assert not math.isfinite(row["heuristic_boundary_score_34"])
    assert isinstance(row["heuristic_distance_zscore_34"], float)
    assert not math.isfinite(row["heuristic_distance_zscore_34"])
    assert isinstance(row["heuristic_smoothed_distance_zscore_34"], float)
    assert not math.isfinite(row["heuristic_smoothed_distance_zscore_34"])
    assert isinstance(row["heuristic_time_gap_boost_34"], float)
    assert not math.isfinite(row["heuristic_time_gap_boost_34"])
    assert row["heuristic_boundary_label_34"] == CANONICAL_MISSING


def test_build_candidate_feature_row_flattens_scalar_descriptor_fields() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"upper_garment": "Top", "lower_garment": "Skirt"},
        "p2": {"upper_garment": "top", "lower_garment": "skirt"},
        "p3": {"upper_garment": "Jacket", "lower_garment": "Skirt"},
        "p4": {"upper_garment": "Top", "lower_garment": "Tutu"},
        "p5": {"upper_garment": "Top", "lower_garment": "Tutu"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_upper_garment"] == "top"
    assert row["right_upper_garment"] == "top"
    assert row["left_lower_garment"] == "skirt"
    assert row["right_lower_garment"] == "tutu"


def test_build_candidate_feature_row_flattens_multivalue_descriptor_fields() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"dominant_colors": ["White", "Purple"]},
        "p2": {"dominant_colors": ["purple"]},
        "p3": {"dominant_colors": ["white", "purple"]},
        "p4": {"dominant_colors": ["Blue", "White"]},
        "p5": {"dominant_colors": ["white"]},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_dominant_colors_01"] == "purple"
    assert row["left_dominant_colors_02"] == "white"
    assert row["left_dominant_colors_03"] == CANONICAL_MISSING
    assert row["left_dominant_colors_04"] == CANONICAL_MISSING
    assert row["left_dominant_colors_05"] == CANONICAL_MISSING
    assert row["right_dominant_colors_01"] == "blue"
    assert row["right_dominant_colors_02"] == "white"
    assert row["right_dominant_colors_03"] == CANONICAL_MISSING
    assert row["right_dominant_colors_04"] == CANONICAL_MISSING
    assert row["right_dominant_colors_05"] == CANONICAL_MISSING


def test_build_candidate_feature_row_splits_text_values_on_list_delimiters_only() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"footwear": "ballet_shoes"},
        "p2": {"footwear": "ballet_shoes"},
        "p3": {"footwear": "ballet_shoes"},
        "p4": {"props": "fan; ribbon"},
        "p5": {"props": "banner/fan"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_footwear"] == "ballet_shoes"
    assert row["right_props_01"] == "banner"
    assert row["right_props_02"] == "fan"
    assert row["right_props_03"] == "ribbon"


def test_build_candidate_feature_row_uses_explicit_registry_for_stable_descriptor_keys() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptor_field_registry = {
        "appearance_upper_garment": "scalar",
        "palette_dominant_colors": "multivalue",
        "props": "multivalue",
    }
    sparse_descriptors = {
        "p1": {"appearance": {"upper_garment": "Top"}},
        "p2": {"appearance": {"upper_garment": "top"}},
        "p3": {"appearance": {"upper_garment": "TOP"}},
        "p4": {"palette": {"dominant_colors": ["Blue"]}},
    }
    empty_descriptors = {}

    sparse_row = build_candidate_feature_row(
        candidate,
        descriptors=sparse_descriptors,
        embeddings=None,
        descriptor_field_registry=descriptor_field_registry,
    )
    empty_row = build_candidate_feature_row(
        candidate,
        descriptors=empty_descriptors,
        embeddings=None,
        descriptor_field_registry=descriptor_field_registry,
    )

    assert set(sparse_row) == set(empty_row)
    assert sparse_row["left_appearance_upper_garment"] == "top"
    assert sparse_row["right_palette_dominant_colors_01"] == "blue"
    assert sparse_row["left_props_01"] == CANONICAL_MISSING
    assert empty_row["left_appearance_upper_garment"] == CANONICAL_MISSING
    assert empty_row["right_palette_dominant_colors_01"] == CANONICAL_MISSING
    assert empty_row["left_props_01"] == CANONICAL_MISSING


def test_build_candidate_feature_row_uses_schema_stable_default_descriptor_keys() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    sparse_descriptors = {
        "p1": {"upper_garment": "Top", "dominant_colors": ["White", "Purple"]},
        "p2": {"upper_garment": "top"},
        "p4": {"props": ["fan"]},
    }
    other_descriptors = {
        "p1": {"footwear": "ballet_shoes"},
        "p4": {"headwear": "hat"},
        "p5": {"dance_style_hint": "jazz"},
    }

    sparse_row = build_candidate_feature_row(candidate, descriptors=sparse_descriptors, embeddings=None)
    other_row = build_candidate_feature_row(candidate, descriptors=other_descriptors, embeddings=None)

    assert set(sparse_row) == set(other_row)
    assert sparse_row["left_upper_garment"] == "top"
    assert sparse_row["left_footwear"] == CANONICAL_MISSING
    assert sparse_row["right_headwear"] == CANONICAL_MISSING
    assert sparse_row["right_dance_style_hint"] == CANONICAL_MISSING


def test_build_candidate_feature_row_default_schema_ignores_extra_flattened_keys() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {
            "upper_garment": "Top",
            "appearance": {"costume": {"silhouette": "Bell"}},
        },
        "p2": {
            "upper_garment": "top",
            "appearance": {"costume": {"silhouette": "bell"}},
        },
        "p3": {
            "upper_garment": "Jacket",
            "appearance": {"costume": {"silhouette": "Cape"}},
        },
        "p4": {
            "metadata": {"shot_type": "Closeup"},
            "props": ["fan"],
            "scene": {"lighting": {"accent_colors": ["Blue", "Gold"]}},
        },
        "p5": {
            "metadata": {"shot_type": "Closeup"},
            "scene": {"lighting": {"accent_colors": ["gold", "silver"]}},
        },
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_upper_garment"] == "top"
    assert "left_appearance_costume_silhouette" not in row
    assert "right_scene_lighting_accent_colors_01" not in row
    assert "right_scene_lighting_accent_colors_02" not in row
    assert "right_scene_lighting_accent_colors_03" not in row
    assert "right_metadata_shot_type" not in row


def test_build_candidate_feature_row_default_schema_is_shape_stable_across_sparse_rows() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    sparse_descriptors = {
        "p1": {"upper_garment": "Top"},
        "p2": {"upper_garment": "top"},
        "p4": {"props": "fan; ribbon"},
    }
    mixed_shape_descriptors = {
        "p1": {"dominant_colors": "White/Purple", "upper_garment": "Top"},
        "p2": {"upper_garment": ["Top", "Jacket"]},
        "p3": {"upper_garment": "top"},
        "p4": {"props": ["fan"]},
        "p5": {"headwear": "hat"},
    }

    sparse_row = build_candidate_feature_row(candidate, descriptors=sparse_descriptors, embeddings=None)
    mixed_shape_row = build_candidate_feature_row(
        candidate,
        descriptors=mixed_shape_descriptors,
        embeddings=None,
    )

    assert set(sparse_row) == set(mixed_shape_row)
    assert "left_upper_garment_01" not in sparse_row
    assert "left_upper_garment_01" not in mixed_shape_row
    assert sparse_row["left_upper_garment"] == "top"
    assert mixed_shape_row["left_upper_garment"] == "top"
    assert mixed_shape_row["left_dominant_colors_01"] == "purple"
    assert mixed_shape_row["left_dominant_colors_02"] == "white"
    assert mixed_shape_row["right_props_01"] == "fan"
    assert mixed_shape_row["right_props_02"] == CANONICAL_MISSING


def test_build_candidate_feature_row_supports_extra_nested_fields_only_with_explicit_registry() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {
            "appearance": {"costume": {"silhouette": "Bell"}},
        },
        "p2": {
            "appearance": {"costume": {"silhouette": "bell"}},
        },
        "p4": {
            "scene": {"lighting": {"accent_colors": ["Blue", "Gold"]}},
        },
        "p5": {
            "scene": {"lighting": {"accent_colors": ["gold", "silver"]}},
        },
    }
    descriptor_field_registry = {
        "appearance_costume_silhouette": "scalar",
        "scene_lighting_accent_colors": "multivalue",
    }

    row = build_candidate_feature_row(
        candidate,
        descriptors=descriptors,
        embeddings=None,
        descriptor_field_registry=descriptor_field_registry,
    )

    assert row["left_appearance_costume_silhouette"] == "bell"
    assert row["right_scene_lighting_accent_colors_01"] == "blue"
    assert row["right_scene_lighting_accent_colors_02"] == "gold"
    assert row["right_scene_lighting_accent_colors_03"] == "silver"
    assert row["right_scene_lighting_accent_colors_04"] == CANONICAL_MISSING


def test_build_candidate_feature_row_emits_missing_schema_columns_by_default() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }

    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=None)
    expected_descriptor_columns = _default_descriptor_feature_columns()
    non_descriptor_left_right_columns = {
        "left_internal_gap_mean",
        "right_internal_gap_mean",
    }
    descriptor_columns = {
        key
        for key in row
        if (key.startswith("left_") or key.startswith("right_"))
        and key not in non_descriptor_left_right_columns
    }

    assert row["left_people_count"] == CANONICAL_MISSING
    assert row["right_performer_view"] == CANONICAL_MISSING
    assert row["left_headwear"] == CANONICAL_MISSING
    assert row["right_footwear"] == CANONICAL_MISSING
    assert row["left_dominant_colors_01"] == CANONICAL_MISSING
    assert row["left_dominant_colors_05"] == CANONICAL_MISSING
    assert row["right_props_01"] == CANONICAL_MISSING
    assert row["right_props_05"] == CANONICAL_MISSING
    assert descriptor_columns == expected_descriptor_columns


def test_build_candidate_feature_row_ignores_malformed_multivalue_items() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"dominant_colors": ["White", None, True, "", " / ", "Blue"]},
        "p2": {"dominant_colors": [False, "blue"]},
        "p3": {"props": ["fan", None, True, "", " ribbon "]},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_dominant_colors_01"] == "blue"
    assert row["left_dominant_colors_02"] == "white"
    assert row["left_dominant_colors_03"] == CANONICAL_MISSING
    assert row["left_props_01"] == "fan"
    assert row["left_props_02"] == "ribbon"
    assert row["left_props_03"] == CANONICAL_MISSING


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
        "p1": {"upper_garment": "top"},
        "p2": "not-a-mapping",
    }

    with pytest.raises(ValueError, match="must be a mapping"):
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
