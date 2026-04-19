import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_truth import FinalPhotoTruth, build_final_photo_truth


def test_build_final_photo_truth_assigns_segment_fields() -> None:
    rows = [
        {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "s2", "segment_type": "ceremony"},
    ]

    truth = build_final_photo_truth(rows)

    assert truth["p1"] == FinalPhotoTruth(photo_id="p1", segment_id="s1", segment_type="performance")
    assert truth["p2"].segment_type == "ceremony"


def test_build_final_photo_truth_rejects_duplicate_photo_id() -> None:
    rows = [
        {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p1", "segment_id": "s2", "segment_type": "ceremony"},
    ]

    with pytest.raises(ValueError, match="duplicate photo_id"):
        build_final_photo_truth(rows)


def test_build_final_photo_truth_rejects_missing_or_blank_identity_fields() -> None:
    cases = [
        ({}, "photo_id"),
        ({"photo_id": "", "segment_id": "s1", "segment_type": "performance"}, "photo_id"),
        ({"photo_id": "p1", "segment_type": "performance"}, "segment_id"),
        ({"photo_id": "p1", "segment_id": " ", "segment_type": "performance"}, "segment_id"),
        ({"photo_id": "p1", "segment_id": "s1"}, "segment_type"),
        ({"photo_id": "p1", "segment_id": "s1", "segment_type": ""}, "segment_type"),
    ]

    for row, field_name in cases:
        with pytest.raises(ValueError, match=field_name):
            build_final_photo_truth([row])


def test_build_final_photo_truth_rejects_invalid_segment_type() -> None:
    row = {"photo_id": "p1", "segment_id": "s1", "segment_type": "reception"}

    with pytest.raises(ValueError, match="segment_type"):
        build_final_photo_truth([row])
