import sys
from pathlib import Path

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


def test_build_final_photo_truth_keeps_last_duplicate_photo_id() -> None:
    rows = [
        {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p1", "segment_id": "s2", "segment_type": "ceremony"},
    ]

    truth = build_final_photo_truth(rows)

    assert truth["p1"] == FinalPhotoTruth(photo_id="p1", segment_id="s2", segment_type="ceremony")
