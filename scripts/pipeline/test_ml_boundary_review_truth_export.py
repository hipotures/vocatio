import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_review_truth_export import flatten_final_display_sets, rebuild_final_display_sets


def _photo(
    photo_id: str,
    filename: str,
    adjusted_start_local: str,
) -> dict[str, str]:
    return {
        "photo_id": photo_id,
        "filename": filename,
        "adjusted_start_local": adjusted_start_local,
    }


def _performance(
    *,
    performance_number: str,
    set_id: str,
    photos: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "performance_number": performance_number,
        "set_id": set_id,
        "timeline_status": "normal",
        "performance_start_local": photos[0]["adjusted_start_local"] if photos else "",
        "performance_end_local": photos[-1]["adjusted_start_local"] if photos else "",
        "photos": photos,
    }


def test_flatten_final_display_sets_keeps_unsplit_reviewed_set_in_one_segment() -> None:
    review_index_payload = {
        "performances": [
            _performance(
                performance_number="101",
                set_id="set-101",
                photos=[
                    _photo("p1", "IMG_0001.JPG", "2026-03-23T10:00:00"),
                    _photo("p2", "IMG_0002.JPG", "2026-03-23T10:00:05"),
                ],
            )
        ]
    }
    review_state = {
        "splits": {},
        "merges": [],
    }

    rows = flatten_final_display_sets(rebuild_final_display_sets(review_index_payload, review_state))

    assert rows == [
        {"photo_id": "p1", "segment_id": "set-101", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "set-101", "segment_type": "performance"},
    ]


def test_flatten_final_display_sets_uses_start_filename_split_ids_and_v1_segment_types() -> None:
    review_index_payload = {
        "performances": [
            _performance(
                performance_number="102",
                set_id="set-102",
                photos=[
                    _photo("p1", "IMG_0001.JPG", "2026-03-23T11:00:00"),
                    _photo("p2", "IMG_0002.JPG", "2026-03-23T11:00:05"),
                    _photo("p3", "IMG_0003.JPG", "2026-03-23T11:00:10"),
                ],
            )
        ]
    }
    review_state = {
        "splits": {
            "set-102": [
                {"start_filename": "IMG_0002.JPG", "new_name": "Ceremony"},
                {"start_filename": "IMG_0003.JPG", "new_name": "Warmup"},
            ]
        },
        "merges": [],
    }

    rows = flatten_final_display_sets(rebuild_final_display_sets(review_index_payload, review_state))

    assert rows == [
        {"photo_id": "p1", "segment_id": "set-102", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "set-102::IMG_0002.JPG", "segment_type": "ceremony"},
        {"photo_id": "p3", "segment_id": "set-102::IMG_0003.JPG", "segment_type": "warmup"},
    ]


def test_flatten_final_display_sets_migrates_legacy_split_keys_from_performance_number() -> None:
    review_index_payload = {
        "performances": [
            _performance(
                performance_number="101",
                set_id="set-101",
                photos=[
                    _photo("p1", "a.jpg", "2026-03-23T11:00:00"),
                    _photo("p2", "b.jpg", "2026-03-23T11:00:05"),
                ],
            )
        ]
    }
    review_state = {
        "splits": {
            "101": [
                {"start_filename": "b.jpg", "new_name": "Ceremony"},
            ]
        },
        "merges": [],
    }

    rows = flatten_final_display_sets(rebuild_final_display_sets(review_index_payload, review_state))

    assert rows == [
        {"photo_id": "p1", "segment_id": "set-101", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "set-101::b.jpg", "segment_type": "ceremony"},
    ]


def test_flatten_final_display_sets_uses_final_merge_target_segment_for_all_merged_rows() -> None:
    review_index_payload = {
        "performances": [
            _performance(
                performance_number="103",
                set_id="set-103",
                photos=[
                    _photo("p1", "IMG_0001.JPG", "2026-03-23T12:00:00"),
                    _photo("p2", "IMG_0002.JPG", "2026-03-23T12:00:05"),
                ],
            ),
            _performance(
                performance_number="104",
                set_id="set-104",
                photos=[
                    _photo("p3", "IMG_0003.JPG", "2026-03-23T12:00:10"),
                    _photo("p4", "IMG_0004.JPG", "2026-03-23T12:00:15"),
                ],
            ),
        ]
    }
    review_state = {
        "splits": {
            "set-103": [
                {"start_filename": "IMG_0002.JPG", "new_name": "Ceremony"},
            ]
        },
        "merges": [
            {
                "target_set_id": "set-103::IMG_0002.JPG",
                "source_set_id": "set-104",
            }
        ],
    }

    rows = flatten_final_display_sets(rebuild_final_display_sets(review_index_payload, review_state))

    assert rows == [
        {"photo_id": "p1", "segment_id": "set-103", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "set-103::IMG_0002.JPG", "segment_type": "ceremony"},
        {"photo_id": "p3", "segment_id": "set-103::IMG_0002.JPG", "segment_type": "ceremony"},
        {"photo_id": "p4", "segment_id": "set-103::IMG_0002.JPG", "segment_type": "ceremony"},
    ]
