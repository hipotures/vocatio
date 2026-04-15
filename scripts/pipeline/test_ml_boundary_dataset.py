import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.ml_boundary_dataset import canonical_candidate_id, sort_photo_rows


def test_sort_photo_rows_uses_timestamp_order_idx_photo_id() -> None:
    rows = [
        {"photo_id": "p3", "order_idx": "10", "timestamp": "2025-03-25T08:00:00.000"},
        {"photo_id": "p4", "order_idx": "2", "timestamp": "2025-03-25T08:00:01.000"},
        {"photo_id": "p2", "order_idx": "2", "timestamp": "2025-03-25T08:00:00.000"},
        {"photo_id": "p0", "order_idx": "99", "timestamp": "2025-03-25T07:59:59.000"},
        {"photo_id": "p1", "order_idx": 2, "timestamp": "2025-03-25T08:00:00.000"},
    ]

    ordered = sort_photo_rows(rows)

    assert [row["photo_id"] for row in ordered] == ["p0", "p1", "p2", "p3", "p4"]


def test_sort_photo_rows_rejects_missing_or_blank_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows([{"photo_id": "p1", "timestamp": "2025-03-25T08:00:00.000"}])

    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows(
            [{"photo_id": "p1", "order_idx": "", "timestamp": "2025-03-25T08:00:00.000"}]
        )


def test_canonical_candidate_id_is_stable() -> None:
    first = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    second = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    changed = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v2",
    )

    assert first == second
    assert changed != first
