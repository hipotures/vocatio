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


def test_sort_photo_rows_normalizes_numeric_string_timestamps() -> None:
    rows = [
        {"photo_id": "p10", "order_idx": "1", "timestamp": "10"},
        {"photo_id": "p2", "order_idx": "1", "timestamp": "2"},
    ]

    ordered = sort_photo_rows(rows)

    assert [row["photo_id"] for row in ordered] == ["p2", "p10"]


def test_sort_photo_rows_orders_mixed_numeric_and_iso_timestamps() -> None:
    rows = [
        {"photo_id": "p2", "order_idx": "1", "timestamp": "1970-01-01T00:00:01+00:00"},
        {"photo_id": "p1", "order_idx": "1", "timestamp": "0"},
    ]

    ordered = sort_photo_rows(rows)

    assert [row["photo_id"] for row in ordered] == ["p1", "p2"]


def test_sort_photo_rows_orders_iso_timestamps_with_timezone_offsets() -> None:
    rows = [
        {"photo_id": "p2", "order_idx": "1", "timestamp": "2025-03-25T08:30:00+00:00"},
        {"photo_id": "p1", "order_idx": "1", "timestamp": "2025-03-25T10:00:00+02:00"},
    ]

    ordered = sort_photo_rows(rows)

    assert [row["photo_id"] for row in ordered] == ["p1", "p2"]


def test_sort_photo_rows_handles_mixed_aware_and_naive_iso_as_utc() -> None:
    rows = [
        {"photo_id": "p2", "order_idx": "1", "timestamp": "2025-03-25T08:00:00+00:00"},
        {"photo_id": "p1", "order_idx": "1", "timestamp": "2025-03-25T08:00:00"},
    ]

    ordered = sort_photo_rows(rows)

    assert [row["photo_id"] for row in ordered] == ["p1", "p2"]


def test_sort_photo_rows_rejects_boolean_or_non_finite_timestamp() -> None:
    for timestamp in (True, False, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="timestamp"):
            sort_photo_rows([{"photo_id": "p1", "order_idx": "1", "timestamp": timestamp}])


def test_sort_photo_rows_rejects_invalid_or_blank_timestamp() -> None:
    for timestamp in (None, "", "   ", "not-a-timestamp"):
        with pytest.raises(ValueError, match="timestamp"):
            sort_photo_rows([{"photo_id": "p1", "order_idx": "1", "timestamp": timestamp}])


def test_sort_photo_rows_rejects_missing_or_blank_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows([{"photo_id": "p1", "timestamp": "2025-03-25T08:00:00.000"}])

    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows(
            [{"photo_id": "p1", "order_idx": "", "timestamp": "2025-03-25T08:00:00.000"}]
        )


def test_sort_photo_rows_rejects_non_numeric_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows(
            [{"photo_id": "p1", "order_idx": "abc", "timestamp": "2025-03-25T08:00:00.000"}]
        )


def test_sort_photo_rows_rejects_non_integral_float_order_idx() -> None:
    with pytest.raises(ValueError, match="order_idx"):
        sort_photo_rows(
            [{"photo_id": "p1", "order_idx": 2.5, "timestamp": "2025-03-25T08:00:00.000"}]
        )


def test_sort_photo_rows_rejects_missing_or_blank_photo_id() -> None:
    cases = [
        {"order_idx": "1", "timestamp": "2025-03-25T08:00:00.000"},
        {"photo_id": "", "order_idx": "1", "timestamp": "2025-03-25T08:00:00.000"},
        {"photo_id": " ", "order_idx": "1", "timestamp": "2025-03-25T08:00:00.000"},
    ]

    for row in cases:
        with pytest.raises(ValueError, match="photo_id"):
            sort_photo_rows([row])


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
    changed_day = canonical_candidate_id(
        day_id="20250326",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    changed_left = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p5",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    changed_right = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p6",
        candidate_rule_version="gap-v1",
    )
    swapped = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p4",
        center_right_photo_id="p3",
        candidate_rule_version="gap-v1",
    )

    assert first == second
    assert changed_day != first
    assert changed_left != first
    assert changed_right != first
    assert swapped != first


def test_canonical_candidate_id_avoids_delimiter_collision() -> None:
    first = canonical_candidate_id(
        day_id="a|b",
        center_left_photo_id="c",
        center_right_photo_id="d",
        candidate_rule_version="e",
    )
    second = canonical_candidate_id(
        day_id="a",
        center_left_photo_id="b",
        center_right_photo_id="c|d",
        candidate_rule_version="e",
    )

    assert first != second
