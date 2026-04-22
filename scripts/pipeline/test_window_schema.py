import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import window_schema


def _row(relative_path: str, start_epoch_ms: int) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "start_epoch_ms": str(start_epoch_ms),
    }


def test_defaults_and_validation() -> None:
    assert window_schema.DEFAULT_WINDOW_SCHEMA == "consecutive"
    assert window_schema.DEFAULT_WINDOW_SCHEMA_SEED == 42
    assert window_schema.parse_window_schema("") == "consecutive"
    assert window_schema.parse_window_schema("random") == "random"
    assert window_schema.parse_window_schema_seed("") == 42
    assert window_schema.parse_window_schema_seed("99") == 99

    with pytest.raises(ValueError, match="time_boundary_spread"):
        window_schema.parse_window_schema("weird")


def test_consecutive_selects_rows_nearest_gap() -> None:
    rows = [
        _row("cam/a.jpg", 1000),
        _row("cam/b.jpg", 2000),
        _row("cam/c.jpg", 3000),
        _row("cam/d.jpg", 4000),
    ]

    left_selected = window_schema.select_segment_rows(
        rows,
        radius=2,
        schema="consecutive",
        gap_side="left",
        schema_seed=42,
    )
    right_selected = window_schema.select_segment_rows(
        rows,
        radius=2,
        schema="consecutive",
        gap_side="right",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in left_selected] == ["cam/c.jpg", "cam/d.jpg"]
    assert [row["relative_path"] for row in right_selected] == ["cam/a.jpg", "cam/b.jpg"]


def test_small_segment_returns_all_available_rows_without_padding() -> None:
    rows = [_row("cam/a.jpg", 1000), _row("cam/b.jpg", 2000)]

    selected = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="random",
        gap_side="left",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in selected] == ["cam/a.jpg", "cam/b.jpg"]


def test_random_is_deterministic_and_unique() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 8)]

    first = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="random",
        gap_side="right",
        schema_seed=42,
    )
    second = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="random",
        gap_side="right",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in first] == [row["relative_path"] for row in second]
    assert len({row["relative_path"] for row in first}) == 3


def test_index_quantile_spreads_evenly_by_index() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 11)]

    selected = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="index_quantile",
        gap_side="left",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in selected] == [
        "cam/1.jpg",
        "cam/5.jpg",
        "cam/10.jpg",
    ]


def test_time_quantile_uses_timestamps_and_tie_breaks_away_from_gap() -> None:
    rows = [
        _row("cam/1.jpg", 1000),
        _row("cam/2.jpg", 2000),
        _row("cam/3.jpg", 3000),
        _row("cam/4.jpg", 4000),
        _row("cam/5.jpg", 5000),
        _row("cam/20.jpg", 20000),
        _row("cam/21.jpg", 21000),
        _row("cam/22.jpg", 22000),
        _row("cam/23.jpg", 23000),
        _row("cam/24.jpg", 24000),
    ]

    selected = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="time_quantile",
        gap_side="left",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in selected] == [
        "cam/1.jpg",
        "cam/5.jpg",
        "cam/24.jpg",
    ]


def test_time_max_min_maximizes_spread() -> None:
    rows = [
        _row("cam/1.jpg", 1000),
        _row("cam/2.jpg", 2000),
        _row("cam/3.jpg", 3000),
        _row("cam/10.jpg", 10000),
        _row("cam/11.jpg", 11000),
        _row("cam/20.jpg", 20000),
    ]

    selected = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="time_max_min",
        gap_side="left",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in selected] == [
        "cam/1.jpg",
        "cam/10.jpg",
        "cam/20.jpg",
    ]


def test_time_boundary_spread_keeps_boundary_nearest_photo() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 8)]

    selected = window_schema.select_segment_rows(
        rows,
        radius=3,
        schema="time_boundary_spread",
        gap_side="left",
        schema_seed=42,
    )

    assert [row["relative_path"] for row in selected] == [
        "cam/1.jpg",
        "cam/4.jpg",
        "cam/7.jpg",
    ]
