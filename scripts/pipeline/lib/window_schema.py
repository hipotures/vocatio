from __future__ import annotations

import math
import random
from typing import Literal, Mapping, Sequence, TypeAlias

GapSide: TypeAlias = Literal["left", "right"]
WindowSchemaName: TypeAlias = Literal[
    "consecutive",
    "random",
    "index_quantile",
    "time_quantile",
    "time_max_min",
    "time_boundary_spread",
]

DEFAULT_WINDOW_SCHEMA: WindowSchemaName = "consecutive"
DEFAULT_WINDOW_SCHEMA_SEED = 42
WINDOW_SCHEMA_VALUES: tuple[WindowSchemaName, ...] = (
    "consecutive",
    "random",
    "index_quantile",
    "time_quantile",
    "time_max_min",
    "time_boundary_spread",
)


def parse_window_schema(value: object) -> WindowSchemaName:
    normalized = str(value or "").strip()
    if normalized == "":
        return DEFAULT_WINDOW_SCHEMA
    if normalized not in WINDOW_SCHEMA_VALUES:
        raise ValueError("window schema must be one of: " + ", ".join(WINDOW_SCHEMA_VALUES))
    return normalized


def parse_window_schema_seed(value: object) -> int:
    normalized = str(value or "").strip()
    if normalized == "":
        return DEFAULT_WINDOW_SCHEMA_SEED
    return int(normalized)


def _distance_from_gap(index: int, row_count: int, gap_side: GapSide) -> int:
    if gap_side == "left":
        return (row_count - 1) - index
    return index


def _timestamp_for_row(row: Mapping[str, str]) -> float:
    raw_value = str(row.get("start_epoch_ms", "") or "").strip()
    if raw_value == "":
        raise ValueError("time-based window schema requires start_epoch_ms")
    timestamp = float(raw_value)
    if not math.isfinite(timestamp):
        raise ValueError("time-based window schema requires finite start_epoch_ms")
    return timestamp


def _normalize_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(row) for row in rows]


def _select_closest_unique_indexes(
    targets: Sequence[float],
    values: Sequence[float],
    *,
    gap_side: GapSide,
) -> list[int]:
    row_count = len(values)
    remaining_indexes = set(range(row_count))
    selected_indexes: list[int] = []
    for target in targets:
        best_index = min(
            remaining_indexes,
            key=lambda index: (
                abs(values[index] - target),
                -_distance_from_gap(index, row_count, gap_side),
                index,
            ),
        )
        selected_indexes.append(best_index)
        remaining_indexes.remove(best_index)
    return sorted(selected_indexes)


def _choose_index_quantile_indexes(
    row_count: int,
    selected_count: int,
    *,
    gap_side: GapSide,
) -> list[int]:
    if selected_count >= row_count:
        return list(range(row_count))
    if selected_count == 1:
        return [0 if gap_side == "left" else row_count - 1]
    targets = [index * (row_count - 1) / (selected_count - 1) for index in range(selected_count)]
    values = [float(index) for index in range(row_count)]
    return _select_closest_unique_indexes(targets, values, gap_side=gap_side)


def _choose_time_quantile_indexes(
    rows: Sequence[Mapping[str, str]],
    selected_count: int,
    *,
    gap_side: GapSide,
) -> list[int]:
    timestamps = [_timestamp_for_row(row) for row in rows]
    if selected_count >= len(rows):
        return list(range(len(rows)))
    if selected_count == 1:
        return [0 if gap_side == "left" else len(rows) - 1]
    start_time = timestamps[0]
    end_time = timestamps[-1]
    targets = [
        start_time + index * (end_time - start_time) / (selected_count - 1)
        for index in range(selected_count)
    ]
    return _select_closest_unique_indexes(targets, timestamps, gap_side=gap_side)


def _greedy_max_min_indexes(
    rows: Sequence[Mapping[str, str]],
    selected_count: int,
    *,
    gap_side: GapSide,
    seed_indexes: Sequence[int],
) -> list[int]:
    row_count = len(rows)
    if selected_count >= row_count:
        return list(range(row_count))
    timestamps = [_timestamp_for_row(row) for row in rows]
    selected = list(dict.fromkeys(seed_indexes))
    if not selected:
        selected = [0 if gap_side == "left" else row_count - 1]
    while len(selected) < selected_count:
        remaining = [index for index in range(row_count) if index not in selected]
        best_index = max(
            remaining,
            key=lambda index: (
                min(abs(timestamps[index] - timestamps[chosen]) for chosen in selected),
                _distance_from_gap(index, row_count, gap_side),
                -index,
            ),
        )
        selected.append(best_index)
    return sorted(selected)


def select_segment_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    radius: int,
    schema: WindowSchemaName | str,
    gap_side: GapSide,
    schema_seed: int,
) -> list[dict[str, str]]:
    normalized_rows = _normalize_rows(rows)
    selected_count = min(int(radius), len(normalized_rows))
    if selected_count <= 0:
        return []

    normalized_schema = parse_window_schema(schema)
    if selected_count >= len(normalized_rows):
        return normalized_rows
    if normalized_schema == "consecutive":
        if gap_side == "left":
            return normalized_rows[-selected_count:]
        return normalized_rows[:selected_count]
    if normalized_schema == "random":
        relative_paths = [str(row.get("relative_path", "") or "") for row in normalized_rows]
        rng = random.Random(f"{schema_seed}:{gap_side}:{'|'.join(relative_paths)}")
        chosen_indexes = sorted(rng.sample(range(len(normalized_rows)), selected_count))
        return [normalized_rows[index] for index in chosen_indexes]
    if normalized_schema == "index_quantile":
        chosen_indexes = _choose_index_quantile_indexes(
            len(normalized_rows),
            selected_count,
            gap_side=gap_side,
        )
        return [normalized_rows[index] for index in chosen_indexes]
    if normalized_schema == "time_quantile":
        chosen_indexes = _choose_time_quantile_indexes(
            normalized_rows,
            selected_count,
            gap_side=gap_side,
        )
        return [normalized_rows[index] for index in chosen_indexes]
    if normalized_schema == "time_max_min":
        seed_indexes = [0, len(normalized_rows) - 1] if selected_count > 1 else []
        chosen_indexes = _greedy_max_min_indexes(
            normalized_rows,
            selected_count,
            gap_side=gap_side,
            seed_indexes=seed_indexes,
        )
        return [normalized_rows[index] for index in chosen_indexes]
    chosen_indexes = _greedy_max_min_indexes(
        normalized_rows,
        selected_count,
        gap_side=gap_side,
        seed_indexes=[len(normalized_rows) - 1 if gap_side == "left" else 0],
    )
    return [normalized_rows[index] for index in chosen_indexes]
