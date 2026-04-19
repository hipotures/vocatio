from __future__ import annotations

import argparse

DEFAULT_WINDOW_RADIUS = 5


def positive_window_radius_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def window_radius_to_window_size(window_radius: int) -> int:
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1")
    return window_radius * 2


def build_window_start_indexes(total_rows: int, window_radius: int) -> list[int]:
    window_size = window_radius_to_window_size(window_radius)
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    starts = list(range(0, total_rows - window_size + 1, window_radius))
    final_start = total_rows - window_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_centered_window_bounds(total_rows: int, cut_index: int, window_radius: int) -> tuple[int, int]:
    window_size = window_radius_to_window_size(window_radius)
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    if cut_index < 0 or cut_index >= total_rows - 1:
        raise ValueError("cut_index must reference a boundary between consecutive rows")
    final_start = total_rows - window_size
    start_index = cut_index - window_radius + 1
    if start_index < 0:
        start_index = 0
    if start_index > final_start:
        start_index = final_start
    return start_index, start_index + window_size
