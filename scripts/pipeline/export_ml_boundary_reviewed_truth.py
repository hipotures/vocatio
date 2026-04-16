#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

try:
    from lib.ml_boundary_review_truth_export import (
        flatten_final_display_sets,
        load_review_index_payload_json,
        load_review_state_json,
        rebuild_final_display_sets,
    )
    from lib.pipeline_io import atomic_write_csv
    from lib.workspace_dir import resolve_workspace_dir
except ModuleNotFoundError:
    from scripts.pipeline.lib.ml_boundary_review_truth_export import (
        flatten_final_display_sets,
        load_review_index_payload_json,
        load_review_state_json,
        rebuild_final_display_sets,
    )
    from scripts.pipeline.lib.pipeline_io import atomic_write_csv
    from scripts.pipeline.lib.workspace_dir import resolve_workspace_dir


console = Console()

DEFAULT_INDEX_FILENAME = "performance_proxy_index.json"
DEFAULT_STATE_FILENAME = "review_state.json"
DEFAULT_OUTPUT_FILENAME = "ml_boundary_reviewed_truth.csv"
OUTPUT_HEADERS = ["photo_id", "segment_id", "segment_type"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed ML boundary truth rows from review index and review state."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds review artifacts. Default: DAY/.vocatio WORKSPACE_DIR or DAY/_workspace",
    )
    parser.add_argument(
        "--index",
        help=f"Review index filename or absolute path. Default: WORKSPACE/{DEFAULT_INDEX_FILENAME}",
    )
    parser.add_argument(
        "--state",
        help=f"Review state filename or absolute path. Default: WORKSPACE/{DEFAULT_STATE_FILENAME}",
    )
    parser.add_argument(
        "--output",
        help=f"Output CSV filename or absolute path. Default: WORKSPACE/{DEFAULT_OUTPUT_FILENAME}",
    )
    return parser.parse_args(argv)


def _resolve_workspace_path(workspace_dir: Path, value: Optional[str], default_name: str) -> Path:
    if not value:
        return workspace_dir / default_name
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).expanduser().resolve()
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    index_path = _resolve_workspace_path(workspace_dir, args.index, DEFAULT_INDEX_FILENAME)
    state_path = _resolve_workspace_path(workspace_dir, args.state, DEFAULT_STATE_FILENAME)
    output_path = _resolve_workspace_path(workspace_dir, args.output, DEFAULT_OUTPUT_FILENAME)

    try:
        review_index_payload = load_review_index_payload_json(index_path, day_dir=day_dir)
        review_state = load_review_state_json(state_path, day=str(review_index_payload.get("day", "") or ""))
        display_sets = rebuild_final_display_sets(review_index_payload, review_state)
        rows = flatten_final_display_sets(display_sets)
        atomic_write_csv(output_path, OUTPUT_HEADERS, rows)
    except Exception as error:
        console.print(f"[red]Error: {error}[/red]")
        return 1

    console.print(f"[green]Wrote {len(rows)} reviewed truth rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
