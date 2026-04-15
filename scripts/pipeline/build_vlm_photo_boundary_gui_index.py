#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rich.console import Console

import probe_vlm_photo_boundaries as probe
from lib.pipeline_io import atomic_write_json


from lib.workspace_dir import resolve_workspace_dir
console = Console()

DEFAULT_OUTPUT_FILENAME = probe.DEFAULT_OUTPUT_FILENAME
DEFAULT_GUI_INDEX_FILENAME = "performance_proxy_index.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GUI-compatible image-only review index from existing VLM probe CSV rows."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Probe CSV filename or absolute path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--run-id",
        help="Specific VLM run_id to visualize. Default: latest run metadata in _workspace/vlm_runs",
    )
    parser.add_argument(
        "--gui-index-output",
        default=DEFAULT_GUI_INDEX_FILENAME,
        help=f"GUI index JSON filename or absolute path. Default: {DEFAULT_GUI_INDEX_FILENAME}",
    )
    return parser.parse_args(argv)


def load_run_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_run_metadata(*, workspace_dir: Path, run_id: Optional[str]) -> Dict[str, Any]:
    runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
    if not runs_dir.exists():
        raise ValueError(f"Run metadata directory does not exist: {runs_dir}")
    if run_id:
        metadata_path = runs_dir / f"{run_id}.json"
        if not metadata_path.exists():
            raise ValueError(f"Run metadata JSON does not exist: {metadata_path}")
        return load_run_metadata(metadata_path)
    metadata_paths = sorted(runs_dir.glob("vlm-*.json"))
    if not metadata_paths:
        raise ValueError(f"No run metadata JSON files found in {runs_dir}")
    return load_run_metadata(metadata_paths[-1])


def build_gui_index_for_run(
    *,
    workspace_dir: Path,
    run_metadata: Mapping[str, Any],
    output_csv: Path,
) -> tuple[Dict[str, Any], int]:
    image_variant = str(run_metadata["args"]["image_variant"])
    embedded_manifest_csv = Path(str(run_metadata["embedded_manifest_csv"]))
    photo_manifest_csv = Path(str(run_metadata["photo_manifest_csv"]))
    all_result_rows = probe.read_result_rows(output_csv)
    run_id = str(run_metadata["run_id"])
    run_rows = [row for row in all_result_rows if str(row.get("run_id", "")) == run_id]
    if not run_rows:
        raise ValueError(f"No probe CSV rows found for run_id={run_id}")
    joined_rows = probe.read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )
    ordered_rows = joined_rows
    if not ordered_rows:
        raise ValueError(f"No joined manifest rows found for run_id={run_id}")
    payload = probe.build_gui_index_payload(
        day_name=workspace_dir.parent.name,
        workspace_dir=workspace_dir,
        image_variant=image_variant,
        run_id=run_id,
        ordered_rows=ordered_rows,
        result_rows=run_rows,
    )
    return payload, len(run_rows)


def build_summary_message(
    *,
    run_id: str,
    run_row_count: int,
    payload: Mapping[str, Any],
    gui_index_output: Path,
) -> str:
    return (
        f"Wrote GUI index for {run_id} to {gui_index_output} "
        f"from {run_row_count} VLM rows, {int(payload.get('photo_count', 0))} photos, "
        f"{int(payload.get('performance_count', 0))} set(s)."
    )


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    output_csv = probe.resolve_path(workspace_dir, args.output_csv)
    gui_index_output = probe.resolve_path(workspace_dir, args.gui_index_output)
    if not output_csv.exists():
        raise SystemExit(f"Probe CSV does not exist: {output_csv}")
    run_metadata = select_run_metadata(workspace_dir=workspace_dir, run_id=args.run_id)
    payload, run_row_count = build_gui_index_for_run(
        workspace_dir=workspace_dir,
        run_metadata=run_metadata,
        output_csv=output_csv,
    )
    atomic_write_json(gui_index_output, payload)
    console.print(
        build_summary_message(
            run_id=str(run_metadata["run_id"]),
            run_row_count=run_row_count,
            payload=payload,
            gui_index_output=gui_index_output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
