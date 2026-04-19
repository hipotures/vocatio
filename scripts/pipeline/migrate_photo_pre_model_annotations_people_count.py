#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lib.caption_scene_common import resolve_path
from lib.photo_pre_model_annotations import DEFAULT_OUTPUT_DIRNAME, normalize_annotation_data
from lib.pipeline_io import atomic_write_json
from lib.workspace_dir import resolve_workspace_dir

console = Console()


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate existing photo pre-model annotation JSON files to canonical people_count categories.",
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("--workspace-dir", help="Workspace directory. Default: resolve from .vocatio or DAY/_workspace")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=f"Annotation directory relative to workspace or absolute. Default: {DEFAULT_OUTPUT_DIRNAME}",
    )
    return parser.parse_args(argv)


def load_annotation_paths(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.rglob("*.json") if path.is_file())


def migrate_annotation_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("annotation payload is missing object field 'data'")
    normalized_data = normalize_annotation_data(data)
    if normalized_data == data:
        return False
    payload["data"] = normalized_data
    atomic_write_json(path, payload)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    output_dir = resolve_path(workspace_dir, args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {output_dir}")

    annotation_paths = load_annotation_paths(output_dir)
    migrated = 0
    unchanged = 0
    failed = 0

    with Progress(*build_progress_columns(), console=console, expand=False) as progress:
        task_id = progress.add_task("Migrate annotations".ljust(25), total=len(annotation_paths))
        for path in annotation_paths:
            try:
                changed = migrate_annotation_file(path)
            except Exception as exc:
                failed += 1
                console.print(f"[yellow]Failed migration for {path}: {exc}[/yellow]")
            else:
                if changed:
                    migrated += 1
                else:
                    unchanged += 1
            progress.advance(task_id)

    console.print(
        f"files={len(annotation_paths)}\n"
        f"migrated={migrated}\n"
        f"unchanged={unchanged}\n"
        f"failed={failed}\n"
        f"outdir={output_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
