#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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

from lib.image_pipeline_contracts import PHOTO_MANIFEST_HEADERS
from lib.photo_time_order import pick_capture_time_parts
from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_EXTENSIONS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}
EXIFTOOL_BATCH_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a recursive photo manifest for one day directory into a single logical photo stream CSV."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory where the CSV file will be written. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename or absolute path. Default: WORKSPACE/photo_manifest.csv",
    )
    parser.add_argument(
        "--stream-id",
        default="p-main",
        help="Logical stream ID to write. Default: p-main",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional device label to store in the manifest.",
    )
    return parser.parse_args()


def pick_capture_time(item: Mapping[str, object]) -> tuple[str, str, str]:
    parts = pick_capture_time_parts(item)
    return parts.capture_time_local, parts.capture_subsec, parts.timestamp_source


def collect_source_files(day_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(day_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_parts = path.relative_to(day_dir).parts
        if "_workspace" in relative_parts:
            continue
        if path.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        files.append(path)
    return files


def metadata_by_source_path(items: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    output: Dict[str, Mapping[str, object]] = {}
    for item in items:
        source_file = str(item.get("SourceFile") or "").strip()
        if not source_file:
            raise ValueError("Exif metadata item missing SourceFile")
        output[source_file] = item
    return output


def run_exiftool(paths: Sequence[Path]) -> List[Dict[str, object]]:
    if not paths:
        return []
    base_cmd = [
        "exiftool",
        "-json",
        "-n",
        "-DateTimeOriginal",
        "-SubSecDateTimeOriginal",
        "-CreateDate",
        "-SubSecCreateDate",
        "-FileModifyDate",
        "-FileCreateDate",
        "-Model",
        "-Make",
    ]
    items: List[Dict[str, object]] = []
    for index in range(0, len(paths), EXIFTOOL_BATCH_SIZE):
        batch = paths[index:index + EXIFTOOL_BATCH_SIZE]
        cmd = list(base_cmd)
        cmd.extend(str(path) for path in batch)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = result.stdout.strip()
        if not payload:
            raise ValueError("Exiftool returned empty metadata output")
        items.extend(json.loads(payload))
    return items


def build_manifest_rows(
    day_dir: Path,
    stream_id: str,
    device: str,
    metadata_by_path: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, str]]:
    day_name = day_dir.name
    files = collect_source_files(day_dir)
    if not files:
        raise ValueError(f"No photo files found under {day_dir}")
    rows_with_sort: List[tuple[tuple[object, str, str], Dict[str, str]]] = []
    for path in files:
        metadata = metadata_by_path.get(str(path))
        if metadata is None:
            raise ValueError(f"Missing metadata for {path}")
        time_parts = pick_capture_time_parts(metadata)
        relative_path = path.relative_to(day_dir).as_posix()
        source_rel_dir = path.parent.relative_to(day_dir).as_posix()
        row = {header: "" for header in PHOTO_MANIFEST_HEADERS}
        row.update(
            {
                "day": day_name,
                "stream_id": stream_id,
                "device": device,
                "media_type": "photo",
                "source_root": str(day_dir),
                "source_dir": str(path.parent),
                "source_rel_dir": "" if source_rel_dir == "." else source_rel_dir,
                "path": str(path),
                "relative_path": relative_path,
                "photo_id": relative_path,
                "filename": relative_path,
                "extension": path.suffix.lower(),
                "capture_time_local": time_parts.capture_time_local,
                "capture_subsec": time_parts.capture_subsec,
                "start_local": time_parts.start_local,
                "start_epoch_ms": time_parts.start_epoch_ms,
                "timestamp_source": time_parts.timestamp_source,
                "model": str(metadata.get("Model") or ""),
                "make": str(metadata.get("Make") or ""),
                "actual_size_bytes": str(path.stat().st_size),
                "create_date_raw": str(metadata.get("CreateDate") or ""),
                "datetime_original_raw": str(metadata.get("DateTimeOriginal") or ""),
                "subsec_datetime_original_raw": str(metadata.get("SubSecDateTimeOriginal") or ""),
                "subsec_create_date_raw": str(metadata.get("SubSecCreateDate") or ""),
                "file_modify_date_raw": str(metadata.get("FileModifyDate") or ""),
                "file_create_date_raw": str(metadata.get("FileCreateDate") or ""),
            }
        )
        rows_with_sort.append(((time_parts.sort_dt, row["capture_subsec"], row["relative_path"]), row))
    rows = [row for _sort_key, row in sorted(rows_with_sort, key=lambda item: item[0])]
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)
    return rows


def resolve_output_path(workspace_dir: Path, output_value: Optional[str], stream_id: str) -> Path:
    if not output_value:
        return workspace_dir / "photo_manifest.csv"
    candidate = Path(output_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def export_recursive_photo_csv(
    day_dir: Path,
    output_path: Path,
    stream_id: str,
    device: str,
) -> int:
    files = collect_source_files(day_dir)
    if not files:
        raise ValueError(f"No photo files found under {day_dir}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    ) as progress:
        discover_task = progress.add_task("Discover photos".ljust(25), total=len(files))
        progress.update(discover_task, completed=len(files))
        metadata_task = progress.add_task("Read EXIF metadata".ljust(25), total=len(files))
        metadata_items = run_exiftool(files)
        progress.update(metadata_task, completed=len(files))
        rows_task = progress.add_task("Build manifest rows".ljust(25), total=len(files))
        rows = build_manifest_rows(day_dir, stream_id, device, metadata_by_source_path(metadata_items))
        progress.update(rows_task, completed=len(files))
    atomic_write_csv(output_path, PHOTO_MANIFEST_HEADERS, rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    output_path = resolve_output_path(workspace_dir, args.output, args.stream_id)
    row_count = export_recursive_photo_csv(
        day_dir=day_dir,
        output_path=output_path,
        stream_id=args.stream_id,
        device=args.device,
    )
    console.print(f"Wrote {row_count} photo rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
