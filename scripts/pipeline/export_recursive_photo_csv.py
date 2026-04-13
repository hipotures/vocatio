#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lib.image_pipeline_contracts import PHOTO_MANIFEST_HEADERS
from lib.photo_time_order import pick_capture_time_parts


console = Console()

PHOTO_EXTENSIONS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}
EXIFTOOL_BATCH_SIZE = 1000
EXIFTOOL_PROGRESS_RE = re.compile(r"\[(?P<current>\d+)/(?P<total>\d+)\]\s*$")


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


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )


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


def parse_exiftool_progress_line(line: str) -> Optional[tuple[int, int]]:
    match = EXIFTOOL_PROGRESS_RE.search(line.strip())
    if match is None:
        return None
    return int(match.group("current")), int(match.group("total"))


def partial_output_path(output_path: Path) -> Path:
    return Path(f"{output_path}.partial")


def flush_and_sync(handle: TextIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def remove_stale_partial_output(partial_path: Path) -> None:
    partial_path.unlink(missing_ok=True)


def open_partial_manifest_csv(path: Path) -> tuple[TextIO, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=PHOTO_MANIFEST_HEADERS)
    writer.writeheader()
    flush_and_sync(handle)
    return handle, writer


def append_partial_manifest_rows(handle: TextIO, writer: csv.DictWriter, rows: Sequence[Mapping[str, str]]) -> None:
    writer.writerows(rows)
    flush_and_sync(handle)


def rewrite_partial_manifest_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHOTO_MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
        flush_and_sync(handle)


def replace_output_from_partial(partial_path: Path, output_path: Path) -> None:
    os.replace(partial_path, output_path)
    fsync_directory(output_path.parent)


def exiftool_base_command() -> List[str]:
    base_cmd = [
        "exiftool",
        "-json",
        "-progress",
        "-q",
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
    return base_cmd


def run_exiftool_batch(
    paths: Sequence[Path],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, object]]:
    if not paths:
        return []
    cmd = exiftool_base_command()
    cmd.extend(str(path) for path in paths)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_handle:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stderr is not None
        for line in process.stderr:
            parsed_progress = parse_exiftool_progress_line(line)
            if parsed_progress is not None:
                if progress_callback is not None:
                    progress_callback(*parsed_progress)
                continue
            if line.strip():
                console.print(line.rstrip())
        return_code = process.wait()
        stdout_handle.seek(0)
        payload = stdout_handle.read().strip()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    if not payload:
        raise ValueError("Exiftool returned empty metadata output")
    return json.loads(payload)


def run_exiftool(paths: Sequence[Path]) -> List[Dict[str, object]]:
    if not paths:
        return []
    items: List[Dict[str, object]] = []
    for index in range(0, len(paths), EXIFTOOL_BATCH_SIZE):
        batch = paths[index:index + EXIFTOOL_BATCH_SIZE]
        items.extend(run_exiftool_batch(batch))
    return items


def build_manifest_entry(
    day_dir: Path,
    stream_id: str,
    device: str,
    path: Path,
    metadata: Mapping[str, object],
) -> tuple[tuple[object, str, str], Dict[str, str]]:
    day_name = day_dir.name
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
            "photo_order_index": "",
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
    return (time_parts.sort_dt, row["capture_subsec"], row["relative_path"]), row


def build_batch_manifest_rows(
    day_dir: Path,
    stream_id: str,
    device: str,
    paths: Sequence[Path],
    metadata_by_path: Mapping[str, Mapping[str, object]],
) -> List[tuple[tuple[object, str, str], Dict[str, str]]]:
    rows_with_sort: List[tuple[tuple[object, str, str], Dict[str, str]]] = []
    invalid_capture_paths: List[str] = []
    for path in paths:
        metadata = metadata_by_path.get(str(path))
        if metadata is None:
            raise ValueError(f"Missing metadata for {path}")
        try:
            rows_with_sort.append(build_manifest_entry(day_dir, stream_id, device, path, metadata))
        except ValueError as error:
            if str(error) != "Could not determine capture time from trusted EXIF metadata":
                raise
            invalid_capture_paths.append(path.relative_to(day_dir).as_posix())
    if invalid_capture_paths:
        raise ValueError(
            f"Missing trusted EXIF capture time for {len(invalid_capture_paths)} photo file(s): "
            + ", ".join(invalid_capture_paths)
        )
    return rows_with_sort


def finalize_manifest_rows(rows_with_sort: Sequence[tuple[tuple[object, str, str], Dict[str, str]]]) -> List[Dict[str, str]]:
    rows = [dict(row) for _sort_key, row in sorted(rows_with_sort, key=lambda item: item[0])]
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)
    return rows


def build_manifest_rows(
    day_dir: Path,
    stream_id: str,
    device: str,
    metadata_by_path: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, str]]:
    files = collect_source_files(day_dir)
    if not files:
        raise ValueError(f"No photo files found under {day_dir}")
    rows_with_sort = build_batch_manifest_rows(day_dir, stream_id, device, files, metadata_by_path)
    return finalize_manifest_rows(rows_with_sort)


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
    partial_path = partial_output_path(output_path)
    remove_stale_partial_output(partial_path)
    rows_with_sort: List[tuple[tuple[object, str, str], Dict[str, str]]] = []
    with Progress(
        *build_progress_columns(),
        expand=False,
        console=console,
    ) as progress:
        discover_task = progress.add_task("Discover photos".ljust(25), total=len(files))
        progress.update(discover_task, completed=len(files))
        metadata_task = progress.add_task("Read EXIF metadata".ljust(25), total=len(files))
        rows_task = progress.add_task("Build manifest rows".ljust(25), total=len(files))
        partial_handle, partial_writer = open_partial_manifest_csv(partial_path)
        try:
            for batch_start in range(0, len(files), EXIFTOOL_BATCH_SIZE):
                batch = files[batch_start:batch_start + EXIFTOOL_BATCH_SIZE]
                metadata_items = run_exiftool_batch(
                    batch,
                    progress_callback=lambda current, total: progress.update(
                        metadata_task,
                        completed=batch_start + current,
                    ),
                )
                batch_metadata_by_path = metadata_by_source_path(metadata_items)
                batch_rows_with_sort = build_batch_manifest_rows(day_dir, stream_id, device, batch, batch_metadata_by_path)
                rows_with_sort.extend(batch_rows_with_sort)
                append_partial_manifest_rows(
                    partial_handle,
                    partial_writer,
                    [row for _sort_key, row in batch_rows_with_sort],
                )
                progress.update(metadata_task, completed=batch_start + len(batch))
            rows = finalize_manifest_rows(rows_with_sort)
            rewrite_partial_manifest_csv(partial_path, rows)
            progress.update(rows_task, completed=len(files))
        finally:
            partial_handle.close()
    replace_output_from_partial(partial_path, output_path)
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
