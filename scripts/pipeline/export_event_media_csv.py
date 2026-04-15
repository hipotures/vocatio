#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
from rich.table import Table


from lib.workspace_dir import resolve_workspace_dir
console = Console()

PHOTO_EXTENSIONS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mts"}
DAY_PATTERN = re.compile(r"^\d{8}$")
STREAM_PATTERN = re.compile(r"^(?P<prefix>[pv])-(?P<device>[A-Za-z0-9._-]+)$")
EXIFTOOL_BATCH_SIZE = 1000
PHOTO_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<sequence>\d+)_(?P<embedded_size>\d+)\.[^.]+$",
    re.IGNORECASE,
)
VIDEO_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<width>\d+)x(?P<height>\d+)_(?P<fps>\d+)fps_(?P<embedded_size>\d+)\.[^.]+$",
    re.IGNORECASE,
)

PHOTO_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_dir",
    "path",
    "filename",
    "extension",
    "start_local",
    "start_epoch_ms",
    "timestamp_source",
    "model",
    "make",
    "sequence",
    "embedded_size_bytes",
    "actual_size_bytes",
    "create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]

VIDEO_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_dir",
    "path",
    "filename",
    "extension",
    "start_local",
    "end_local",
    "start_epoch_ms",
    "end_epoch_ms",
    "duration_seconds",
    "timestamp_source",
    "model",
    "make",
    "width",
    "height",
    "fps",
    "embedded_size_bytes",
    "actual_size_bytes",
    "create_date_raw",
    "track_create_date_raw",
    "media_create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export normalized media metadata for a single event day into per-stream CSV files."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory where CSV files will be written. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help='Optional stream IDs to rescan, for example "p-a7r5" "v-gh7"',
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List detected stream IDs and exit",
    )
    return parser.parse_args()


def parse_local_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace("T", " ")
    formats = [
        "%Y:%m:%d %H:%M:%S.%f%z",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    return None


def dt_to_iso(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="milliseconds" if value.microsecond else "seconds")


def dt_to_epoch_ms(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return str(int(value.timestamp() * 1000))


def detect_streams(day_dir: Path) -> Dict[str, Dict[str, str]]:
    streams: Dict[str, Dict[str, str]] = {}
    for path in sorted(day_dir.iterdir()):
        if not path.is_dir():
            continue
        match = STREAM_PATTERN.match(path.name)
        if not match:
            continue
        media_type = "photo" if match.group("prefix") == "p" else "video"
        streams[path.name] = {
            "stream_id": path.name,
            "device": match.group("device"),
            "media_type": media_type,
            "source_dir": str(path),
        }
    return streams


def collect_files(stream_dir: Path, media_type: str) -> List[Path]:
    allowed = PHOTO_EXTENSIONS if media_type == "photo" else VIDEO_EXTENSIONS
    files: List[Path] = []
    for path in sorted(stream_dir.iterdir()):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        files.append(path)
    return files


def build_photo_rows(
    day: str,
    stream_id: str,
    device: str,
    source_dir: str,
    files: Sequence[Path],
    metadata: Sequence[Dict[str, object]],
    on_file_processed: Optional[Callable[[], None]] = None,
) -> List[Dict[str, str]]:
    meta_by_path = {str(item["SourceFile"]): item for item in metadata}
    rows: List[Dict[str, str]] = []
    for path in files:
        item = meta_by_path.get(str(path), {})
        parsed = PHOTO_PATTERN.match(path.name)
        start, timestamp_source = pick_photo_start(item, path)
        embedded_size = parsed.group("embedded_size") if parsed else ""
        sequence = parsed.group("sequence") if parsed else ""
        try:
            actual_size = str(path.stat().st_size)
        except OSError:
            actual_size = ""
        rows.append(
            {
                "day": day,
                "stream_id": stream_id,
                "device": device,
                "media_type": "photo",
                "source_dir": source_dir,
                "path": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "start_local": dt_to_iso(start),
                "start_epoch_ms": dt_to_epoch_ms(start),
                "timestamp_source": timestamp_source,
                "model": str(item.get("Model") or ""),
                "make": str(item.get("Make") or ""),
                "sequence": sequence,
                "embedded_size_bytes": embedded_size,
                "actual_size_bytes": actual_size,
                "create_date_raw": str(item.get("CreateDate") or ""),
                "datetime_original_raw": str(item.get("DateTimeOriginal") or ""),
                "subsec_datetime_original_raw": str(item.get("SubSecDateTimeOriginal") or ""),
                "subsec_create_date_raw": str(item.get("SubSecCreateDate") or ""),
                "file_modify_date_raw": str(item.get("FileModifyDate") or ""),
                "file_create_date_raw": str(item.get("FileCreateDate") or ""),
            }
        )
        if on_file_processed is not None:
            on_file_processed()
    rows.sort(key=lambda row: (row["start_local"], row["filename"]))
    return rows


def run_exiftool(
    paths: Sequence[Path],
    on_batch_processed: Optional[Callable[[int], None]] = None,
) -> List[Dict[str, object]]:
    if not paths:
        return []
    base_cmd = [
        "exiftool",
        "-api",
        "QuickTimeUTC=1",
        "-json",
        "-n",
        "-CreateDate",
        "-TrackCreateDate",
        "-MediaCreateDate",
        "-DateTimeOriginal",
        "-SubSecDateTimeOriginal",
        "-SubSecCreateDate",
        "-FileModifyDate",
        "-FileCreateDate",
        "-Duration",
        "-Model",
        "-Make",
    ]
    items: List[Dict[str, object]] = []
    for index in range(0, len(paths), EXIFTOOL_BATCH_SIZE):
        batch = paths[index:index + EXIFTOOL_BATCH_SIZE]
        cmd = list(base_cmd)
        cmd.extend(str(path) for path in batch)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items.extend(json.loads(result.stdout))
        if on_batch_processed is not None:
            on_batch_processed(len(batch))
    return items


def file_mtime_datetime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def pick_photo_start(item: Dict[str, object], path: Path) -> Tuple[Optional[datetime], str]:
    candidates = [
        ("subsec_create_date", item.get("SubSecCreateDate")),
        ("subsec_datetime_original", item.get("SubSecDateTimeOriginal")),
        ("create_date", item.get("CreateDate")),
        ("datetime_original", item.get("DateTimeOriginal")),
        ("file_create_date", item.get("FileCreateDate")),
        ("file_modify_date", item.get("FileModifyDate")),
    ]
    for label, raw in candidates:
        if raw is None:
            continue
        dt = parse_local_datetime(str(raw))
        if dt is not None:
            return dt, label
    return file_mtime_datetime(path), "file_mtime"


def pick_video_start(item: Dict[str, object]) -> Tuple[Optional[datetime], str]:
    candidates = [
        ("track_create_date", item.get("TrackCreateDate")),
        ("create_date", item.get("CreateDate")),
        ("media_create_date", item.get("MediaCreateDate")),
        ("subsec_datetime_original", item.get("SubSecDateTimeOriginal")),
        ("datetime_original", item.get("DateTimeOriginal")),
        ("file_create_date", item.get("FileCreateDate")),
        ("file_modify_date", item.get("FileModifyDate")),
    ]
    for label, raw in candidates:
        if raw is None:
            continue
        dt = parse_local_datetime(str(raw))
        if dt is not None:
            return dt, label
    return None, ""


def build_video_rows(
    day: str,
    stream_id: str,
    device: str,
    source_dir: str,
    files: Sequence[Path],
    metadata: Sequence[Dict[str, object]],
    on_file_processed: Optional[Callable[[], None]] = None,
) -> List[Dict[str, str]]:
    meta_by_path = {str(item["SourceFile"]): item for item in metadata}
    rows: List[Dict[str, str]] = []
    for path in files:
        item = meta_by_path.get(str(path), {})
        metadata_start, metadata_timestamp_source = pick_video_start(item)
        if metadata_start is not None:
            start = metadata_start
            timestamp_source = metadata_timestamp_source
        else:
            start = file_mtime_datetime(path)
            timestamp_source = "file_mtime"
        duration_raw = item.get("Duration")
        try:
            duration_seconds = float(duration_raw) if duration_raw is not None else 0.0
        except (TypeError, ValueError):
            duration_seconds = 0.0
        end = start + timedelta(seconds=duration_seconds) if start is not None else None
        parsed = VIDEO_PATTERN.match(path.name)
        try:
            actual_size = str(path.stat().st_size)
        except OSError:
            actual_size = ""
        rows.append(
            {
                "day": day,
                "stream_id": stream_id,
                "device": device,
                "media_type": "video",
                "source_dir": source_dir,
                "path": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "start_local": dt_to_iso(start),
                "end_local": dt_to_iso(end),
                "start_epoch_ms": dt_to_epoch_ms(start),
                "end_epoch_ms": dt_to_epoch_ms(end),
                "duration_seconds": f"{duration_seconds:.3f}",
                "timestamp_source": timestamp_source,
                "model": str(item.get("Model") or ""),
                "make": str(item.get("Make") or ""),
                "width": parsed.group("width") if parsed else "",
                "height": parsed.group("height") if parsed else "",
                "fps": parsed.group("fps") if parsed else "",
                "embedded_size_bytes": parsed.group("embedded_size") if parsed else "",
                "actual_size_bytes": actual_size,
                "create_date_raw": str(item.get("CreateDate") or ""),
                "track_create_date_raw": str(item.get("TrackCreateDate") or ""),
                "media_create_date_raw": str(item.get("MediaCreateDate") or ""),
                "datetime_original_raw": str(item.get("DateTimeOriginal") or ""),
                "subsec_datetime_original_raw": str(item.get("SubSecDateTimeOriginal") or ""),
                "file_modify_date_raw": str(item.get("FileModifyDate") or ""),
                "file_create_date_raw": str(item.get("FileCreateDate") or ""),
            }
        )
        if on_file_processed is not None:
            on_file_processed()
    rows.sort(key=lambda row: (row["start_local"], row["filename"]))
    return rows


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def build_summary_table(summary_rows: Sequence[Tuple[str, str, str, int, str, str]]) -> Table:
    table = Table(title="Export Summary", expand=False)
    table.add_column("Day", style="cyan")
    table.add_column("Stream", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Rows", justify="right", style="yellow")
    table.add_column("First", style="white")
    table.add_column("Last", style="white")
    for row in summary_rows:
        table.add_row(*[str(item) for item in row])
    return table


def selected_streams(
    streams: Dict[str, Dict[str, str]],
    targets: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    if not targets:
        return [streams[key] for key in sorted(streams)]
    missing = [target for target in targets if target not in streams]
    if missing:
        console.print(f"[red]Error: unknown targets: {', '.join(missing)}[/red]")
        sys.exit(1)
    return [streams[target] for target in targets]


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    day = day_dir.name
    streams = detect_streams(day_dir)
    if not streams:
        console.print(f"[red]Error: no p-/v- streams found in {day_dir}.[/red]")
        return 1

    if args.list_targets:
        for stream_id in sorted(streams):
            info = streams[stream_id]
            console.print(f"{stream_id}  {info['media_type']}  {info['source_dir']}")
        return 0

    streams_to_process = selected_streams(streams, args.targets)
    file_counts = {
        info["stream_id"]: len(collect_files(Path(info["source_dir"]), info["media_type"]))
        for info in streams_to_process
    }
    total_files = sum(file_counts.values())

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )

    summary_rows: List[Tuple[str, str, str, int, str, str]] = []

    with progress:
        stream_task = progress.add_task("Streams".ljust(25), total=len(streams_to_process))
        file_task = progress.add_task("Files".ljust(25), total=max(total_files, 1))
        for info in streams_to_process:
            stream_dir = Path(info["source_dir"])
            files = collect_files(stream_dir, info["media_type"])
            progress.update(
                file_task,
                description=f"Files ({info['stream_id']})".ljust(25),
            )
            if info["media_type"] == "photo":
                metadata = run_exiftool(
                    files,
                    on_batch_processed=lambda count, task_id=file_task: progress.advance(task_id, count),
                )
                rows = build_photo_rows(
                    day,
                    info["stream_id"],
                    info["device"],
                    info["source_dir"],
                    files,
                    metadata,
                )
                headers = PHOTO_HEADERS
                first = rows[0]["start_local"] if rows else ""
                last = rows[-1]["start_local"] if rows else ""
            else:
                metadata = run_exiftool(
                    files,
                    on_batch_processed=lambda count, task_id=file_task: progress.advance(task_id, count),
                )
                rows = build_video_rows(
                    day,
                    info["stream_id"],
                    info["device"],
                    info["source_dir"],
                    files,
                    metadata,
                )
                headers = VIDEO_HEADERS
                first = rows[0]["start_local"] if rows else ""
                last = rows[-1]["end_local"] if rows else ""

            csv_path = workspace_dir / f"{info['stream_id']}.csv"
            count = write_csv(csv_path, headers, rows)
            summary_rows.append((day, info["stream_id"], info["media_type"], count, first, last))
            progress.advance(stream_task)

    summary_csv = workspace_dir / "summary.csv"
    write_csv(
        summary_csv,
        ["day", "stream_id", "media_type", "rows", "first", "last"],
        [
            {
                "day": day_value,
                "stream_id": stream_id,
                "media_type": media_type,
                "rows": str(count),
                "first": first,
                "last": last,
            }
            for day_value, stream_id, media_type, count, first, last in summary_rows
        ],
    )

    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote CSV files to {workspace_dir}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
