#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS
from lib.photo_time_order import (
    CaptureTimeParts,
    format_capture_subsec,
    format_capture_time_local,
    format_start_epoch_ms,
    format_start_local,
    normalize_sort_datetime,
    parse_exif_datetime,
    pick_capture_time_parts,
)


console = Console()
DAY_PATTERN = re.compile(r"^\d{8}$")
STREAM_PATTERN = re.compile(r"^(?P<prefix>[pv])-(?P<device>[A-Za-z0-9._-]+)$")
PHOTO_FALLBACK_TIME_FIELDS: Sequence[tuple[str, str]] = (
    ("SubSecCreateDate", "subsec_create_date"),
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("CreateDate", "create_date"),
    ("DateTimeOriginal", "datetime_original"),
    ("FileCreateDate", "file_create_date"),
    ("FileModifyDate", "file_modify_date"),
)


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one canonical media manifest for a single event day."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory where the CSV file will be written. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output",
        default="media_manifest.csv",
        help="Output CSV filename or absolute path. Default: media_manifest.csv",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help='Optional stream IDs to scan, for example "p-a7r5" "v-gh7"',
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List detected stream IDs and exit",
    )
    parser.add_argument(
        "--media-types",
        choices=["all", "photo", "video"],
        default="all",
        help="Media types to include. Default: all",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int_arg,
        default=4,
        help="Number of worker jobs. Default: 4",
    )
    return parser.parse_args(argv)


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


def empty_media_row() -> Dict[str, str]:
    return {header: "" for header in MEDIA_MANIFEST_HEADERS}


def metadata_text(metadata: Optional[Mapping[str, object]], key: str) -> str:
    if metadata is None:
        return ""
    return str(metadata.get(key) or "")


def capture_time_parts(
    value: datetime,
    capture_subsec: str,
    timestamp_source: str,
    aware_value: Optional[datetime] = None,
) -> CaptureTimeParts:
    return CaptureTimeParts(
        capture_time_local=format_capture_time_local(value),
        capture_subsec=format_capture_subsec(capture_subsec),
        timestamp_source=timestamp_source,
        start_local=format_start_local(value),
        start_epoch_ms=format_start_epoch_ms(value, aware_value),
        sort_dt=normalize_sort_datetime(value, aware_value),
    )


def capture_subsec_from_datetime(value: datetime) -> str:
    return f"{value.microsecond:06d}".rstrip("0")


def safe_path_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def actual_size_bytes_text(path_stat: Optional[os.stat_result]) -> str:
    if path_stat is None:
        return ""
    return str(path_stat.st_size)


def file_mtime_datetime(path_stat: Optional[os.stat_result]) -> Optional[datetime]:
    if path_stat is None:
        return None
    return datetime.fromtimestamp(path_stat.st_mtime)


def pick_fallback_capture_time_parts(
    metadata: Optional[Mapping[str, object]],
    path_stat: Optional[os.stat_result],
) -> Optional[CaptureTimeParts]:
    if metadata:
        for field_name, source_name in PHOTO_FALLBACK_TIME_FIELDS:
            parsed = parse_exif_datetime(metadata.get(field_name))
            if parsed is None:
                continue
            return capture_time_parts(
                value=parsed.local_dt,
                capture_subsec=parsed.fraction,
                timestamp_source=source_name,
                aware_value=parsed.aware_dt,
            )
    mtime = file_mtime_datetime(path_stat)
    if mtime is None:
        return None
    return capture_time_parts(
        value=mtime,
        capture_subsec=capture_subsec_from_datetime(mtime),
        timestamp_source="file_mtime",
    )


def build_photo_manifest_entry(
    day_dir: Path,
    stream_id: str,
    device: str,
    path: Path,
    metadata: Optional[Mapping[str, object]],
) -> tuple[tuple[datetime, str, str], Dict[str, str]]:
    relative_path = path.relative_to(day_dir).as_posix()
    source_dir = path.parent
    source_rel_dir = source_dir.relative_to(day_dir).as_posix()
    path_stat = safe_path_stat(path)
    row = empty_media_row()
    row.update(
        {
            "day": day_dir.name,
            "stream_id": stream_id,
            "device": device,
            "media_type": "photo",
            "source_root": str(day_dir),
            "source_dir": str(source_dir),
            "source_rel_dir": "" if source_rel_dir == "." else source_rel_dir,
            "path": str(path),
            "relative_path": relative_path,
            "media_id": relative_path,
            "photo_id": relative_path,
            "filename": relative_path,
            "extension": path.suffix.lower(),
            "model": metadata_text(metadata, "Model"),
            "make": metadata_text(metadata, "Make"),
            "actual_size_bytes": actual_size_bytes_text(path_stat),
            "create_date_raw": metadata_text(metadata, "CreateDate"),
            "datetime_original_raw": metadata_text(metadata, "DateTimeOriginal"),
            "subsec_datetime_original_raw": metadata_text(metadata, "SubSecDateTimeOriginal"),
            "subsec_create_date_raw": metadata_text(metadata, "SubSecCreateDate"),
            "file_modify_date_raw": metadata_text(metadata, "FileModifyDate"),
            "file_create_date_raw": metadata_text(metadata, "FileCreateDate"),
        }
    )
    if not metadata:
        time_parts = pick_fallback_capture_time_parts(metadata, path_stat)
        row["metadata_status"] = "error"
        row["metadata_error"] = "Missing metadata"
    else:
        try:
            time_parts = pick_capture_time_parts(metadata)
            row["metadata_status"] = "ok"
            row["metadata_error"] = ""
        except ValueError as error:
            time_parts = pick_fallback_capture_time_parts(metadata, path_stat)
            row["metadata_status"] = "partial"
            row["metadata_error"] = str(error)

    row.update(
        {
            "capture_time_local": time_parts.capture_time_local if time_parts is not None else "",
            "capture_subsec": time_parts.capture_subsec if time_parts is not None else "",
            "start_local": time_parts.start_local if time_parts is not None else "",
            "start_epoch_ms": time_parts.start_epoch_ms if time_parts is not None else "",
            "timestamp_source": time_parts.timestamp_source if time_parts is not None else "",
        }
    )
    sort_dt = time_parts.sort_dt if time_parts is not None else datetime.max
    return (sort_dt, row["capture_subsec"], relative_path), row


def assign_photo_order_indexes(rows: Sequence[Dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)


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


def filter_streams_by_media_types(
    streams: Dict[str, Dict[str, str]],
    media_types: str,
) -> Dict[str, Dict[str, str]]:
    if media_types == "all":
        return dict(sorted(streams.items()))
    return {
        stream_id: info
        for stream_id, info in sorted(streams.items())
        if info["media_type"] == media_types
    }


def selected_streams(
    streams: Dict[str, Dict[str, str]],
    targets: Optional[Sequence[str]],
) -> List[Dict[str, str]] | None:
    if not targets:
        return [streams[key] for key in sorted(streams)]
    missing = [target for target in targets if target not in streams]
    if missing:
        console.print(f"[red]Error: unknown targets: {', '.join(missing)}[/red]")
        return None
    return [streams[target] for target in targets]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    streams = detect_streams(day_dir)
    if not streams:
        console.print(f"[red]Error: no p-/v- streams found in {day_dir}.[/red]")
        return 1

    streams_to_process = selected_streams(streams, args.targets)
    if streams_to_process is None:
        return 1

    if args.media_types != "all":
        streams_to_process = [info for info in streams_to_process if info["media_type"] == args.media_types]
        if not streams_to_process:
            if args.targets:
                console.print(
                    "[red]Error: requested targets matched no streams for "
                    f"media-types={args.media_types}: {', '.join(args.targets)}[/red]"
                )
            else:
                console.print(f"[red]Error: no streams matched media-types={args.media_types} in {day_dir}.[/red]")
            return 1

    if args.list_targets:
        for info in streams_to_process:
            console.print(f"{info['stream_id']}  {info['media_type']}  {info['source_dir']}")
        return 0
    console.print(
        "[red]Error: export orchestration is not implemented yet. Use --list-targets to inspect detected streams.[/red]"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
