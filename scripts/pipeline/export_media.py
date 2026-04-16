#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import signal
import subprocess
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

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

from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS
from lib.media_manifest import read_media_manifest
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
from lib.workspace_dir import resolve_workspace_dir as resolve_configured_workspace_dir


console = Console()
DAY_PATTERN = re.compile(r"^\d{8}$")
STREAM_PATTERN = re.compile(r"^(?P<prefix>[pv])-(?P<device>[A-Za-z0-9._-]+)$")
VIDEO_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<width>\d+)x(?P<height>\d+)_(?P<fps>\d+)fps_(?P<embedded_size>\d+)\.[^.]+$",
    re.IGNORECASE,
)
PHOTO_FALLBACK_TIME_FIELDS: Sequence[tuple[str, str]] = (
    ("SubSecCreateDate", "subsec_create_date"),
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("CreateDate", "create_date"),
    ("DateTimeOriginal", "datetime_original"),
    ("FileCreateDate", "file_create_date"),
    ("FileModifyDate", "file_modify_date"),
)
VIDEO_FALLBACK_DURATION_TEXT = "0"
VIDEO_FALLBACK_DIMENSION_TEXT = "1"
VIDEO_FALLBACK_FPS_TEXT = "1"
VIDEO_TIME_FIELDS: Sequence[tuple[str, str]] = (
    ("TrackCreateDate", "track_create_date"),
    ("CreateDate", "create_date"),
    ("MediaCreateDate", "media_create_date"),
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("DateTimeOriginal", "datetime_original"),
    ("FileCreateDate", "file_create_date"),
    ("FileModifyDate", "file_modify_date"),
)
PHOTO_EXTENSIONS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mts"}
EXIFTOOL_BATCH_SIZE = 1000
PROCESS_PROGRESS_BATCH_SIZE = 100
DISCOVERY_PROGRESS_BATCH_SIZE = 1000


class GracefulInterruptState:
    def __init__(self) -> None:
        self.stop_requested = False
        self.signal_count = 0

    def handle(self, _signum, _frame) -> None:
        self.signal_count += 1
        if self.signal_count == 1:
            self.stop_requested = True
            console.print(
                "[yellow]Interrupt received. Finishing active batch(es) before exit. "
                "Press Ctrl+C again to abort immediately.[/yellow]"
            )
            return
        raise KeyboardInterrupt


def install_interrupt_handler() -> tuple[GracefulInterruptState, object]:
    state = GracefulInterruptState()
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, state.handle)
    return state, previous_handler


def restore_interrupt_handler(previous_handler: object) -> None:
    signal.signal(signal.SIGINT, previous_handler)


class ExiftoolBatchError(RuntimeError):
    def __init__(
        self,
        error: subprocess.CalledProcessError,
        processed_items: Sequence[Dict[str, object]],
        remaining_paths: Sequence[Path],
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.processed_items = [dict(item) for item in processed_items]
        self.remaining_paths = list(remaining_paths)


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
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing partial/final manifest state and rebuild from scratch.",
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
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value)


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


def placeholder_photo_capture_time_parts(day_dir: Path) -> CaptureTimeParts:
    if DAY_PATTERN.match(day_dir.name):
        placeholder_dt = datetime.strptime(day_dir.name, "%Y%m%d")
    else:
        placeholder_dt = datetime(1970, 1, 1)
    return capture_time_parts(
        value=placeholder_dt,
        capture_subsec="0",
        timestamp_source="placeholder",
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
        if time_parts is None:
            time_parts = placeholder_photo_capture_time_parts(day_dir)
        row["metadata_status"] = "error"
        row["metadata_error"] = "Missing metadata"
    else:
        try:
            time_parts = pick_capture_time_parts(metadata)
            row["metadata_status"] = "ok"
            row["metadata_error"] = ""
        except ValueError as error:
            time_parts = pick_fallback_capture_time_parts(metadata, path_stat)
            if time_parts is None:
                time_parts = placeholder_photo_capture_time_parts(day_dir)
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


def parse_video_start_fields(
    metadata: Optional[Mapping[str, object]],
) -> tuple[Optional[datetime], Optional[datetime], str, List[str]]:
    errors: List[str] = []
    if not metadata:
        return None, None, "", errors

    saw_candidate = False
    for field_name, source_name in VIDEO_TIME_FIELDS:
        raw_value = metadata.get(field_name)
        if raw_value in (None, ""):
            continue
        saw_candidate = True
        parsed = parse_exif_datetime(raw_value)
        if parsed is None:
            errors.append(f"Invalid {field_name}")
            continue
        return parsed.local_dt, parsed.aware_dt, source_name, errors

    if saw_candidate:
        errors.append("Could not determine video start time from metadata")
    else:
        errors.append("Missing video timestamp metadata")
    return None, None, "", errors


def parse_optional_float_text(
    metadata: Optional[Mapping[str, object]],
    key: str,
    *,
    allow_zero: bool = True,
) -> tuple[str, Optional[float], Optional[str]]:
    text = metadata_text(metadata, key).strip()
    if not text:
        return "", None, None
    try:
        value = float(text)
    except ValueError:
        return "", None, f"Invalid {key}"
    if not math.isfinite(value):
        return "", None, f"Invalid {key}"
    if value < 0 or (not allow_zero and value == 0):
        return "", None, f"Invalid {key}"
    return str(value), value, None


def parse_optional_positive_int_text(
    metadata: Optional[Mapping[str, object]],
    key: str,
) -> tuple[str, Optional[int], Optional[str]]:
    text = metadata_text(metadata, key).strip()
    if not text:
        return "", None, None
    try:
        value = int(text)
    except ValueError:
        try:
            value_float = float(text)
        except ValueError:
            return "", None, f"Invalid {key}"
        if not math.isfinite(value_float):
            return "", None, f"Invalid {key}"
        if not value_float.is_integer():
            return "", None, f"Invalid {key}"
        value = int(value_float)
    if value <= 0:
        return "", None, f"Invalid {key}"
    return str(value), value, None


def parse_video_filename_parts(path: Path) -> tuple[Optional[datetime], Dict[str, str], List[str]]:
    match = VIDEO_PATTERN.match(path.name)
    if match is None:
        return None, {}, []

    errors: List[str] = []
    start_dt: Optional[datetime] = None
    try:
        start_dt = datetime.strptime(
            f"{match.group('date')}_{match.group('time')}",
            "%Y%m%d_%H%M%S",
        )
    except ValueError:
        errors.append("Invalid filename timestamp")

    return (
        start_dt,
        {
            "width": match.group("width"),
            "height": match.group("height"),
            "fps": match.group("fps"),
            "embedded_size_bytes": match.group("embedded_size"),
        },
        errors,
    )


def placeholder_video_start_datetime(day_dir: Path) -> datetime:
    if DAY_PATTERN.match(day_dir.name):
        return datetime.strptime(day_dir.name, "%Y%m%d")
    return datetime(1970, 1, 1)


def build_video_manifest_entry(
    day_dir: Path,
    stream_id: str,
    device: str,
    path: Path,
    metadata: Optional[Mapping[str, object]],
) -> Dict[str, str]:
    relative_path = path.relative_to(day_dir).as_posix()
    source_dir = path.parent
    source_rel_dir = source_dir.relative_to(day_dir).as_posix()
    path_stat = safe_path_stat(path)
    filename_start_dt, filename_parts, filename_errors = parse_video_filename_parts(path)
    row = empty_media_row()
    row.update(
        {
            "day": day_dir.name,
            "stream_id": stream_id,
            "device": device,
            "media_type": "video",
            "source_root": str(day_dir),
            "source_dir": str(source_dir),
            "source_rel_dir": "" if source_rel_dir == "." else source_rel_dir,
            "path": str(path),
            "relative_path": relative_path,
            "media_id": relative_path,
            "photo_id": "",
            "filename": relative_path,
            "extension": path.suffix.lower(),
            "model": metadata_text(metadata, "Model"),
            "make": metadata_text(metadata, "Make"),
            "embedded_size_bytes": filename_parts.get("embedded_size_bytes", ""),
            "actual_size_bytes": actual_size_bytes_text(path_stat),
            "create_date_raw": metadata_text(metadata, "CreateDate"),
            "track_create_date_raw": metadata_text(metadata, "TrackCreateDate"),
            "media_create_date_raw": metadata_text(metadata, "MediaCreateDate"),
            "datetime_original_raw": metadata_text(metadata, "DateTimeOriginal"),
            "subsec_datetime_original_raw": metadata_text(metadata, "SubSecDateTimeOriginal"),
            "subsec_create_date_raw": metadata_text(metadata, "SubSecCreateDate"),
            "file_modify_date_raw": metadata_text(metadata, "FileModifyDate"),
            "file_create_date_raw": metadata_text(metadata, "FileCreateDate"),
        }
    )

    error_messages: List[str] = []
    if not metadata:
        error_messages.append("Missing metadata")

    start_dt, aware_start_dt, timestamp_source, start_errors = parse_video_start_fields(metadata)
    error_messages.extend(start_errors)

    if start_dt is None:
        if filename_start_dt is not None:
            start_dt = filename_start_dt
            aware_start_dt = None
            timestamp_source = "filename_pattern"
            error_messages.append("Used filename-derived start time")
        else:
            error_messages.extend(filename_errors)
            mtime = file_mtime_datetime(path_stat)
            if mtime is not None:
                start_dt = mtime
                aware_start_dt = None
                timestamp_source = "file_mtime"
                error_messages.append("Used file_mtime start time")
            else:
                start_dt = placeholder_video_start_datetime(day_dir)
                aware_start_dt = None
                timestamp_source = "placeholder"
                error_messages.append("Used placeholder start time")

    duration_text, duration_value, duration_error = parse_optional_float_text(metadata, "Duration")
    width_text, _, width_error = parse_optional_positive_int_text(metadata, "ImageWidth")
    height_text, _, height_error = parse_optional_positive_int_text(metadata, "ImageHeight")
    fps_text, _, fps_error = parse_optional_float_text(metadata, "VideoFrameRate", allow_zero=False)

    for error in (duration_error, width_error, height_error, fps_error):
        if error is not None:
            error_messages.append(error)

    if duration_value is None:
        duration_value = 0.0
        duration_text = VIDEO_FALLBACK_DURATION_TEXT
        error_messages.append("Used zero duration fallback")

    if not width_text:
        if filename_parts.get("width"):
            width_text = filename_parts["width"]
            error_messages.append("Used filename-derived width")
        else:
            width_text = VIDEO_FALLBACK_DIMENSION_TEXT
            error_messages.append("Used placeholder width")

    if not height_text:
        if filename_parts.get("height"):
            height_text = filename_parts["height"]
            error_messages.append("Used filename-derived height")
        else:
            height_text = VIDEO_FALLBACK_DIMENSION_TEXT
            error_messages.append("Used placeholder height")

    if not fps_text:
        if filename_parts.get("fps"):
            fps_text = filename_parts["fps"]
            error_messages.append("Used filename-derived fps")
        else:
            fps_text = VIDEO_FALLBACK_FPS_TEXT
            error_messages.append("Used placeholder fps")

    row["start_local"] = format_start_local(start_dt)
    row["start_epoch_ms"] = format_start_epoch_ms(start_dt, aware_start_dt)
    row["timestamp_source"] = timestamp_source

    end_dt = start_dt + timedelta(seconds=duration_value)
    aware_end_dt = aware_start_dt + timedelta(seconds=duration_value) if aware_start_dt is not None else None
    row["end_local"] = format_start_local(end_dt)
    row["end_epoch_ms"] = format_start_epoch_ms(end_dt, aware_end_dt)

    row["duration_seconds"] = duration_text
    row["width"] = width_text
    row["height"] = height_text
    row["fps"] = fps_text

    if not metadata:
        row["metadata_status"] = "error"
        row["metadata_error"] = "; ".join(error_messages)
    elif error_messages:
        row["metadata_status"] = "partial"
        row["metadata_error"] = "; ".join(error_messages)
    else:
        row["metadata_status"] = "ok"
        row["metadata_error"] = ""
    return row


def write_media_manifest_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    row_list = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
            writer.writeheader()
            writer.writerows(row_list)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def partial_output_path(output_path: Path) -> Path:
    return Path(f"{output_path}.partial")


def assign_photo_order_indexes(rows: Sequence[Dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)


def parse_int_or_fallback(value: str, fallback: int) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return fallback


def photo_snapshot_sort_key(row: Mapping[str, str]) -> tuple[int, int, str]:
    return (
        parse_int_or_fallback(str(row.get("start_epoch_ms") or ""), 2**63 - 1),
        parse_int_or_fallback(str(row.get("capture_subsec") or ""), 0),
        str(row.get("relative_path") or ""),
    )


def build_manifest_snapshot(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    photo_rows = sorted(
        (dict(row) for row in rows if str(row.get("media_type") or "").strip() == "photo"),
        key=photo_snapshot_sort_key,
    )
    assign_photo_order_indexes(photo_rows)
    video_rows = [
        dict(row)
        for row in rows
        if str(row.get("media_type") or "").strip() == "video"
    ]
    snapshot_rows = photo_rows + video_rows
    snapshot_rows.sort(key=final_manifest_sort_key)
    return snapshot_rows


def persist_manifest_snapshot(
    output_path: Path,
    rows_by_relative_path: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, str]]:
    snapshot_rows = build_manifest_snapshot(rows_by_relative_path.values())
    partial_path = partial_output_path(output_path)
    write_media_manifest_csv(partial_path, snapshot_rows)
    write_media_manifest_csv(output_path, snapshot_rows)
    return snapshot_rows


def finalize_successful_export(output_path: Path) -> None:
    partial_output_path(output_path).unlink(missing_ok=True)


def load_existing_manifest(path: Path, label: str) -> List[Dict[str, str]] | None:
    if not path.exists():
        return []
    if path.is_dir():
        console.print(f"[red]Error: {label} path is a directory: {path}[/red]")
        return None
    if path.stat().st_size == 0:
        console.print(
            f"[red]Error: {label} is empty: {path}. Use --restart to rebuild from scratch.[/red]"
        )
        return None
    try:
        return read_media_manifest(path)
    except Exception as error:
        console.print(
            f"[red]Error: failed to read {label} {path}: {error}. "
            "Use --restart to rebuild from scratch.[/red]"
        )
        return None


def initialize_resume_rows(output_path: Path, restart: bool) -> List[Dict[str, str]] | None:
    partial_path = partial_output_path(output_path)
    if restart:
        partial_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return []

    partial_rows = load_existing_manifest(partial_path, "partial manifest")
    if partial_rows is None:
        return None
    if partial_rows:
        return partial_rows

    final_rows = load_existing_manifest(output_path, "existing manifest")
    if final_rows is None:
        return None
    if final_rows:
        write_media_manifest_csv(partial_path, final_rows)
        return final_rows
    return []


def build_batch_specs(
    stream_files: Sequence[tuple[Dict[str, str], Sequence[Path]]],
    processed_relative_paths: set[str],
    day_dir: Path,
) -> List[tuple[Dict[str, str], List[Path]]]:
    batch_specs: List[tuple[Dict[str, str], List[Path]]] = []
    for info, files in stream_files:
        pending_files = [
            path
            for path in files
            if path.relative_to(day_dir).as_posix() not in processed_relative_paths
        ]
        for index in range(0, len(pending_files), EXIFTOOL_BATCH_SIZE):
            batch_specs.append((info, pending_files[index:index + EXIFTOOL_BATCH_SIZE]))
    return batch_specs


def build_rows_for_batch(
    day_dir: Path,
    info: Mapping[str, str],
    batch: Sequence[Path],
    on_batch_processed: Optional[Callable[[int], None]] = None,
) -> List[Dict[str, str]]:
    metadata_by_path = extract_metadata_batch_fail_open(
        str(info["stream_id"]),
        batch,
        on_batch_processed=on_batch_processed,
    )
    rows: List[Dict[str, str]] = []
    for path in batch:
        metadata = metadata_by_path.get(str(path))
        if info["media_type"] == "photo":
            _sort_key, row = build_photo_manifest_entry(
                day_dir,
                str(info["stream_id"]),
                str(info["device"]),
                path,
                metadata,
            )
        else:
            row = build_video_manifest_entry(
                day_dir,
                str(info["stream_id"]),
                str(info["device"]),
                path,
                metadata,
            )
        rows.append(row)
    return rows


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


def collect_files(
    stream_dir: Path,
    media_type: str,
    on_files_discovered: Optional[Callable[[int], None]] = None,
) -> List[Path]:
    allowed_extensions = PHOTO_EXTENSIONS if media_type == "photo" else VIDEO_EXTENSIONS
    files: List[Path] = []
    pending_progress_count = 0
    for root, dirnames, filenames in os.walk(stream_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(stream_dir)
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if dirname != "_workspace" and not (root_path / dirname).is_symlink()
        )
        if "_workspace" in relative_root.parts:
            continue
        for filename in sorted(filenames):
            path = root_path / filename
            if path.is_symlink():
                continue
            if path.suffix.lower() not in allowed_extensions:
                continue
            files.append(path)
            if on_files_discovered is not None:
                pending_progress_count += 1
                if pending_progress_count >= DISCOVERY_PROGRESS_BATCH_SIZE:
                    on_files_discovered(pending_progress_count)
                    pending_progress_count = 0
    if on_files_discovered is not None and pending_progress_count:
        on_files_discovered(pending_progress_count)
    return sorted(files, key=lambda path: path.relative_to(stream_dir).as_posix())


def metadata_by_source_path(items: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    output: Dict[str, Mapping[str, object]] = {}
    for item in items:
        source_file = str(item.get("SourceFile") or "").strip()
        if not source_file:
            continue
        output[source_file] = item
    return output


def resolve_workspace_dir(day_dir: Path, workspace_value: Optional[str]) -> Path:
    return resolve_configured_workspace_dir(day_dir, workspace_value)


def resolve_output_path(workspace_dir: Path, output_value: str) -> Path:
    candidate = Path(output_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_dir / candidate).resolve()


def exiftool_base_command() -> List[str]:
    return [
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
        "-ImageWidth",
        "-ImageHeight",
        "-VideoFrameRate",
        "-Model",
        "-Make",
    ]


def run_exiftool(
    paths: Sequence[Path],
    on_batch_processed: Optional[Callable[[int], None]] = None,
) -> List[Dict[str, object]]:
    if not paths:
        return []
    items: List[Dict[str, object]] = []
    for index in range(0, len(paths), PROCESS_PROGRESS_BATCH_SIZE):
        batch = paths[index:index + PROCESS_PROGRESS_BATCH_SIZE]
        cmd = exiftool_base_command()
        cmd.extend(str(path) for path in batch)
        result = subprocess.run(cmd, capture_output=True, text=True, start_new_session=True)
        if result.returncode != 0:
            raise ExiftoolBatchError(
                subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    output=result.stdout,
                    stderr=result.stderr,
                ),
                processed_items=items,
                remaining_paths=paths[index:],
            )
        payload = result.stdout.strip()
        if payload:
            loaded = json.loads(payload)
            if isinstance(loaded, list):
                items.extend(item for item in loaded if isinstance(item, dict))
        if on_batch_processed is not None:
            on_batch_processed(len(batch))
    return items


def warn_metadata_batch_failure(stream_id: str, batch: Sequence[Path], error: Exception) -> None:
    console.print(
        "[yellow]Warning: failed to read metadata for "
        f"{stream_id} batch ({len(batch)} file(s)): {error}[/yellow]"
    )


def warn_metadata_file_failure(stream_id: str, path: Path, error: Exception) -> None:
    console.print(
        "[yellow]Warning: failed to read metadata for "
        f"{stream_id} file {path.name}: {error}[/yellow]"
    )


def extract_metadata_batch_fail_open(
    stream_id: str,
    batch: Sequence[Path],
    on_batch_processed: Optional[Callable[[int], None]] = None,
) -> Dict[str, Mapping[str, object]]:
    try:
        return metadata_by_source_path(
            run_exiftool(batch, on_batch_processed=on_batch_processed)
        )
    except ExiftoolBatchError as error:
        warn_metadata_batch_failure(stream_id, batch, error.error)
        metadata_by_path = metadata_by_source_path(error.processed_items)
        remaining_paths = error.remaining_paths
        if len(remaining_paths) <= 1:
            for path in remaining_paths:
                try:
                    metadata_by_path.update(
                        metadata_by_source_path(
                            run_exiftool([path], on_batch_processed=on_batch_processed)
                        )
                    )
                except Exception as path_error:
                    warn_metadata_file_failure(stream_id, path, path_error)
                    if on_batch_processed is not None:
                        on_batch_processed(1)
            return metadata_by_path

        for path in remaining_paths:
            try:
                metadata_by_path.update(
                    metadata_by_source_path(
                        run_exiftool([path], on_batch_processed=on_batch_processed)
                    )
                )
            except Exception as path_error:
                warn_metadata_file_failure(stream_id, path, path_error)
                if on_batch_processed is not None:
                    on_batch_processed(1)
        return metadata_by_path
    except Exception as error:
        warn_metadata_batch_failure(stream_id, batch, error)
        if len(batch) <= 1:
            if on_batch_processed is not None:
                on_batch_processed(len(batch))
            return {}

    metadata_by_path: Dict[str, Mapping[str, object]] = {}
    for path in batch:
        try:
            metadata_by_path.update(
                metadata_by_source_path(
                    run_exiftool([path], on_batch_processed=on_batch_processed)
                )
            )
        except Exception as error:
            warn_metadata_file_failure(stream_id, path, error)
            if on_batch_processed is not None:
                on_batch_processed(1)
    return metadata_by_path


def extract_metadata_by_stream(
    stream_files: Sequence[tuple[Dict[str, str], Sequence[Path]]],
    jobs: int,
    progress: Progress,
    metadata_task: object,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    metadata_for_streams: Dict[str, Dict[str, Mapping[str, object]]] = {
        info["stream_id"]: {}
        for info, _files in stream_files
    }
    batch_specs = [
        (info["stream_id"], files[index:index + EXIFTOOL_BATCH_SIZE])
        for info, files in stream_files
        for index in range(0, len(files), EXIFTOOL_BATCH_SIZE)
    ]
    if not batch_specs:
        return metadata_for_streams

    if jobs <= 1 or len(batch_specs) <= 1:
        for stream_id, batch in batch_specs:
            metadata_for_streams[stream_id].update(extract_metadata_batch_fail_open(stream_id, batch))
            progress.advance(metadata_task, len(batch))
        return metadata_for_streams

    max_workers = min(jobs, len(batch_specs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(extract_metadata_batch_fail_open, stream_id, batch): (stream_id, batch)
            for stream_id, batch in batch_specs
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            stream_id, batch = future_to_batch[future]
            metadata_for_streams[stream_id].update(future.result())
            progress.advance(metadata_task, len(batch))
    return metadata_for_streams


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


def final_manifest_sort_key(row: Mapping[str, str]) -> tuple[object, ...]:
    if row.get("media_type") == "photo":
        return (
            0,
            int(row.get("photo_order_index") or "0"),
            row.get("relative_path", ""),
        )

    start_epoch_ms_text = str(row.get("start_epoch_ms") or "").strip()
    try:
        start_epoch_ms = int(start_epoch_ms_text)
    except ValueError:
        start_epoch_ms = 2**63 - 1
    return (
        1,
        start_epoch_ms,
        row.get("start_local", ""),
        row.get("stream_id", ""),
        row.get("relative_path", ""),
    )


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
    ordered_unique_targets: List[str] = []
    seen_targets = set()
    for target in targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        ordered_unique_targets.append(target)
    return [streams[target] for target in ordered_unique_targets]


def build_export_context_lines(
    day_dir: Path,
    workspace_dir: Path,
    output_path: Path,
    streams_to_process: Sequence[Mapping[str, str]],
) -> List[str]:
    lines = [
        f"Day directory: {day_dir}",
        f"Workspace directory: {workspace_dir}",
        f"Output manifest: {output_path}",
    ]
    for media_type, label in (("photo", "Photo source directories"), ("video", "Video source directories")):
        source_dirs = sorted(
            {
                str(Path(str(info["source_dir"])).resolve())
                for info in streams_to_process
                if str(info.get("media_type") or "") == media_type
            }
        )
        if not source_dirs:
            continue
        lines.append(f"{label}:")
        lines.extend(f"  - {source_dir}" for source_dir in source_dirs)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    if workspace_dir.exists() and not workspace_dir.is_dir():
        console.print(f"[red]Error: workspace path is not a directory: {workspace_dir}[/red]")
        return 1
    output_path = resolve_output_path(workspace_dir, args.output)
    if output_path.exists() and output_path.is_dir():
        console.print(f"[red]Error: output path is a directory: {output_path}[/red]")
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

    for line in build_export_context_lines(day_dir, workspace_dir, output_path, streams_to_process):
        console.print(line)

    resume_rows = initialize_resume_rows(output_path, args.restart)
    if resume_rows is None:
        return 1

    with Progress(*build_progress_columns(), expand=False, console=console) as progress:
        discover_task = progress.add_task("Discover files".ljust(25), total=None)
        stream_files = [
            (
                info,
                collect_files(
                    Path(info["source_dir"]),
                    info["media_type"],
                    on_files_discovered=lambda batch_size, task_id=discover_task: progress.advance(task_id, batch_size),
                ),
            )
            for info in streams_to_process
        ]
        total_files = sum(len(files) for _info, files in stream_files)
        progress.update(discover_task, total=total_files, completed=total_files)
        if total_files == 0:
            console.print(
                "[red]Error: no supported media files found in selected streams under "
                f"{day_dir}.[/red]"
            )
            return 1
        discovered_relative_paths = {
            path.relative_to(day_dir).as_posix()
            for _info, files in stream_files
            for path in files
        }
        rows_by_relative_path: Dict[str, Dict[str, str]] = {
            str(row["relative_path"]): dict(row)
            for row in resume_rows
            if str(row.get("relative_path") or "") in discovered_relative_paths
        }
        batch_specs = build_batch_specs(
            stream_files,
            set(rows_by_relative_path),
            day_dir,
        )
        remaining_files = sum(len(batch) for _info, batch in batch_specs)

        if not batch_specs:
            if rows_by_relative_path:
                snapshot_rows = persist_manifest_snapshot(output_path, rows_by_relative_path)
                finalize_successful_export(output_path)
                console.print(
                    f"[green]Manifest already up to date with {len(snapshot_rows)} rows at {output_path}[/green]"
                )
                return 0
            console.print(
                "[red]Error: no remaining media files matched the current selection under "
                f"{day_dir}.[/red]"
            )
            return 1

        process_task = progress.add_task("Process media".ljust(25), total=remaining_files)

        interrupt_state, previous_handler = install_interrupt_handler()
        try:
            if args.jobs <= 1 or len(batch_specs) <= 1:
                for info, batch in batch_specs:
                    batch_rows = build_rows_for_batch(
                        day_dir,
                        info,
                        batch,
                        on_batch_processed=lambda count, task_id=process_task: progress.advance(task_id, count),
                    )
                    for row in batch_rows:
                        rows_by_relative_path[str(row["relative_path"])] = row
                    persist_manifest_snapshot(output_path, rows_by_relative_path)
                    if interrupt_state.stop_requested:
                        break
            else:
                max_workers = min(args.jobs, len(batch_specs))
                batch_iter = iter(batch_specs)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch: Dict[concurrent.futures.Future[List[Dict[str, str]]], tuple[Dict[str, str], List[Path]]] = {}
                    while len(future_to_batch) < max_workers:
                        try:
                            info, batch = next(batch_iter)
                        except StopIteration:
                            break
                        future = executor.submit(
                            build_rows_for_batch,
                            day_dir,
                            info,
                            batch,
                            lambda count, task_id=process_task: progress.advance(task_id, count),
                        )
                        future_to_batch[future] = (info, batch)

                    while future_to_batch:
                        done, _pending = concurrent.futures.wait(
                            future_to_batch,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            _info, batch = future_to_batch.pop(future)
                            batch_rows = future.result()
                            for row in batch_rows:
                                rows_by_relative_path[str(row["relative_path"])] = row
                            persist_manifest_snapshot(output_path, rows_by_relative_path)

                        if interrupt_state.stop_requested:
                            continue

                        while len(future_to_batch) < max_workers:
                            try:
                                info, batch = next(batch_iter)
                            except StopIteration:
                                break
                            future = executor.submit(
                                build_rows_for_batch,
                                day_dir,
                                info,
                                batch,
                                lambda count, task_id=process_task: progress.advance(task_id, count),
                            )
                            future_to_batch[future] = (info, batch)
        finally:
            restore_interrupt_handler(previous_handler)

    snapshot_rows = build_manifest_snapshot(rows_by_relative_path.values())
    if interrupt_state.stop_requested:
        console.print(
            "[yellow]Stopped after completing active batch(es). "
            f"Saved {len(snapshot_rows)} rows to {partial_output_path(output_path)} and {output_path}.[/yellow]"
        )
        return 130

    finalize_successful_export(output_path)
    console.print(f"[green]Wrote {len(snapshot_rows)} rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
