#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
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


def collect_files(stream_dir: Path, media_type: str) -> List[Path]:
    allowed_extensions = PHOTO_EXTENSIONS if media_type == "photo" else VIDEO_EXTENSIONS
    files: List[Path] = []
    for path in sorted(stream_dir.iterdir()):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_extensions:
            continue
        files.append(path)
    return files


def metadata_by_source_path(items: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    output: Dict[str, Mapping[str, object]] = {}
    for item in items:
        source_file = str(item.get("SourceFile") or "").strip()
        if not source_file:
            continue
        output[source_file] = item
    return output


def resolve_workspace_dir(day_dir: Path, workspace_value: Optional[str]) -> Path:
    if workspace_value:
        return Path(workspace_value).resolve()
    return day_dir / "_workspace"


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
    for index in range(0, len(paths), EXIFTOOL_BATCH_SIZE):
        batch = paths[index:index + EXIFTOOL_BATCH_SIZE]
        cmd = exiftool_base_command()
        cmd.extend(str(path) for path in batch)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
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
            try:
                items = run_exiftool(batch)
            except Exception as error:
                warn_metadata_batch_failure(stream_id, batch, error)
                items = []
            metadata_for_streams[stream_id].update(metadata_by_source_path(items))
            progress.advance(metadata_task, len(batch))
        return metadata_for_streams

    max_workers = min(jobs, len(batch_specs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(run_exiftool, batch): (stream_id, batch)
            for stream_id, batch in batch_specs
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            stream_id, batch = future_to_batch[future]
            try:
                items = future.result()
            except Exception as error:
                warn_metadata_batch_failure(stream_id, batch, error)
                items = []
            metadata_for_streams[stream_id].update(metadata_by_source_path(items))
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

    stream_files = [
        (info, collect_files(Path(info["source_dir"]), info["media_type"]))
        for info in streams_to_process
    ]
    total_files = sum(len(files) for _info, files in stream_files)
    if total_files == 0:
        console.print(
            "[red]Error: no supported media files found in selected streams under "
            f"{day_dir}.[/red]"
        )
        return 1

    photo_rows_with_sort: List[tuple[tuple[datetime, str, str], Dict[str, str]]] = []
    video_rows: List[Dict[str, str]] = []
    with Progress(*build_progress_columns(), expand=False, console=console) as progress:
        discover_task = progress.add_task("Discover files".ljust(25), total=total_files)
        progress.update(discover_task, completed=total_files)
        metadata_task = progress.add_task("Read metadata".ljust(25), total=total_files)
        row_task = progress.add_task("Build manifest rows".ljust(25), total=total_files)
        metadata_for_streams = extract_metadata_by_stream(
            stream_files,
            args.jobs,
            progress,
            metadata_task,
        )

        for info, files in stream_files:
            stream_metadata = metadata_for_streams.get(info["stream_id"], {})
            for path in files:
                metadata = stream_metadata.get(str(path))
                if info["media_type"] == "photo":
                    sort_key, row = build_photo_manifest_entry(
                        day_dir,
                        info["stream_id"],
                        info["device"],
                        path,
                        metadata,
                    )
                    photo_rows_with_sort.append((sort_key, row))
                else:
                    video_rows.append(
                        build_video_manifest_entry(
                            day_dir,
                            info["stream_id"],
                            info["device"],
                            path,
                            metadata,
                        )
                    )
                progress.advance(row_task)

    ordered_photo_rows = [
        row
        for _sort_key, row in sorted(photo_rows_with_sort, key=lambda item: item[0])
    ]
    assign_photo_order_indexes(ordered_photo_rows)
    rows = ordered_photo_rows + video_rows
    rows.sort(
        key=lambda row: (
            row.get("start_local", ""),
            row.get("media_type", ""),
            row.get("stream_id", ""),
            row.get("relative_path", ""),
        )
    )
    write_media_manifest_csv(output_path, rows)
    console.print(f"[green]Wrote {len(rows)} rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
