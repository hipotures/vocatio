#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple

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

from lib.media_manifest import read_media_manifest, select_photo_rows
from lib.workspace_dir import resolve_workspace_dir
console = Console()

PHOTO_EXTENSIONS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}
MANIFEST_HEADERS = [
    "relative_path",
    "photo_order_index",
    "path",
    "source_path",
    "photo_id",
    "filename",
    "extension",
    "thumb_path",
    "preview_path",
    "thumb_exists",
    "preview_exists",
    "thumb_width",
    "thumb_height",
    "preview_width",
    "preview_height",
    "preview_source",
]
PREVIEW_TAGS = ("PreviewImage", "JpgFromRaw")
THUMB_TAGS = ("ThumbnailImage",)
DEFAULT_PREVIEW_LONG_EDGE = 1600
DEFAULT_THUMB_LONG_EDGE = 160
MEDIA_MANIFEST_NAME = "media_manifest.csv"
PREVIEW_SOURCE_EMBEDDED = "embedded_preview"
PREVIEW_SOURCE_EMBEDDED_RAW = "embedded_jpg_from_raw"
PREVIEW_SOURCE_GENERATED = "generated_from_source"
PREVIEW_SOURCE_EXISTING = "existing_preview"
JPEG_SOI = b"\xff\xd8"
SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embedded preview and thumb JPG files into the day workspace."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output",
        default="photo_embedded_manifest.csv",
        help="Manifest filename or relative path inside workspace. Default: photo_embedded_manifest.csv",
    )
    parser.add_argument(
        "--thumb-long-edge",
        type=positive_int_arg,
        default=DEFAULT_THUMB_LONG_EDGE,
        help="Resize thumbs so the longer edge fits within this size. Default: 160",
    )
    parser.add_argument(
        "--preview-long-edge",
        type=positive_int_arg,
        default=DEFAULT_PREVIEW_LONG_EDGE,
        help="Resize previews so the longer edge fits within this size. Default: 1600",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preview and thumb JPG files",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int_arg,
        default=4,
        help="Number of photos to process in parallel. Default: 4",
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
    writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS)
    writer.writeheader()
    flush_and_sync(handle)
    return handle, writer


def append_partial_manifest_row(handle: TextIO, writer: csv.DictWriter, row: Dict[str, str]) -> None:
    writer.writerow(row)
    flush_and_sync(handle)


def replace_output_from_partial(partial_path: Path, output_path: Path) -> None:
    os.replace(partial_path, output_path)
    fsync_directory(output_path.parent)


def resolve_output_path(workspace_dir: Path, output_value: str) -> Path:
    workspace_dir = workspace_dir.resolve()
    candidate = Path(output_value)
    output_path = candidate if candidate.is_absolute() else workspace_dir / candidate
    resolved_output_path = output_path.resolve()
    try:
        resolved_output_path.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"Output path must stay under {workspace_dir}") from exc
    return resolved_output_path


def normalize_manifest_relative_path(relative_path: str) -> Path:
    candidate = PurePosixPath(relative_path.strip())
    if not candidate.parts:
        raise ValueError(f"{MEDIA_MANIFEST_NAME} relative_path is empty")
    if candidate.is_absolute():
        raise ValueError(f"{MEDIA_MANIFEST_NAME} relative_path must stay under workspace: {relative_path}")
    normalized_parts: List[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                raise ValueError(f"{MEDIA_MANIFEST_NAME} relative_path must stay under workspace: {relative_path}")
            normalized_parts.pop()
            continue
        normalized_parts.append(part)
    if not normalized_parts:
        raise ValueError(f"{MEDIA_MANIFEST_NAME} relative_path must stay under workspace: {relative_path}")
    return Path(*normalized_parts)


def resolve_workspace_derived_output_path(workspace_dir: Path, relative_path: str, variant: str) -> Path:
    workspace_dir = workspace_dir.resolve()
    normalized_relative_path = normalize_manifest_relative_path(relative_path)
    output_name = f"{normalized_relative_path.stem}.jpg"
    candidate = workspace_dir / "embedded_jpg" / variant / normalized_relative_path.parent / output_name
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"Derived output path must stay under {workspace_dir}: {candidate}") from exc
    return resolved_candidate


def build_output_paths(workspace_dir: Path, relative_path: str) -> Dict[str, Path]:
    return {
        "thumb_path": resolve_workspace_derived_output_path(workspace_dir, relative_path, "thumb"),
        "preview_path": resolve_workspace_derived_output_path(workspace_dir, relative_path, "preview"),
    }


def validate_unique_output_paths(workspace_dir: Path, photo_rows: Sequence[Dict[str, str]]) -> None:
    preview_owner_by_path: Dict[Path, str] = {}
    for photo_row in photo_rows:
        relative_path = str(photo_row.get("relative_path") or "").strip()
        if not relative_path:
            raise ValueError(f"{MEDIA_MANIFEST_NAME} photo row missing relative_path")
        preview_path = build_output_paths(workspace_dir, relative_path)["preview_path"]
        existing_owner = preview_owner_by_path.get(preview_path)
        if existing_owner is None:
            preview_owner_by_path[preview_path] = relative_path
            continue
        raise ValueError(
            f"Conflicting derived preview path {preview_path}: {existing_owner}, {relative_path}"
        )


def serialize_workspace_path(workspace_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_dir.resolve()).as_posix()


def load_photo_rows(workspace_dir: Path) -> List[Dict[str, str]]:
    manifest_path = workspace_dir / MEDIA_MANIFEST_NAME
    if not manifest_path.exists():
        raise ValueError(f"Required manifest not found: {manifest_path}")
    rows = select_photo_rows(read_media_manifest(manifest_path))
    if not rows:
        raise ValueError(f"No photo rows found in {manifest_path}")
    return rows


def normalize_preview_source(tag_name: Optional[str]) -> str:
    if tag_name == "PreviewImage":
        return PREVIEW_SOURCE_EMBEDDED
    if tag_name == "JpgFromRaw":
        return PREVIEW_SOURCE_EMBEDDED_RAW
    return PREVIEW_SOURCE_GENERATED


def build_manifest_row(
    workspace_dir: Path,
    source_path: Path,
    relative_path: str,
    photo_order_index: str,
    thumb_path: Path,
    preview_path: Path,
    preview_source: str,
    thumb_dimensions: Optional[Tuple[int, int]] = None,
    preview_dimensions: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    thumb_exists = thumb_path.exists()
    preview_exists = preview_path.exists()
    resolved_thumb_dimensions = thumb_dimensions if thumb_exists else None
    resolved_preview_dimensions = preview_dimensions if preview_exists else None
    if resolved_thumb_dimensions is None and thumb_exists:
        resolved_thumb_dimensions = read_jpeg_dimensions(thumb_path)
    if resolved_preview_dimensions is None and preview_exists:
        resolved_preview_dimensions = read_jpeg_dimensions(preview_path)
    row = {header: "" for header in MANIFEST_HEADERS}
    row.update(
        {
            "relative_path": relative_path,
            "photo_order_index": photo_order_index,
            "path": relative_path,
            "source_path": relative_path,
            "photo_id": relative_path,
            "filename": source_path.name,
            "extension": source_path.suffix.lower(),
            "thumb_path": serialize_workspace_path(workspace_dir, thumb_path),
            "preview_path": serialize_workspace_path(workspace_dir, preview_path),
            "thumb_exists": "1" if thumb_exists else "0",
            "preview_exists": "1" if preview_exists else "0",
            "thumb_width": str(resolved_thumb_dimensions[0]) if resolved_thumb_dimensions else "",
            "thumb_height": str(resolved_thumb_dimensions[1]) if resolved_thumb_dimensions else "",
            "preview_width": str(resolved_preview_dimensions[0]) if resolved_preview_dimensions else "",
            "preview_height": str(resolved_preview_dimensions[1]) if resolved_preview_dimensions else "",
            "preview_source": preview_source,
        }
    )
    return row


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    if not payload:
        raise ValueError(f"Refusing to write empty output for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if tmp_path.stat().st_size <= 0:
            raise ValueError(f"Refusing to replace {path.name} with empty output")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_replace_from_command(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = f".tmp{path.suffix}" if path.suffix else ".tmp"
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=tmp_suffix, dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        full_command = list(command)
        full_command.append(str(tmp_path))
        subprocess.run(full_command, capture_output=True, check=True)
        if tmp_path.stat().st_size <= 0:
            raise ValueError(f"Refusing to replace {path.name} with empty output")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def publish_processed_jpeg(
    path: Path,
    populate: Callable[[Path], None],
    post_process: Optional[Callable[[Path], None]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = path.suffix if path.suffix else ".tmp"
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=tmp_suffix, dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        populate(tmp_path)
        if tmp_path.stat().st_size <= 0:
            raise ValueError(f"Refusing to replace {path.name} with empty output")
        if post_process is not None:
            post_process(tmp_path)
            if tmp_path.stat().st_size <= 0:
                raise ValueError(f"Refusing to replace {path.name} with empty output")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def extract_exiftool_jpeg(source_path: Path, tag_name: str) -> bytes:
    result = subprocess.run(
        ["exiftool", "-b", f"-{tag_name}", str(source_path)],
        capture_output=True,
        check=True,
    )
    return result.stdout


def looks_like_jpeg(payload: bytes) -> bool:
    return payload.startswith(JPEG_SOI)


def detect_generation_backend() -> str:
    if shutil.which("magick"):
        return "magick"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    raise RuntimeError("Install ImageMagick or ffmpeg to generate JPG files.")


def auto_orient_jpeg(jpeg_path: Path) -> None:
    backend = detect_generation_backend()
    if backend == "magick":
        command = [
            "magick",
            str(jpeg_path),
            "-auto-orient",
            "-strip",
            "-sampling-factor",
            "4:4:4",
            "-depth",
            "8",
            "-quality",
            "90",
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(jpeg_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
        ]
    atomic_replace_from_command(jpeg_path, command)


def generate_resized_jpeg(source_path: Path, output_path: Path, max_edge: int) -> None:
    backend = detect_generation_backend()
    if backend == "magick":
        command = [
            "magick",
            str(source_path),
            "-auto-orient",
            "-colorspace",
            "sRGB",
            "-filter",
            "Lanczos",
            "-resize",
            f"{max_edge}x{max_edge}>",
            "-sampling-factor",
            "4:4:4",
            "-strip",
            "-depth",
            "8",
            "-quality",
            "90",
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale=if(gte(iw\\,ih)\\,min(iw\\,{max_edge})\\,-2):if(gte(ih\\,iw)\\,min(ih\\,{max_edge})\\,-2)",
            "-q:v",
            "2",
        ]
    publish_processed_jpeg(output_path, lambda tmp_path: atomic_replace_from_command(tmp_path, command), post_process=None)


def extract_first_embedded_jpeg(source_path: Path, tag_names: Sequence[str]) -> Tuple[Optional[str], Optional[bytes]]:
    for tag_name in tag_names:
        try:
            payload = extract_exiftool_jpeg(source_path, tag_name)
        except FileNotFoundError:
            return (None, None)
        except subprocess.CalledProcessError:
            continue
        if looks_like_jpeg(payload):
            return (tag_name, payload)
    return (None, None)


def parse_jpeg_dimensions(payload: bytes) -> Tuple[int, int]:
    if not looks_like_jpeg(payload):
        raise ValueError("Not a JPEG payload")
    index = 2
    payload_length = len(payload)
    while index < payload_length:
        while index < payload_length and payload[index] == 0xFF:
            index += 1
        if index >= payload_length:
            break
        marker = payload[index]
        index += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > payload_length:
            break
        segment_length = int.from_bytes(payload[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > payload_length:
            break
        if marker in SOF_MARKERS:
            if index + 7 > payload_length:
                break
            height = int.from_bytes(payload[index + 3:index + 5], "big")
            width = int.from_bytes(payload[index + 5:index + 7], "big")
            if width <= 0 or height <= 0:
                break
            return (width, height)
        index += segment_length
    raise ValueError("Could not parse JPEG dimensions")


def read_jpeg_dimensions(path: Path) -> Tuple[int, int]:
    return parse_jpeg_dimensions(path.read_bytes())


def resolve_manifest_source_path(day_dir: Path, relative_path: str, source_value: str) -> Path:
    day_dir = day_dir.resolve()
    expected_relative_path = normalize_manifest_relative_path(relative_path)
    manifest_source_path = day_dir / expected_relative_path
    expected_source_path = manifest_source_path.resolve()

    source_candidate = Path(source_value.strip())
    source_path = source_candidate.resolve() if source_candidate.is_absolute() else (day_dir / source_candidate).resolve()
    if source_path != expected_source_path:
        raise ValueError(
            f"Source photo path does not match relative_path {relative_path}: {source_value}"
        )
    return source_path


def ensure_preview_jpg(
    source_path: Path,
    preview_path: Path,
    overwrite: bool,
    long_edge: int = DEFAULT_PREVIEW_LONG_EDGE,
) -> Tuple[str, Tuple[int, int]]:
    if preview_path.exists() and not overwrite:
        return (PREVIEW_SOURCE_EXISTING, read_jpeg_dimensions(preview_path))
    tag_name, payload = extract_first_embedded_jpeg(source_path, PREVIEW_TAGS)
    preview_source = normalize_preview_source(tag_name)
    if payload is not None:
        publish_processed_jpeg(preview_path, lambda tmp_path: atomic_write_bytes(tmp_path, payload), post_process=auto_orient_jpeg)
        return (preview_source, read_jpeg_dimensions(preview_path))
    generate_resized_jpeg(source_path, preview_path, long_edge)
    return (PREVIEW_SOURCE_GENERATED, read_jpeg_dimensions(preview_path))


def ensure_thumb_jpg(
    source_path: Path,
    thumb_path: Path,
    preview_path: Path,
    overwrite: bool,
    long_edge: int = DEFAULT_THUMB_LONG_EDGE,
) -> Tuple[int, int]:
    if thumb_path.exists() and not overwrite:
        return read_jpeg_dimensions(thumb_path)
    _tag_name, payload = extract_first_embedded_jpeg(source_path, THUMB_TAGS)
    if payload is not None:
        publish_processed_jpeg(thumb_path, lambda tmp_path: atomic_write_bytes(tmp_path, payload), post_process=auto_orient_jpeg)
        return read_jpeg_dimensions(thumb_path)
    generate_resized_jpeg(preview_path if preview_path.exists() else source_path, thumb_path, long_edge)
    return read_jpeg_dimensions(thumb_path)


def process_manifest_row(
    day_dir: Path,
    workspace_dir: Path,
    photo_row: Dict[str, str],
    overwrite: bool,
    thumb_long_edge: int = DEFAULT_THUMB_LONG_EDGE,
    preview_long_edge: int = DEFAULT_PREVIEW_LONG_EDGE,
) -> Dict[str, str]:
    relative_path = photo_row["relative_path"]
    photo_order_index = str(photo_row.get("photo_order_index") or "").strip()
    source_value = str(photo_row.get("path") or "").strip()
    if not photo_order_index:
        raise ValueError(f"{MEDIA_MANIFEST_NAME} photo row missing photo_order_index for {relative_path}")
    if not source_value:
        raise ValueError(f"{MEDIA_MANIFEST_NAME} photo row missing path for {relative_path}")
    source_path = resolve_manifest_source_path(day_dir, relative_path, source_value)
    if not source_path.exists() or not source_path.is_file():
        raise ValueError(f"Source photo listed in {MEDIA_MANIFEST_NAME} does not exist: {source_path}")
    if source_path.suffix.lower() not in PHOTO_EXTENSIONS:
        raise ValueError(f"Unsupported photo extension for {source_path}")
    output_paths = build_output_paths(workspace_dir, relative_path)
    preview_source, preview_dimensions = ensure_preview_jpg(
        source_path,
        output_paths["preview_path"],
        overwrite,
        preview_long_edge,
    )
    thumb_dimensions = ensure_thumb_jpg(
        source_path,
        output_paths["thumb_path"],
        output_paths["preview_path"],
        overwrite,
        thumb_long_edge,
    )
    return build_manifest_row(
        workspace_dir=workspace_dir,
        source_path=source_path,
        relative_path=relative_path,
        photo_order_index=photo_order_index,
        thumb_path=output_paths["thumb_path"],
        preview_path=output_paths["preview_path"],
        preview_source=preview_source,
        thumb_dimensions=thumb_dimensions,
        preview_dimensions=preview_dimensions,
    )


def build_manifest_rows(
    day_dir: Path,
    workspace_dir: Path,
    overwrite: bool,
    thumb_long_edge: int = DEFAULT_THUMB_LONG_EDGE,
    preview_long_edge: int = DEFAULT_PREVIEW_LONG_EDGE,
) -> List[Dict[str, str]]:
    photo_rows = load_photo_rows(workspace_dir)
    validate_unique_output_paths(workspace_dir, photo_rows)
    rows: List[Dict[str, str]] = []
    with Progress(
        *build_progress_columns(),
        expand=False,
        console=console,
    ) as progress:
        task_id = progress.add_task("Extract embedded JPG".ljust(25), total=len(photo_rows))
        for photo_row in photo_rows:
            rows.append(
                process_manifest_row(
                    day_dir=day_dir,
                    workspace_dir=workspace_dir,
                    photo_row=photo_row,
                    overwrite=overwrite,
                    thumb_long_edge=thumb_long_edge,
                    preview_long_edge=preview_long_edge,
                )
            )
            progress.advance(task_id)
    return rows


def extract_embedded_photo_jpg(
    day_dir: Path,
    workspace_dir: Path,
    output_path: Path,
    overwrite: bool,
    thumb_long_edge: int = DEFAULT_THUMB_LONG_EDGE,
    preview_long_edge: int = DEFAULT_PREVIEW_LONG_EDGE,
    jobs: int = 4,
) -> int:
    photo_rows = load_photo_rows(workspace_dir)
    validate_unique_output_paths(workspace_dir, photo_rows)
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    partial_path = partial_output_path(output_path)
    remove_stale_partial_output(partial_path)
    row_count = 0
    with Progress(
        *build_progress_columns(),
        expand=False,
        console=console,
    ) as progress:
        task_id = progress.add_task("Extract embedded JPG".ljust(25), total=len(photo_rows))
        partial_handle, partial_writer = open_partial_manifest_csv(partial_path)
        try:
            max_workers = min(jobs, max(1, len(photo_rows)))
            next_write_index = 0
            pending_rows: Dict[int, Dict[str, str]] = {}
            future_to_index: Dict[concurrent.futures.Future, int] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for index, photo_row in enumerate(photo_rows):
                    future = executor.submit(
                        process_manifest_row,
                        day_dir,
                        workspace_dir,
                        photo_row,
                        overwrite,
                        thumb_long_edge,
                        preview_long_edge,
                    )
                    future_to_index[future] = index
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    pending_rows[index] = future.result()
                    progress.advance(task_id)
                    while next_write_index in pending_rows:
                        append_partial_manifest_row(partial_handle, partial_writer, pending_rows.pop(next_write_index))
                        row_count += 1
                        next_write_index += 1
        finally:
            partial_handle.close()
    replace_output_from_partial(partial_path, output_path)
    return row_count


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    output_path = resolve_output_path(workspace_dir, args.output)
    row_count = extract_embedded_photo_jpg(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        output_path=output_path,
        overwrite=args.overwrite,
        thumb_long_edge=args.thumb_long_edge,
        preview_long_edge=args.preview_long_edge,
        jobs=args.jobs,
    )
    console.print(f"Wrote {row_count} embedded JPG rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
