#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

SUPPORTED_FORMATS = {".arw", ".cr3", ".hif", ".heif", ".jpg", ".jpeg", ".nef"}


class FixedMofNColumn(ProgressColumn):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def render(self, task) -> Text:
        total = int(task.total) if task.total is not None else 0
        completed = int(task.completed)
        return Text(f"{completed:>{self.width}}/{total:>{self.width}}")


class EtaColumn(ProgressColumn):
    def render(self, task) -> Text:
        eta_seconds = task.fields.get("eta_seconds")
        if eta_seconds is None:
            return Text("ETA --:--:--")
        return Text(f"ETA {timedelta(seconds=int(eta_seconds))}")


def estimate_eta_seconds(task) -> int | None:
    if task.total is None or task.total <= 0 or task.completed <= 0 or task.elapsed is None:
        return None
    remaining_steps = max(0.0, float(task.total) - float(task.completed))
    if remaining_steps <= 0:
        return 0
    seconds_per_step = float(task.elapsed) / float(task.completed)
    return max(0, int(math.ceil(seconds_per_step * remaining_steps)))


def get_progress_task(progress: Progress, task_id: int):
    for task in progress.tasks:
        if task.id == task_id:
            return task
    raise RuntimeError(f"Unknown progress task id: {task_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate proxy JPG files from exported photo CSV rows into the day workspace."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output-root",
        help="Proxy output root directory. Default: DAY/_workspace/proxy_jpg",
    )
    parser.add_argument(
        "--manifest",
        default="photo_proxy_manifest.csv",
        help="Manifest filename inside workspace or absolute path. Default: photo_proxy_manifest.csv",
    )
    parser.add_argument(
        "--streams",
        nargs="*",
        help='Specific photo stream IDs to process, for example "p-a7r5"',
    )
    parser.add_argument(
        "--all-streams",
        action="store_true",
        help="Process every exported photo CSV in the workspace",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available photo stream IDs and exit",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit for the number of files to process after filtering",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=800,
        help="Resize so the longer edge fits within this size. Default: 800",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality for ImageMagick backend. Default: 85",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of files to process in parallel. Default: 4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing proxy JPG files",
    )
    parser.add_argument(
        "--continue-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def detect_backend() -> str:
    if shutil.which("magick"):
        return "magick"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    raise RuntimeError("Install ImageMagick or ffmpeg to generate proxy JPG files.")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def discover_photo_csvs(workspace_dir: Path) -> Dict[str, Path]:
    csv_paths = {}
    for path in sorted(workspace_dir.glob("p-*.csv")):
        csv_paths[path.stem] = path
    return csv_paths


def select_streams(available: Sequence[str], streams: Sequence[str] | None, all_streams: bool) -> List[str]:
    available_set = set(available)
    if streams:
        missing = [stream_id for stream_id in streams if stream_id not in available_set]
        if missing:
            console.print(f"[red]Error: unknown photo stream IDs: {', '.join(missing)}[/red]")
            raise SystemExit(1)
        return list(streams)
    if all_streams:
        return list(available)
    if len(available) == 1:
        return list(available)
    console.print("[red]Error: multiple photo streams available. Use --streams or --all-streams.[/red]")
    raise SystemExit(1)


def select_rows(
    csv_map: Dict[str, Path],
    stream_ids: Sequence[str],
) -> List[Dict[str, str]]:
    selected_rows: List[Dict[str, str]] = []
    for stream_id in stream_ids:
        for row in read_csv_rows(csv_map[stream_id]):
            if row.get("media_type") != "photo":
                continue
            if Path(row["path"]).suffix.lower() not in SUPPORTED_FORMATS:
                continue
            selected_rows.append(row)
    selected_rows.sort(key=lambda row: (row.get("start_local", ""), row["stream_id"], row["filename"]))
    return selected_rows


def convert_with_magick(source: Path, destination: Path, long_edge: int, quality: int) -> None:
    command = [
        "magick",
        str(source),
        "-auto-orient",
        "-colorspace",
        "sRGB",
        "-filter",
        "Lanczos",
        "-resize",
        f"{long_edge}x{long_edge}>",
        "-sampling-factor",
        "4:4:4",
        "-strip",
        "-depth",
        "8",
        "-quality",
        str(quality),
        str(destination),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def extract_heif_preview(source: Path, preview_path: Path) -> bool:
    probe_command = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_streams",
        str(source),
    ]
    probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
    stream_specs: List[tuple[int, int]] = []
    current_index = None
    current_width = None
    current_height = None
    for raw_line in probe_result.stdout.splitlines():
        line = raw_line.strip()
        if line == "[STREAM]":
            current_index = None
            current_width = None
            current_height = None
            continue
        if line == "[/STREAM]":
            if current_index is not None and current_width and current_height:
                long_edge = max(current_width, current_height)
                stream_specs.append((current_index, long_edge))
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "index":
            current_index = int(value)
        elif key == "width":
            current_width = int(value)
        elif key == "height":
            current_height = int(value)
    preview_streams = [item for item in stream_specs if 512 <= item[1] < 3000]
    if not preview_streams:
        return False
    preview_index = max(preview_streams, key=lambda item: item[1])[0]
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        f"0:v:{preview_index}",
        "-frames:v",
        "1",
        str(preview_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return preview_path.exists()


def resize_existing_jpg(source: Path, destination: Path, long_edge: int, quality: int) -> None:
    command = [
        "magick",
        str(source),
        "-auto-orient",
        "-colorspace",
        "sRGB",
        "-filter",
        "Lanczos",
        "-resize",
        f"{long_edge}x{long_edge}>",
        "-sampling-factor",
        "4:4:4",
        "-strip",
        "-depth",
        "8",
        "-quality",
        str(quality),
        str(destination),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def normalize_existing_jpg(source: Path, destination: Path, quality: int) -> None:
    command = [
        "magick",
        str(source),
        "-auto-orient",
        "-colorspace",
        "sRGB",
        "-sampling-factor",
        "4:4:4",
        "-strip",
        "-depth",
        "8",
        "-quality",
        str(quality),
        str(destination),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def convert_with_ffmpeg(source: Path, destination: Path, long_edge: int, overwrite: bool) -> None:
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        "ffmpeg",
        overwrite_flag,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale={long_edge}:{long_edge}:force_original_aspect_ratio=decrease:flags=lanczos",
        "-q:v",
        "2",
        str(destination),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def convert_file(source: Path, destination: Path, backend: str, long_edge: int, quality: int, overwrite: bool) -> None:
    if source.suffix.lower() in {".hif", ".heif"} and backend == "magick":
        with tempfile.TemporaryDirectory(prefix="scriptoza-proxy-") as temp_dir:
            preview_path = Path(temp_dir) / f"{source.stem}_preview.jpg"
            try:
                if extract_heif_preview(source, preview_path):
                    normalize_existing_jpg(preview_path, destination, quality)
                    return
            except subprocess.CalledProcessError:
                pass
    if backend == "magick":
        convert_with_magick(source, destination, long_edge, quality)
        return
    if backend == "ffmpeg":
        convert_with_ffmpeg(source, destination, long_edge, overwrite)
        return
    raise RuntimeError("No supported backend available")


def count_existing_outputs(output_dirs: Dict[str, Path]) -> int:
    total = 0
    for path in output_dirs.values():
        total += len(list(path.glob("*.jpg")))
    return total


def resolve_existing_output_mode(args: argparse.Namespace, output_dirs: Dict[str, Path]) -> tuple[bool, int]:
    if args.overwrite and args.continue_existing:
        console.print("[red]Error: use only one of --overwrite or --continue-existing.[/red]")
        raise SystemExit(1)
    existing_count = count_existing_outputs(output_dirs)
    if args.overwrite:
        if existing_count > 0:
            if not sys.stdin.isatty():
                console.print("[red]Error: --overwrite requires confirmation in an interactive terminal.[/red]")
                raise SystemExit(1)
            answer = console.input(
                f"Overwrite {existing_count} existing proxy JPG files? [y/N]: "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                raise SystemExit(1)
        return True, existing_count
    return False, existing_count


def limit_rows_for_run(
    selected_rows: Sequence[Dict[str, str]],
    output_dirs: Dict[str, Path],
    overwrite_mode: bool,
    max_files: int | None,
) -> List[Dict[str, str]]:
    if overwrite_mode:
        rows = list(selected_rows)
        if max_files is not None:
            rows = rows[:max_files]
        return rows
    missing_rows: List[Dict[str, str]] = []
    for row in selected_rows:
        output_dir = output_dirs[row["stream_id"]]
        destination = output_dir / f"{Path(row['filename']).stem}.jpg"
        if destination.exists():
            continue
        missing_rows.append(row)
        if max_files is not None and len(missing_rows) >= max_files:
            break
    return missing_rows


def build_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Photo Proxy Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Done", justify="right", style="yellow")
    table.add_column("Skipped", justify="right", style="cyan")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Output Dir", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def build_start_table(
    *,
    day_dir: Path,
    workspace_dir: Path,
    output_root: Path,
    backend: str,
    selected_streams: Sequence[str],
    selected_row_count: int,
    run_row_count: int,
    existing_count: int,
    overwrite_mode: bool,
    long_edge: int,
    quality: int,
    jobs: int,
    max_files: int | None,
) -> Table:
    table = Table(title="Photo Proxy Plan", expand=False)
    table.add_column("Setting", style="green")
    table.add_column("Value", style="white")
    table.add_row("Day", str(day_dir))
    table.add_row("Workspace", str(workspace_dir))
    table.add_row("Output Root", str(output_root))
    table.add_row("Backend", backend)
    table.add_row("Streams", ", ".join(selected_streams))
    table.add_row("Mode", "overwrite" if overwrite_mode else "continue")
    table.add_row("Existing Proxy JPG", str(existing_count))
    table.add_row("Selected Rows", str(selected_row_count))
    table.add_row("Rows In This Run", str(run_row_count))
    table.add_row("Long Edge", str(long_edge))
    table.add_row("Quality", str(quality))
    table.add_row("Jobs", str(jobs))
    table.add_row("Max Files", str(max_files) if max_files is not None else "unlimited")
    return table


def process_proxy_row(
    row: Dict[str, str],
    output_dir: Path,
    backend: str,
    long_edge: int,
    quality: int,
    overwrite_mode: bool,
) -> Dict[str, str]:
    source = Path(row["path"])
    destination = output_dir / f"{Path(row['filename']).stem}.jpg"
    status = "done"
    error_message = ""
    if destination.exists() and not overwrite_mode:
        status = "skipped_existing"
    else:
        try:
            convert_file(source, destination, backend, long_edge, quality, overwrite_mode)
        except subprocess.CalledProcessError as error:
            status = "failed"
            error_message = error.stderr.strip() or error.stdout.strip() or str(error)
        except Exception as error:
            status = "failed"
            error_message = str(error)
    return {
        "day": row["day"],
        "stream_id": row["stream_id"],
        "device": row["device"],
        "source_path": row["path"],
        "filename": row["filename"],
        "start_local": row.get("start_local", ""),
        "proxy_path": str(destination),
        "backend": backend,
        "long_edge": str(long_edge),
        "quality": str(quality),
        "status": status,
        "error": error_message,
    }


def main() -> int:
    args = parse_args()

    if args.long_edge < 64:
        console.print("[red]Error: --long-edge must be at least 64.[/red]")
        return 1
    if not 1 <= args.quality <= 100:
        console.print("[red]Error: --quality must be between 1 and 100.[/red]")
        return 1
    if args.jobs < 1:
        console.print("[red]Error: --jobs must be at least 1.[/red]")
        return 1

    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    output_root = Path(args.output_root).resolve() if args.output_root else workspace_dir / "proxy_jpg"
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace_dir / manifest_path

    backend = detect_backend()
    csv_map = discover_photo_csvs(workspace_dir)
    available_streams = sorted(csv_map)

    if args.list_streams:
        for stream_id in available_streams:
            console.print(stream_id)
        return 0

    if not available_streams:
        console.print(f"[red]Error: no photo CSV files found in {workspace_dir}.[/red]")
        return 1

    selected_streams = select_streams(available_streams, args.streams, args.all_streams)
    selected_rows = select_rows(csv_map, selected_streams)
    if not selected_rows:
        console.print("[red]Error: no photo rows selected for proxy generation.[/red]")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = {stream_id: output_root / stream_id for stream_id in selected_streams}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    overwrite_mode, existing_count = resolve_existing_output_mode(args, output_dirs)
    selected_row_count = len(selected_rows)
    selected_rows = limit_rows_for_run(selected_rows, output_dirs, overwrite_mode, args.max_files)
    console.print(
        build_start_table(
            day_dir=day_dir,
            workspace_dir=workspace_dir,
            output_root=output_root,
            backend=backend,
            selected_streams=selected_streams,
            selected_row_count=selected_row_count,
            run_row_count=len(selected_rows),
            existing_count=existing_count,
            overwrite_mode=overwrite_mode,
            long_edge=args.long_edge,
            quality=args.quality,
            jobs=args.jobs,
            max_files=args.max_files,
        )
    )
    if not selected_rows:
        console.print("[yellow]Nothing to do for the selected rows.[/yellow]")
        return 0

    count_width = max(len(str(len(selected_rows))), len(str(len(selected_streams))), 2)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        FixedMofNColumn(count_width),
        TaskProgressColumn(),
        EtaColumn(),
        console=console,
        expand=False,
    )

    manifest_rows: List[Dict[str, str]] = []
    summary: Dict[str, Dict[str, int]] = {}
    failed_messages: List[str] = []

    with progress:
        stream_task = None
        if len(selected_streams) > 1:
            stream_task = progress.add_task("Streams".ljust(25), total=len(selected_streams))
        file_task = progress.add_task("Photos".ljust(25), total=len(selected_rows), eta_seconds=None)
        last_eta_update = 0.0
        progress.update(file_task, description="Photos".ljust(25))
        pending_per_stream: Dict[str, int] = {}
        for stream_id in selected_streams:
            stream_rows = [row for row in selected_rows if row["stream_id"] == stream_id]
            summary.setdefault(stream_id, {"files": len(stream_rows), "done": 0, "skipped": 0, "failed": 0})
            pending_per_stream[stream_id] = len(stream_rows)

        future_to_index: Dict[concurrent.futures.Future, int] = {}
        manifest_rows_by_index: Dict[int, Dict[str, str]] = {}
        max_workers = min(args.jobs, max(1, len(selected_rows)), max(1, os.cpu_count() or 1))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in enumerate(selected_rows):
                future = executor.submit(
                    process_proxy_row,
                    row,
                    output_dirs[row["stream_id"]],
                    backend,
                    args.long_edge,
                    args.quality,
                    overwrite_mode,
                )
                future_to_index[future] = index

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                row = selected_rows[index]
                result = future.result()
                manifest_rows_by_index[index] = result
                stream_id = row["stream_id"]
                if result["status"] == "done":
                    summary[stream_id]["done"] += 1
                elif result["status"] == "skipped_existing":
                    summary[stream_id]["skipped"] += 1
                else:
                    summary[stream_id]["failed"] += 1
                    failed_messages.append(f"{Path(row['path']).name} -> {result['error']}")

                pending_per_stream[stream_id] -= 1
                if stream_task is not None and pending_per_stream[stream_id] == 0:
                    progress.advance(stream_task)

                progress.advance(file_task)
                now = time.monotonic()
                if now - last_eta_update >= 1.0:
                    eta_seconds = estimate_eta_seconds(get_progress_task(progress, file_task))
                    progress.update(file_task, eta_seconds=eta_seconds)
                    last_eta_update = now
        progress.update(file_task, eta_seconds=0)

    for index in range(len(selected_rows)):
        manifest_rows.append(manifest_rows_by_index[index])

    manifest_headers = [
        "day",
        "stream_id",
        "device",
        "source_path",
        "filename",
        "start_local",
        "proxy_path",
        "backend",
        "long_edge",
        "quality",
        "status",
        "error",
    ]
    write_csv(manifest_path, manifest_headers, manifest_rows)

    summary_rows = []
    for stream_id in selected_streams:
        stats = summary.get(stream_id, {"files": 0, "done": 0, "skipped": 0, "failed": 0})
        summary_rows.append(
            [
                stream_id,
                str(stats["files"]),
                str(stats["done"]),
                str(stats["skipped"]),
                str(stats["failed"]),
                str(output_dirs[stream_id]),
            ]
        )

    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote manifest to {manifest_path}[/green]")
    if failed_messages:
        for message in failed_messages[:10]:
            console.print(f"[red]Failed:[/red] {message}")
        if len(failed_messages) > 10:
            console.print(f"[red]Additional failures omitted:[/red] {len(failed_messages) - 10}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
