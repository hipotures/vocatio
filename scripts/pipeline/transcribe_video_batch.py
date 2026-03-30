#!/usr/bin/env python3

import argparse
import csv
import json
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


console = Console()

OUTPUT_FORMAT_EXTENSIONS = {
    "json": ".json",
    "vtt": ".vtt",
    "srt": ".srt",
    "tsv": ".tsv",
    "txt": ".txt",
    "aud": ".wav",
    "all": ".json",
}


class FixedMofNColumn(ProgressColumn):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def render(self, task) -> Text:
        total = int(task.total) if task.total is not None else 0
        completed = int(task.completed)
        return Text(f"{completed:>{self.width}}/{total:>{self.width}}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch transcribe synced video clips with WhisperX into the day workspace."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video_synced.csv and sync_map.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--merged-csv",
        help="Synced video CSV path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    parser.add_argument(
        "--sync-map",
        help="Sync map CSV path. Default: DAY/_workspace/sync_map.csv",
    )
    parser.add_argument(
        "--output-root",
        help="Transcript output root directory. Default: DAY/_workspace/transcripts",
    )
    parser.add_argument(
        "--manifest",
        default="transcripts_manifest.csv",
        help="Manifest filename inside workspace or absolute path. Default: transcripts_manifest.csv",
    )
    parser.add_argument(
        "--whisperx-bin",
        help="WhisperX executable path. Default: DAY/.venv/bin/whisperx or PATH whisperx",
    )
    parser.add_argument(
        "--streams",
        nargs="*",
        help='Specific stream IDs to transcribe, for example "v-pocket3" "v-gh7"',
    )
    parser.add_argument(
        "--all-streams",
        action="store_true",
        help="Transcribe every stream in merged_video_synced.csv",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available stream IDs and exit",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit for the number of files to process after filtering",
    )
    parser.add_argument(
        "--model",
        default="large",
        help="WhisperX model name. Default: large",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="WhisperX device. Default: cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX batch size. Default: 16",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="WhisperX chunk size. Default: 10",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["default", "float16", "float32", "int8"],
        help="WhisperX compute type. Default: float16",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="WhisperX CPU thread hint. Default: 8",
    )
    parser.add_argument(
        "--language",
        default="pl",
        help='Language code, for example "pl", or "auto" to let WhisperX detect it. Default: pl',
    )
    parser.add_argument(
        "--output-format",
        default="json",
        choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"],
        help="WhisperX output format. Default: json",
    )
    parser.set_defaults(no_align=True)
    parser.add_argument(
        "--no-align",
        dest="no_align",
        action="store_true",
        help="Disable WhisperX alignment. Default: enabled",
    )
    parser.add_argument(
        "--align",
        dest="no_align",
        action="store_false",
        help="Enable WhisperX alignment",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token for alignment or diarization models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Transcribe even if the primary output file already exists",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=0.0,
        help="Optional minimum clip duration after filtering. Default: 0",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_whisperx_binary(day_dir: Path, explicit_path: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(day_dir / ".venv" / "bin" / "whisperx")
    candidates.append(Path(".venv/bin/whisperx"))
    resolved = shutil.which("whisperx")
    if resolved:
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    console.print("[red]Error: WhisperX executable not found.[/red]")
    raise SystemExit(1)


def load_reference_stream_id(sync_map_path: Path) -> Optional[str]:
    if not sync_map_path.exists():
        return None
    rows = read_csv_rows(sync_map_path)
    if not rows:
        return None
    return rows[0].get("reference_stream_id") or None


def select_rows(
    rows: Sequence[Dict[str, str]],
    streams: Optional[Sequence[str]],
    all_streams: bool,
    reference_stream_id: Optional[str],
    max_files: Optional[int],
    min_duration_seconds: float,
) -> List[Dict[str, str]]:
    available_streams = sorted({row["stream_id"] for row in rows})
    if streams:
        missing = [stream_id for stream_id in streams if stream_id not in available_streams]
        if missing:
            console.print(f"[red]Error: unknown streams: {', '.join(missing)}[/red]")
            raise SystemExit(1)
        selected_streams = list(streams)
    elif all_streams:
        selected_streams = available_streams
    elif reference_stream_id:
        selected_streams = [reference_stream_id]
    else:
        selected_streams = available_streams

    filtered = [
        row
        for row in rows
        if row["stream_id"] in selected_streams and float(row.get("duration_seconds") or 0.0) >= min_duration_seconds
    ]
    filtered.sort(key=lambda row: (row.get("start_synced", ""), row["stream_id"], row["filename"]))
    if max_files is not None:
        filtered = filtered[:max_files]
    return filtered


def json_has_segments(path: Path) -> Optional[bool]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    return bool(segments)


def primary_output_path(output_dir: Path, filename: str, output_format: str) -> Path:
    base_name = Path(filename).stem
    extension = OUTPUT_FORMAT_EXTENSIONS[output_format]
    return output_dir / f"{base_name}{extension}"


def build_whisperx_command(
    whisperx_bin: Path,
    row: Dict[str, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        str(whisperx_bin),
        str(Path(row["path"])),
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--chunk_size",
        str(args.chunk_size),
        "--compute_type",
        args.compute_type,
        "--threads",
        str(args.threads),
        "--task",
        "transcribe",
        "--output_dir",
        str(output_dir),
        "--output_format",
        args.output_format,
        "--verbose",
        "False",
        "--print_progress",
        "False",
    ]
    if args.language.lower() != "auto":
        command.extend(["--language", args.language])
    if args.no_align:
        command.append("--no_align")
    if args.hf_token:
        command.extend(["--hf_token", args.hf_token])
    return command


def run_whisperx_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            return subprocess.CompletedProcess(
                args=list(command),
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            continue


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def build_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Transcription Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Done", justify="right", style="yellow")
    table.add_column("Empty", justify="right", style="white")
    table.add_column("Skipped", justify="right", style="cyan")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Output Dir", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video_synced.csv"
    sync_map_path = Path(args.sync_map).resolve() if args.sync_map else workspace_dir / "sync_map.csv"
    output_root = Path(args.output_root).resolve() if args.output_root else workspace_dir / "transcripts"
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace_dir / manifest_path

    if not merged_csv.exists():
        console.print(f"[red]Error: merged synced video CSV not found: {merged_csv}[/red]")
        return 1

    whisperx_bin = find_whisperx_binary(day_dir, args.whisperx_bin)
    rows = read_csv_rows(merged_csv)
    available_streams = sorted({row["stream_id"] for row in rows})
    if args.list_streams:
        for stream_id in available_streams:
            console.print(stream_id)
        return 0

    reference_stream_id = load_reference_stream_id(sync_map_path)
    selected_rows = select_rows(
        rows,
        streams=args.streams,
        all_streams=args.all_streams,
        reference_stream_id=reference_stream_id,
        max_files=args.max_files,
        min_duration_seconds=args.min_duration_seconds,
    )
    if not selected_rows:
        console.print("[red]Error: no video rows selected for transcription.[/red]")
        return 1

    count_width = max(len(str(len(selected_rows))), len(str(len(sorted({row['stream_id'] for row in selected_rows})))), 2)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        FixedMofNColumn(count_width),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )

    manifest_rows: List[Dict[str, str]] = []
    summary: Dict[str, Dict[str, int]] = {}
    output_dirs: Dict[str, Path] = {}
    unique_streams = sorted({row["stream_id"] for row in selected_rows})
    stop_requested = False
    stop_message_shown = False
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame) -> None:
        nonlocal stop_requested, stop_message_shown
        stop_requested = True
        if not stop_message_shown:
            console.print("\n[yellow]Stop requested. Waiting for the current file to finish...[/yellow]")
            stop_message_shown = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        with progress:
            stream_task = progress.add_task("Streams".ljust(25), total=len(unique_streams))
            file_task = progress.add_task("Files".ljust(25), total=len(selected_rows))
            for stream_id in unique_streams:
                stream_rows = [row for row in selected_rows if row["stream_id"] == stream_id]
                progress.update(stream_task, description=f"Streams ({stream_id})".ljust(25))
                output_dir = output_root / stream_id
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[stream_id] = output_dir
                summary.setdefault(stream_id, {"files": 0, "done": 0, "empty": 0, "skipped": 0, "failed": 0})
                progress.update(file_task, description="Files".ljust(25))
                for row in stream_rows:
                    if stop_requested:
                        break
                    summary[stream_id]["files"] += 1
                    expected_path = primary_output_path(output_dir, row["filename"], args.output_format)
                    status = "done"
                    error_message = ""
                    if expected_path.exists() and not args.force:
                        status = "skipped_existing"
                        summary[stream_id]["skipped"] += 1
                    else:
                        command = build_whisperx_command(whisperx_bin, row, output_dir, args)
                        result = run_whisperx_command(command)
                        if result.returncode == 0:
                            has_segments = None
                            if args.output_format in {"json", "all"}:
                                has_segments = json_has_segments(expected_path)
                            if has_segments is False:
                                status = "done_empty"
                                summary[stream_id]["empty"] += 1
                            else:
                                summary[stream_id]["done"] += 1
                        else:
                            status = "failed"
                            summary[stream_id]["failed"] += 1
                            error_message = (result.stderr or result.stdout).strip().splitlines()[-1] if (result.stderr or result.stdout).strip() else "unknown error"
                    manifest_rows.append(
                        {
                            "day": row["day"],
                            "stream_id": row["stream_id"],
                            "device": row["device"],
                            "path": row["path"],
                            "filename": row["filename"],
                            "start_synced": row.get("start_synced", ""),
                            "end_synced": row.get("end_synced", ""),
                            "output_dir": str(output_dir),
                            "primary_output": str(expected_path),
                            "output_format": args.output_format,
                            "model": args.model,
                            "language": args.language,
                            "status": status,
                            "error": error_message,
                        }
                    )
                    progress.advance(file_task)
                progress.advance(stream_task)
                if stop_requested:
                    break
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)

    manifest_headers = [
        "day",
        "stream_id",
        "device",
        "path",
        "filename",
        "start_synced",
        "end_synced",
        "output_dir",
        "primary_output",
        "output_format",
        "model",
        "language",
        "status",
        "error",
    ]
    write_csv(manifest_path, manifest_headers, manifest_rows)

    summary_rows: List[List[str]] = []
    for stream_id in unique_streams:
        stats = summary[stream_id]
        summary_rows.append(
            [
                stream_id,
                str(stats["files"]),
                str(stats["done"]),
                str(stats["empty"]),
                str(stats["skipped"]),
                str(stats["failed"]),
                str(output_dirs[stream_id]),
            ]
        )
    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote manifest to {manifest_path}[/green]")
    if stop_requested:
        console.print("[yellow]Stopped after completing the current file.[/yellow]")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
