#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import logging
import os
import signal
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

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


def ensure_venv_python() -> None:
    if os.environ.get("SCRIPTOZA_VENV_BOOTSTRAPPED") == "1":
        return
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    env = os.environ.copy()
    env["SCRIPTOZA_VENV_BOOTSTRAPPED"] = "1"
    os.execve(str(venv_python), [str(venv_python), *sys.argv], env)


ensure_venv_python()

warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Lightning automatically upgraded your loaded checkpoint.*",
)

import torch
import whisperx
from whisperx.utils import get_writer


console = Console()

OUTPUT_FORMAT_EXTENSIONS = {
    "json": ".json",
    "vtt": ".vtt",
    "srt": ".srt",
    "tsv": ".tsv",
    "txt": ".txt",
    "aud": ".aud",
    "all": ".json",
}

WRITER_OPTIONS = {
    "max_line_width": None,
    "max_line_count": None,
    "highlight_words": False,
}

DEFAULT_INITIAL_PROMPT = (
    "Polish dance competition announcements. Transcribe only spoken announcer speech. "
    "Do not translate. Ignore song lyrics, background singing, subtitles, watermarks, outros, and credits."
)
DEFAULT_HOTWORDS = "numer,numer startowy,kategoria,solo,duo,trio,formacja"


def configure_runtime_noise(print_progress: bool) -> None:
    if print_progress:
        return
    logging.getLogger("whisperx").setLevel(logging.WARNING)
    logging.getLogger("whisperx.vads.pyannote").setLevel(logging.WARNING)
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("pyannote.audio").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


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
        description="Batch transcribe synced video clips with WhisperX Python API into the day workspace."
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
        "--filenames",
        nargs="*",
        help="Optional exact video filenames to process",
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
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Enable WhisperX internal progress printing",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="ASR temperature. Default: 0.0",
    )
    parser.add_argument(
        "--log-prob-threshold",
        type=float,
        default=-0.5,
        help="ASR log probability threshold. Default: -0.5",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.8,
        help="ASR no-speech threshold. Default: 0.8",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=1.0,
        help="ASR hallucination silence threshold. Default: 1.0",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="ASR initial prompt used to bias the transcription",
    )
    parser.add_argument(
        "--hotwords",
        default=DEFAULT_HOTWORDS,
        help="Comma-separated ASR hotwords. Default: competition announcement keywords",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    filenames: Optional[Sequence[str]],
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
    if filenames:
        requested = set(filenames)
        available = {row["filename"] for row in filtered}
        missing = sorted(requested - available)
        if missing:
            console.print(f"[red]Error: unknown filenames: {', '.join(missing)}[/red]")
            raise SystemExit(1)
        filtered = [row for row in filtered if row["filename"] in requested]
    filtered.sort(key=lambda row: (row.get("start_synced", ""), row["stream_id"], row["filename"]))
    if max_files is not None:
        filtered = filtered[:max_files]
    return filtered


def primary_output_path(output_dir: Path, filename: str, output_format: str) -> Path:
    base_name = Path(filename).stem
    extension = OUTPUT_FORMAT_EXTENSIONS[output_format]
    return output_dir / f"{base_name}{extension}"


def json_has_segments(path: Path) -> Optional[bool]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    return bool(segments)


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


def build_writer(output_format: str, output_dir: Path) -> Callable[[dict, str, dict], None]:
    return get_writer(output_format, str(output_dir))


def cleanup_device_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    configure_runtime_noise(args.print_progress)
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
        filenames=args.filenames,
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
        if args.threads > 0:
            torch.set_num_threads(args.threads)

        output_root.mkdir(parents=True, exist_ok=True)
        output_dirs = {stream_id: output_root / stream_id for stream_id in unique_streams}
        for path in output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        writers: Dict[str, Callable[[dict, str, dict], None]] = {
            stream_id: build_writer(args.output_format, output_dirs[stream_id])
            for stream_id in unique_streams
        }

        model_language = None if args.language.lower() == "auto" else args.language
        hotwords = ",".join(part.strip() for part in args.hotwords.split(",") if part.strip()) or None
        initial_prompt = args.initial_prompt.strip() or None
        asr_options = {
            "temperatures": [args.temperature],
            "log_prob_threshold": args.log_prob_threshold,
            "no_speech_threshold": args.no_speech_threshold,
            "hallucination_silence_threshold": args.hallucination_silence_threshold,
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
        }
        model = whisperx.load_model(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=model_language,
            threads=args.threads,
            asr_options=asr_options,
        )

        align_model = None
        align_metadata = None

        try:
            with progress:
                stream_task = progress.add_task("Streams".ljust(25), total=len(unique_streams))
                file_task = progress.add_task("Files".ljust(25), total=len(selected_rows))
                for stream_id in unique_streams:
                    stream_rows = [row for row in selected_rows if row["stream_id"] == stream_id]
                    progress.update(stream_task, description=f"Streams ({stream_id})".ljust(25))
                    progress.update(file_task, description="Files".ljust(25))
                    summary.setdefault(stream_id, {"files": 0, "done": 0, "empty": 0, "skipped": 0, "failed": 0})
                    output_dir = output_dirs[stream_id]
                    writer = writers[stream_id]
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
                            try:
                                audio = whisperx.load_audio(row["path"])
                                result = model.transcribe(
                                    audio,
                                    batch_size=args.batch_size,
                                    chunk_size=args.chunk_size,
                                    language=model_language,
                                    print_progress=args.print_progress,
                                    verbose=False,
                                )
                                if not args.no_align and len(result.get("segments", [])) > 0:
                                    result_language = result.get("language") or model_language
                                    if result_language and (
                                        align_model is None or align_metadata is None or align_metadata.get("language") != result_language
                                    ):
                                        align_model, align_metadata = whisperx.load_align_model(
                                            language_code=result_language,
                                            device=args.device,
                                            model_dir=None,
                                        )
                                    if align_model is not None and align_metadata is not None:
                                        result = whisperx.align(
                                            result["segments"],
                                            align_model,
                                            align_metadata,
                                            audio,
                                            args.device,
                                            return_char_alignments=False,
                                            print_progress=args.print_progress,
                                        )
                                writer(result, row["path"], WRITER_OPTIONS)
                                has_segments = None
                                if args.output_format in {"json", "all"}:
                                    has_segments = json_has_segments(expected_path)
                                if has_segments is False:
                                    status = "done_empty"
                                    summary[stream_id]["empty"] += 1
                                else:
                                    summary[stream_id]["done"] += 1
                            except Exception as exc:
                                status = "failed"
                                summary[stream_id]["failed"] += 1
                                error_message = str(exc)
                            finally:
                                cleanup_device_memory(args.device)
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
            del model
            if align_model is not None:
                del align_model
            cleanup_device_memory(args.device)
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
        stats = summary.get(stream_id, {"files": 0, "done": 0, "empty": 0, "skipped": 0, "failed": 0})
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
