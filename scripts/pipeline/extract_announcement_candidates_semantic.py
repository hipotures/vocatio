#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

import demo_semantic_announcement_classifier as demo


console = Console()

OUTPUT_HEADERS = [
    "day",
    "stream_id",
    "device",
    "filename",
    "clip_path",
    "transcript_path",
    "segment_index",
    "segment_start_seconds",
    "segment_end_seconds",
    "segment_start_local",
    "segment_end_local",
    "performance_number",
    "match_keyword",
    "match_method",
    "matched_phrase",
    "confidence",
    "language",
    "segment_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract semantic announcement candidates from transcripts using Codex/OpenAI-compatible classification."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260324")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video_synced.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--merged-csv",
        help="Synced video CSV path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    parser.add_argument(
        "--transcripts-root",
        help="Transcript root directory. Default: DAY/_workspace/transcripts",
    )
    parser.add_argument(
        "--output",
        default="announcement_candidates_semantic.csv",
        help="Output filename inside workspace or absolute path. Default: announcement_candidates_semantic.csv",
    )
    parser.add_argument(
        "--streams",
        nargs="*",
        help='Specific transcript stream IDs to parse, for example "v-pocket3" "v-gh7"',
    )
    parser.add_argument(
        "--all-streams",
        action="store_true",
        help="Parse transcripts from every available stream directory",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available transcript stream IDs and exit",
    )
    parser.add_argument(
        "--filenames",
        nargs="*",
        help="Optional exact video filenames to include, for example clip.mov",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit for the number of transcript JSON files to parse after filtering",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Optional limit for total semantic windows after de-duplication",
    )
    parser.add_argument(
        "--keyword-window-before",
        type=int,
        default=2,
        help="Segments to include before a keyword-triggered segment. Default: 2",
    )
    parser.add_argument(
        "--keyword-window-after",
        type=int,
        default=2,
        help="Segments to include after a keyword-triggered segment. Default: 2",
    )
    parser.add_argument(
        "--trim-adjacent-trigger-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim keyword/category windows so they stop before adjacent trigger segments and focus on one local announcement. Default: enabled",
    )
    parser.add_argument(
        "--backend",
        choices=("openai-compatible", "local-openai", "codex-exec", "qwen-local"),
        default="codex-exec",
        help="Classification backend. Use local-openai for the local OpenAI-compatible preset. qwen-local is kept as a legacy alias. Default: codex-exec",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help='Model name for the selected backend. Default: "gpt-5.4"',
    )
    parser.add_argument(
        "--api-base-url",
        default=demo.DEFAULT_API_BASE_URL,
        help=f"Base URL for the OpenAI-compatible endpoint. Default: {demo.DEFAULT_API_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Chat completion temperature. Default: 0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for the OpenAI-compatible backend. Default: 1024",
    )
    parser.add_argument(
        "--response-format-mode",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help="Structured output mode for the OpenAI-compatible backend. Default: json_schema",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a strict information extraction system for Polish dance competition transcripts. "
            "Return only valid JSON that matches the requested schema. "
            "If there is no real competition announcement in the provided window, return an empty detections list."
        ),
        help="System prompt passed to the classifier.",
    )
    parser.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help='Extra codex exec -c key=value options. Default includes model_reasoning_effort="medium".',
    )
    parser.add_argument(
        "--codex-output-dir",
        help="Directory for temporary codex exec output files. Default: WORKSPACE/codex_exec_semantic_candidates",
    )
    parser.add_argument(
        "--codex-schema",
        default=demo.DEFAULT_CODEX_SCHEMA,
        help=f"Output schema path for codex exec. Default: {demo.DEFAULT_CODEX_SCHEMA}",
    )
    parser.add_argument(
        "--codex-batch-schema",
        default=demo.DEFAULT_CODEX_BATCH_SCHEMA,
        help=f"Batch output schema path for codex exec. Default: {demo.DEFAULT_CODEX_BATCH_SCHEMA}",
    )
    parser.add_argument(
        "--codex-batch-size",
        type=int,
        default=10,
        help="Number of windows per codex exec request. Default: 10",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable structured response_format in the chat completion request.",
    )
    return parser.parse_args()


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def select_window_segment(window: Dict, detection: Dict) -> Dict:
    segments = list(window.get("segments", []))
    if not segments:
        raise ValueError(f"window {window['window_id']} has no segments")
    evidence = str(detection.get("evidence") or "").strip()
    normalized_evidence = demo.normalize_text(evidence)
    if normalized_evidence:
        for segment in segments:
            if normalized_evidence and normalized_evidence in demo.normalize_text(segment["text"]):
                return segment
    start_number = detection.get("start_number")
    if start_number is not None:
        needle = f"numer {start_number}"
        alt_needle = f"numer startowy {start_number}"
        for segment in segments:
            normalized_segment = demo.normalize_text(segment["text"])
            if needle in normalized_segment or alt_needle in normalized_segment:
                return segment
    center_segment_indices = set(window.get("center_segment_indices", []))
    for segment in segments:
        if segment["segment_index"] in center_segment_indices:
            return segment
    return segments[0]


def classify_batch(
    args: argparse.Namespace,
    batch: Sequence[Dict],
    codex_output_dir: Path,
    codex_schema: Path,
    codex_batch_schema: Path,
    codex_config_overrides: Sequence[str],
) -> Tuple[str, str, Dict, Dict[str, List[Dict]]]:
    response_status = "ok"
    response_error = ""
    raw_response_text = ""
    response_payload: Dict = {}
    detections_by_window: Dict[str, List[Dict]] = {window["window_id"]: [] for window in batch}
    response_format = demo.build_openai_response_format("none" if args.no_json_mode else args.response_format_mode)
    try:
        if args.backend == "codex-exec":
            if len(batch) > 1:
                raw_response_text = demo.run_codex_exec(
                    args.system_prompt,
                    demo.build_batch_prompt(batch),
                    codex_output_dir / f"batch_{batch[0]['window_id'].replace(':', '_')}",
                    codex_batch_schema,
                    args.model,
                    args.timeout_seconds,
                    codex_config_overrides,
                )
                response_payload = demo.extract_json_payload(raw_response_text, expect_batch=True)
                raw_results = response_payload.get("results", [])
                if not isinstance(raw_results, list):
                    return "error", "response field 'results' is not a list", response_payload, detections_by_window
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    window_id = str(item.get("window_id", "")).strip()
                    if window_id not in detections_by_window:
                        continue
                    raw_detections = item.get("detections", [])
                    if not isinstance(raw_detections, list):
                        continue
                    detections = [
                        demo.normalize_detection(detection)
                        for detection in raw_detections
                        if isinstance(detection, dict)
                    ]
                    detections_by_window[window_id] = [
                        detection for detection in detections if not demo.is_junk_detection(detection)
                    ]
                return response_status, response_error, response_payload, detections_by_window

            window = batch[0]
            raw_response_text = demo.run_codex_exec(
                args.system_prompt,
                window["prompt"],
                codex_output_dir / window["window_id"].replace(":", "_"),
                codex_schema,
                args.model,
                args.timeout_seconds,
                codex_config_overrides,
            )
            response_payload = demo.extract_json_payload(raw_response_text, expected_window_id=window["window_id"])
            raw_detections = response_payload.get("detections", [])
            if not isinstance(raw_detections, list):
                return "error", "response field 'detections' is not a list", response_payload, detections_by_window
            detections = [
                demo.normalize_detection(detection)
                for detection in raw_detections
                if isinstance(detection, dict)
            ]
            detections_by_window[window["window_id"]] = [
                detection for detection in detections if not demo.is_junk_detection(detection)
            ]
            return response_status, response_error, response_payload, detections_by_window

        for window in batch:
            payload = demo.post_chat_completion(
                args.api_base_url,
                args.api_key,
                args.model,
                args.system_prompt,
                window["prompt"],
                args.temperature,
                args.timeout_seconds,
                args.max_output_tokens,
                response_format,
            )
            choice = payload["choices"][0]
            message = choice["message"]
            raw_response_text = str(message.get("content", ""))
            response_payload = demo.extract_json_payload(raw_response_text, expected_window_id=window["window_id"])
            raw_detections = response_payload.get("detections", [])
            if not isinstance(raw_detections, list):
                return "error", "response field 'detections' is not a list", response_payload, detections_by_window
            detections = [
                demo.normalize_detection(detection)
                for detection in raw_detections
                if isinstance(detection, dict)
            ]
            detections_by_window[window["window_id"]] = [
                detection for detection in detections if not demo.is_junk_detection(detection)
            ]
        return response_status, response_error, response_payload, detections_by_window
    except Exception as exc:
        return "error", str(exc), response_payload, detections_by_window


def build_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Semantic Announcement Candidate Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Windows", justify="right", style="yellow")
    table.add_column("Candidates", justify="right", style="cyan")
    table.add_column("Unique Numbers", justify="right", style="red")
    table.add_column("First Candidate", style="white")
    table.add_column("Last Candidate", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def main() -> int:
    args = parse_args()
    args = demo.apply_backend_preset(args)
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not demo.DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260324, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video_synced.csv"
    transcripts_root = Path(args.transcripts_root).resolve() if args.transcripts_root else workspace_dir / "transcripts"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    if not merged_csv.exists():
        console.print(f"[red]Error: merged synced video CSV not found: {merged_csv}[/red]")
        return 1
    if not transcripts_root.exists() or not transcripts_root.is_dir():
        console.print(f"[red]Error: transcripts root not found: {transcripts_root}[/red]")
        return 1

    available_streams = demo.detect_stream_ids(transcripts_root)
    if args.list_streams:
        if available_streams:
            for stream_id in available_streams:
                console.print(stream_id)
            return 0
        console.print("[yellow]No transcript stream directories found.[/yellow]")
        return 0
    if not available_streams:
        console.print("[red]Error: no transcript stream directories found.[/red]")
        return 1

    selected_streams = demo.select_stream_ids(available_streams, args.streams, args.all_streams)
    jobs = demo.collect_transcript_jobs(transcripts_root, selected_streams, args.max_files)
    if not jobs:
        console.print("[red]Error: no transcript JSON files selected.[/red]")
        return 1

    merged_rows = demo.read_csv_rows(merged_csv)
    video_index = demo.load_video_index(merged_rows)
    filename_filters = set(args.filenames or [])
    file_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    windows_by_stream: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    languages_by_clip: Dict[Tuple[str, str], str] = {}
    windows: List[Dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    ) as progress:
        streams_task = progress.add_task("Streams".ljust(25), total=len(selected_streams))
        files_task = progress.add_task("Files".ljust(25), total=len(jobs))
        for stream_id in selected_streams:
            stream_jobs = [job for job in jobs if job[0] == stream_id]
            for _, transcript_path in stream_jobs:
                transcript_stem = transcript_path.stem
                video_row = video_index.get((stream_id, transcript_stem))
                if video_row is None:
                    progress.advance(files_task)
                    continue
                if filename_filters and video_row.get("filename", "") not in filename_filters:
                    progress.advance(files_task)
                    continue
                start_synced = demo.parse_local_datetime(video_row.get("start_synced", ""))
                if start_synced is None:
                    progress.advance(files_task)
                    continue
                try:
                    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
                except Exception:
                    progress.advance(files_task)
                    continue

                key = (stream_id, video_row["filename"])
                languages_by_clip[key] = str(payload.get("language", ""))
                file_counts[stream_id] += 1
                segments_raw = payload.get("segments", [])
                if not isinstance(segments_raw, list):
                    segments_raw = []
                segments: List[Dict] = []
                for segment_index, segment in enumerate(segments_raw, 1):
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        segment_start_seconds = float(segment.get("start", 0.0))
                        segment_end_seconds = float(segment.get("end", segment_start_seconds))
                    except (TypeError, ValueError):
                        continue
                    segment_start_local = start_synced + demo.timedelta(seconds=segment_start_seconds)
                    segment_end_local = start_synced + demo.timedelta(seconds=segment_end_seconds)
                    trigger_types, matched_terms = demo.text_trigger_types(text)
                    segments.append(
                        {
                            "segment_index": segment_index,
                            "segment_start_seconds": segment_start_seconds,
                            "segment_end_seconds": segment_end_seconds,
                            "segment_start_local": demo.format_datetime(segment_start_local),
                            "segment_end_local": demo.format_datetime(segment_end_local),
                            "text": text,
                            "trigger_types": trigger_types,
                            "matched_terms": matched_terms,
                        }
                    )
                clip_windows = demo.build_windows_for_clip(
                    day_dir.name,
                    stream_id,
                    video_row["filename"],
                    video_row.get("path", ""),
                    str(transcript_path),
                    segments,
                    args.keyword_window_before,
                    args.keyword_window_after,
                    sliding_window_size=5,
                    sliding_step=2,
                    keyword_only=True,
                    trim_adjacent_trigger_windows=args.trim_adjacent_trigger_windows,
                )
                for clip_window_index, window in enumerate(clip_windows, 1):
                    window["window_id"] = (
                        f"{stream_id}:{Path(video_row['filename']).stem}:{window['segment_start_index']:04d}:"
                        f"{window['segment_end_index']:04d}"
                    )
                    window["clip_window_index"] = clip_window_index
                    window["prompt"] = demo.build_prompt(window)
                    windows.append(window)
                progress.advance(files_task)
            progress.advance(streams_task)

    windows = demo.dedupe_windows(windows)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        console.print("[red]Error: no semantic windows selected.[/red]")
        return 1

    codex_output_dir = (
        Path(args.codex_output_dir).resolve()
        if args.codex_output_dir
        else workspace_dir / "codex_exec_semantic_candidates"
    )
    codex_schema = Path(args.codex_schema).resolve()
    codex_batch_schema = Path(args.codex_batch_schema).resolve()
    if args.backend == "codex-exec" and not codex_schema.exists():
        console.print(f"[red]Error: codex exec output schema not found: {codex_schema}[/red]")
        return 1
    if args.backend == "codex-exec" and args.codex_batch_size > 1 and not codex_batch_schema.exists():
        console.print(f"[red]Error: codex exec batch output schema not found: {codex_batch_schema}[/red]")
        return 1

    codex_config_overrides = ['model_reasoning_effort="medium"']
    codex_config_overrides.extend(args.codex_config)

    candidate_rows: List[Dict[str, str]] = []
    candidate_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    unique_numbers: Dict[str, set] = {stream_id: set() for stream_id in selected_streams}
    first_candidate: Dict[str, str] = {stream_id: "" for stream_id in selected_streams}
    last_candidate: Dict[str, str] = {stream_id: "" for stream_id in selected_streams}

    pending_windows = list(windows)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    ) as progress:
        windows_task = progress.add_task("Windows".ljust(25), total=len(windows))
        while pending_windows:
            if args.backend == "codex-exec" and args.codex_batch_size > 1:
                batch = pending_windows[: args.codex_batch_size]
                pending_windows = pending_windows[args.codex_batch_size :]
            else:
                batch = [pending_windows.pop(0)]

            response_status, response_error, _response_payload, detections_by_window = classify_batch(
                args,
                batch,
                codex_output_dir,
                codex_schema,
                codex_batch_schema,
                codex_config_overrides,
            )

            for window in batch:
                windows_by_stream[window["stream_id"]] += 1
                if response_status == "ok":
                    detections = detections_by_window.get(window["window_id"], [])
                    for detection in detections:
                        if detection.get("event_type") != "performance_announcement":
                            continue
                        start_number = detection.get("start_number")
                        if start_number is None:
                            continue
                        segment = select_window_segment(window, detection)
                        key = (window["stream_id"], window["filename"])
                        row = {
                            "day": window["day"],
                            "stream_id": window["stream_id"],
                            "device": window["stream_id"],
                            "filename": window["filename"],
                            "clip_path": window["clip_path"],
                            "transcript_path": window["transcript_path"],
                            "segment_index": str(segment["segment_index"]),
                            "segment_start_seconds": f"{segment['segment_start_seconds']:.3f}",
                            "segment_end_seconds": f"{segment['segment_end_seconds']:.3f}",
                            "segment_start_local": segment["segment_start_local"],
                            "segment_end_local": segment["segment_end_local"],
                            "performance_number": str(start_number),
                            "match_keyword": "semantic",
                            "match_method": f"semantic-{args.backend}",
                            "matched_phrase": str(detection.get("evidence") or ""),
                            "confidence": "" if detection.get("confidence") is None else f"{float(detection['confidence']):.2f}",
                            "language": languages_by_clip.get(key, ""),
                            "segment_text": segment["text"],
                        }
                        candidate_rows.append(row)
                        candidate_counts[window["stream_id"]] += 1
                        unique_numbers[window["stream_id"]].add(int(start_number))
                        if not first_candidate[window["stream_id"]] or row["segment_start_local"] < first_candidate[window["stream_id"]]:
                            first_candidate[window["stream_id"]] = row["segment_start_local"]
                        if not last_candidate[window["stream_id"]] or row["segment_start_local"] > last_candidate[window["stream_id"]]:
                            last_candidate[window["stream_id"]] = row["segment_start_local"]
                progress.advance(windows_task)

    candidate_rows.sort(
        key=lambda row: (
            row.get("segment_start_local", ""),
            row.get("stream_id", ""),
            row.get("filename", ""),
            int(row.get("segment_index") or 0),
            int(row.get("performance_number") or 0),
        )
    )
    written = write_csv(output_path, OUTPUT_HEADERS, candidate_rows)

    summary_rows: List[List[str]] = []
    for stream_id in selected_streams:
        summary_rows.append(
            [
                stream_id,
                str(file_counts.get(stream_id, 0)),
                str(windows_by_stream.get(stream_id, 0)),
                str(candidate_counts.get(stream_id, 0)),
                str(len(unique_numbers.get(stream_id, set()))),
                first_candidate.get(stream_id, ""),
                last_candidate.get(stream_id, ""),
            ]
        )

    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote {written} semantic candidate rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
