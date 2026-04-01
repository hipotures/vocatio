#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import subprocess
import time
import urllib.parse
import urllib.request
from collections import defaultdict
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

import copy_reviewed_set_assets as reviewed_sets
import demo_semantic_announcement_classifier as demo


console = Console()

DEFAULT_CASES_CSV = "/tmp/semantic_announcement_benchmark_cases.csv"
DEFAULT_CASES_JSONL = "/tmp/semantic_announcement_benchmark_cases.jsonl"
DEFAULT_RESULTS_CSV = "/tmp/semantic_announcement_benchmark_results.csv"
DEFAULT_RESULTS_JSONL = "/tmp/semantic_announcement_benchmark_results.jsonl"
PROMPT_PROFILES = ("compact", "standard", "full")
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OFFICIAL_CATEGORY_GUIDANCE = (
    "Official DWC class and genre names may appear in Polish or English, including Ballet, Repertoire, "
    "National and Folklore, Lyrical, Showstopper, Jazz, Contemporary, Acro, Tap, Step, Song and Dance, "
    "Street Dance, Commercial, balet, balet repertuarowy, taniec narodowy i ludowy, taniec liryczny, "
    "taniec wspolczesny, wystep wokalno-taneczny, taniec uliczny, and taniec komercyjny."
)
ANNOUNCEMENT_WINDOW_GUIDANCE = (
    "A valid announcement is often one short spoken line that is immediately followed by song lyrics, "
    "music-related text, soundtrack speech, mistranscribed English lyrics rendered as Polish, greetings, "
    "or generic ASR noise from the performance itself. Detect the announcement line itself and do not reject "
    "a valid announcement just because later lines in the same window are unrelated or clearly belong to the performance."
)
POLISH_NUMBER_GUIDANCE = (
    "Convert Polish number words to digits when they clearly refer to a competitor number. "
    "Examples: 'numer siedem' means start_number=7, 'numer sto dwa' means start_number=102, "
    "'numer sto osiemdziesiat' means start_number=180, and 'numer czterysta jedenascie' means start_number=411."
)
PROMPT_EXAMPLES = (
    "Examples:\n"
    '- Transcript: "Numer 182, kategoria Junior Solo Contemporary One, Silent Tide."\n'
    '  Correct JSON detection: {"event_type":"performance_announcement","start_number":182,"category_ordinal":null}\n'
    '- Transcript: "Numer siedem, kategoria Mini Jazz, Bright Lights. I got rhythm, I got music..."\n'
    '  Correct JSON detection: {"event_type":"performance_announcement","start_number":7,"category_ordinal":null}\n'
    '- Transcript: "Numer sto osiemdziesiat, kategoria Junior Solo Ballet Repertoire II, Aurora."\n'
    '  Correct JSON detection: {"event_type":"performance_announcement","start_number":180,"category_ordinal":null}\n'
    '- Transcript: "Pierwszy w kategorii numer z karty 412, Liam Frost."\n'
    '  Correct JSON detection: {"event_type":"performance_announcement","start_number":412,"category_ordinal":1}\n'
)

ORDINAL_TERMS = {
    "pierwszy": 1,
    "drugi": 2,
    "trzeci": 3,
    "czwarty": 4,
    "piaty": 5,
    "szosty": 6,
    "siodmy": 7,
    "osmy": 8,
    "dziewiaty": 9,
    "dziesiaty": 10,
}

CASE_HEADERS = [
    "case_id",
    "day",
    "stream_id",
    "filename",
    "window_id",
    "window_start_local",
    "window_end_local",
    "expected_event_type",
    "expected_start_number",
    "expected_category_ordinal",
    "expected_set_name",
    "expected_set_id",
    "difficulty_score",
    "anchor_delta_seconds",
    "trigger_types",
    "segment_text",
]

RESULT_HEADERS = [
    "model",
    "case_id",
    "day",
    "expected_set_name",
    "stream_id",
    "filename",
    "window_id",
    "expected_event_type",
    "expected_start_number",
    "expected_category_ordinal",
    "difficulty_score",
    "predicted_event_types",
    "predicted_start_numbers",
    "predicted_category_ordinals",
    "detection_count",
    "event_ok",
    "start_number_ok",
    "category_ordinal_ok",
    "strict_ok",
    "response_status",
    "response_error",
    "finish_reason",
    "content_length",
    "reasoning_length",
    "prompt_eval_count",
    "eval_count",
    "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reviewed benchmark cases and compare semantic announcement models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Build benchmark cases from reviewed final sets and announcement candidates.",
    )
    prepare_parser.add_argument(
        "day_dirs",
        nargs="+",
        help="One or more day directories such as /data/20260323",
    )
    prepare_parser.add_argument(
        "--workspace-dir",
        help="Override a single workspace directory. Only use with one day_dir.",
    )
    prepare_parser.add_argument(
        "--index-json",
        help="Override index JSON filename or absolute path. Default: semantic index if present, otherwise regular index.",
    )
    prepare_parser.add_argument(
        "--state-json",
        help="Override review state JSON filename or absolute path. Default: semantic state if present, otherwise regular state.",
    )
    prepare_parser.add_argument(
        "--announcements-csv",
        help="Override announcement candidates CSV filename or absolute path. Default: semantic candidates if present, otherwise regular candidates.",
    )
    prepare_parser.add_argument(
        "--merged-csv",
        help="Override merged synced video CSV filename or absolute path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    prepare_parser.add_argument(
        "--streams",
        nargs="*",
        help='Optional stream filter, for example "v-gh7" "v-pocket3". Default: all streams found in the candidates CSV.',
    )
    prepare_parser.add_argument(
        "--max-cases",
        type=int,
        default=50,
        help="Maximum number of benchmark cases to write. Default: 50",
    )
    prepare_parser.add_argument(
        "--per-day-limit",
        type=int,
        help="Optional cap per day before global limiting.",
    )
    prepare_parser.add_argument(
        "--sampling",
        choices=("hardest", "random", "range", "stratified", "list"),
        default="hardest",
        help="Benchmark case selection mode. Default: hardest",
    )
    prepare_parser.add_argument(
        "--set-range",
        help='Inclusive numeric set range for sampling=range, for example "100-110".',
    )
    prepare_parser.add_argument(
        "--set-list",
        help='Comma-separated set numbers for sampling=list, for example "160,152,235".',
    )
    prepare_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used by random and stratified sampling. Default: 42",
    )
    prepare_parser.add_argument(
        "--keyword-window-before",
        type=int,
        default=2,
        help="Segments to include before the anchor segment. Default: 2",
    )
    prepare_parser.add_argument(
        "--keyword-window-after",
        type=int,
        default=2,
        help="Segments to include after the anchor segment. Default: 2",
    )
    prepare_parser.add_argument(
        "--trim-adjacent-trigger-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim windows before adjacent trigger segments. Default: enabled",
    )
    prepare_parser.add_argument(
        "--max-match-seconds",
        type=float,
        default=300.0,
        help="Maximum allowed time delta between a reviewed set start and the matched announcement candidate. Default: 300",
    )
    prepare_parser.add_argument(
        "--min-word-score",
        type=float,
        default=0.0,
        help="If greater than 0, drop low-confidence transcript words below this threshold when word scores are available. Default: 0 (disabled)",
    )
    prepare_parser.add_argument(
        "--min-photo-count",
        type=int,
        default=1,
        help="Minimum reviewed photo count for a set to become a positive benchmark case. Default: 1",
    )
    prepare_parser.add_argument(
        "--output-csv",
        default=DEFAULT_CASES_CSV,
        help=f"Benchmark cases CSV path. Default: {DEFAULT_CASES_CSV}",
    )
    prepare_parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_CASES_JSONL,
        help=f"Benchmark cases JSONL path. Default: {DEFAULT_CASES_JSONL}",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one or more models on prepared benchmark cases and score them.",
    )
    run_parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_CASES_JSONL,
        help=f"Prepared benchmark cases JSONL path. Default: {DEFAULT_CASES_JSONL}",
    )
    run_parser.add_argument(
        "--output-csv",
        default=DEFAULT_RESULTS_CSV,
        help=f"Benchmark results CSV path. Default: {DEFAULT_RESULTS_CSV}",
    )
    run_parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_RESULTS_JSONL,
        help=f"Benchmark results JSONL path. Default: {DEFAULT_RESULTS_JSONL}",
    )
    run_parser.add_argument(
        "--prompt-profile",
        choices=PROMPT_PROFILES,
        default="compact",
        help="Prompt and output profile. compact minimizes output fields, standard adds evidence, full requests the richest JSON. Default: compact",
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names to test.",
    )
    run_parser.add_argument(
        "--backend",
        choices=("openai-compatible", "local-openai", "codex-exec", "qwen-local"),
        default="local-openai",
        help="Classification backend. Default: local-openai",
    )
    run_parser.add_argument(
        "--api-base-url",
        default=demo.DEFAULT_API_BASE_URL,
        help=f"Base URL for OpenAI-compatible backends. Default: {demo.DEFAULT_API_BASE_URL}",
    )
    run_parser.add_argument(
        "--manage-model-loading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Ollama-compatible local endpoints, stop previous benchmark models, warm up the selected model, confirm it via api/ps, and unload it after the run. Default: enabled",
    )
    run_parser.add_argument(
        "--ollama-keep-alive",
        default="30m",
        help='keep_alive value used for Ollama warmup requests. Default: "30m"',
    )
    run_parser.add_argument(
        "--ollama-load-poll-seconds",
        type=float,
        default=5.0,
        help="Seconds between api/ps polls while confirming load and unload. Default: 5",
    )
    run_parser.add_argument(
        "--ollama-load-poll-attempts",
        type=int,
        default=3,
        help="Number of api/ps confirmation polls for load and unload. Default: 3",
    )
    run_parser.add_argument(
        "--api-key",
        default="",
        help="API key for the OpenAI-compatible backend.",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Chat completion temperature. Default: 0",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Request timeout in seconds. Default: 120",
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for OpenAI-compatible backends. Default: 1024",
    )
    run_parser.add_argument(
        "--response-format-mode",
        choices=("json_schema", "json_object", "none"),
        default="none",
        help="Structured output mode for OpenAI-compatible backends. Default: none",
    )
    run_parser.add_argument(
        "--ollama-think",
        choices=("inherit", "false", "low", "medium", "high"),
        default="inherit",
        help='For Ollama OpenAI-compatible endpoints, map reasoning control to reasoning_effort. Use false to request none. Default: inherit',
    )
    run_parser.add_argument(
        "--ollama-num-predict",
        type=int,
        help="For Ollama-compatible local endpoints, pass options.num_predict. Default: disabled",
    )
    run_parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        help="For Ollama-compatible local endpoints, pass options.num_ctx. Default: disabled",
    )
    run_parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional limit for benchmark cases read from JSONL.",
    )
    run_parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-case OK/NOK details with set number and difficulty.",
    )
    run_parser.add_argument(
        "--system-prompt",
        default=(
            "You are a strict information extraction system for Polish dance competition transcripts. "
            "Return only valid JSON that matches the requested schema. "
            "A category ordinal such as 'pierwszy', 'drugi', 'trzeci', 'piaty', or 'szosty w tej kategorii' must never be used as start_number. "
            "Use start_number only for the explicit competitor number, for example from phrases like 'numer startowy 278' or 'numer 278'. "
            "Treat patterns like '* startowy 350' or 'start number 350' as explicit competitor numbers even if the preceding word was mistranscribed. "
            "Do not infer category_ordinal from class labels, genre names, roman numerals, or words like One or Two inside official DWC category titles. "
            "If there is no real competition announcement in the provided window, return an empty detections list."
        ),
        help="System prompt passed to the classifier.",
    )
    run_parser.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help='Extra codex exec -c key=value options. Default includes model_reasoning_effort="medium".',
    )
    run_parser.add_argument(
        "--codex-output-dir",
        default="/tmp/semantic_announcement_benchmark_codex",
        help="Directory for temporary codex exec output files. Default: /tmp/semantic_announcement_benchmark_codex",
    )
    run_parser.add_argument(
        "--codex-schema",
        default=demo.DEFAULT_CODEX_SCHEMA,
        help=f"Output schema path for codex exec. Default: {demo.DEFAULT_CODEX_SCHEMA}",
    )
    run_parser.add_argument(
        "--debug-response-dir",
        default="/tmp/vocatio-benchmark",
        help="Directory for per-case raw response dumps. Default: /tmp/vocatio-benchmark",
    )
    return parser.parse_args()


def resolve_day_workspace(day_dir: Path, override_workspace: Optional[str]) -> Path:
    if override_workspace:
        return Path(override_workspace).resolve()
    return day_dir / "_workspace"


def pick_existing_file(workspace_dir: Path, preferred: Sequence[str], fallback: Optional[str] = None) -> Optional[Path]:
    for name in preferred:
        path = workspace_dir / name
        if path.exists():
            return path
    if fallback:
        path = workspace_dir / fallback
        if path.exists():
            return path
    return None


def resolve_workspace_file(workspace_dir: Path, explicit: Optional[str], preferred: Sequence[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (workspace_dir / explicit)
    return pick_existing_file(workspace_dir, preferred)


def is_numeric_set_name(value: str) -> bool:
    text = str(value).strip()
    return bool(text) and text.isdigit()


def read_candidate_rows(path: Path, stream_filter: Optional[Sequence[str]]) -> List[Dict[str, str]]:
    rows = demo.read_csv_rows(path)
    if not stream_filter:
        return rows
    allowed = {item.strip() for item in stream_filter if item.strip()}
    return [row for row in rows if row.get("stream_id", "") in allowed]


def build_clip_segments(
    merged_csv: Path,
    candidate_rows: Sequence[Dict[str, str]],
    min_word_score: float,
) -> Tuple[Dict[Tuple[str, str], List[Dict]], Dict[Tuple[str, str], Dict[str, str]]]:
    merged_rows = demo.read_csv_rows(merged_csv)
    video_index = demo.load_video_index(merged_rows)
    transcript_path_index: Dict[Tuple[str, str], Path] = {}
    for row in candidate_rows:
        key = (row.get("stream_id", ""), row.get("filename", ""))
        transcript_path = Path(row.get("transcript_path", ""))
        if key[0] and key[1] and transcript_path.exists():
            transcript_path_index[key] = transcript_path
    needed_keys = {
        (row["stream_id"], Path(row["filename"]).stem)
        for row in candidate_rows
        if row.get("stream_id") and row.get("filename")
    }
    clip_segments: Dict[Tuple[str, str], List[Dict]] = {}
    clip_meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    for stream_id, transcript_stem in sorted(needed_keys):
        video_row = video_index.get((stream_id, transcript_stem))
        if video_row is None:
            continue
        transcript_json = transcript_path_index.get((stream_id, video_row["filename"]))
        if transcript_json is None or not transcript_json.exists():
            transcript_json = Path(
                str(video_row.get("path", "")).replace(f"/{stream_id}/", f"/_workspace/transcripts/{stream_id}/")
            ).with_suffix(".json")
        if not transcript_json.exists():
            continue
        start_synced = demo.parse_local_datetime(video_row.get("start_synced", ""))
        if start_synced is None:
            continue
        try:
            payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            continue
        key = (stream_id, video_row["filename"])
        clip_meta[key] = {
            "clip_path": video_row.get("path", ""),
            "transcript_path": str(transcript_json),
            "filename": video_row["filename"],
        }
        clip_rows: List[Dict] = []
        for segment_index, segment in enumerate(segments, 1):
            if not isinstance(segment, dict):
                continue
            text = demo.select_segment_text(segment, min_word_score)
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
            clip_rows.append(
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
        if clip_rows:
            clip_segments[key] = clip_rows
    return clip_segments, clip_meta


def infer_ordinal_from_text(text: str) -> Optional[int]:
    normalized = demo.normalize_text(text)
    tokens = normalized.split()
    for token in tokens:
        if token in ORDINAL_TERMS:
            return ORDINAL_TERMS[token]
    return None


def difficulty_score(window: Dict, day_candidate_rows: Sequence[Dict[str, str]], expected_number: int) -> int:
    score = 0
    normalized_text = demo.normalize_text(window["text"])
    explicit_number = str(expected_number) in normalized_text.split()
    if explicit_number:
        score += 1
    if infer_ordinal_from_text(window["text"]) is not None:
        score += 2
    if "category" in window["trigger_types"]:
        score += 1
    if len(window["center_segment_indices"]) > 1:
        score += 1
    start_dt = demo.parse_local_datetime(window["window_start_local"])
    if start_dt is not None:
        nearby_conflicts = 0
        for row in day_candidate_rows:
            if row.get("performance_number", "").strip() == str(expected_number):
                continue
            candidate_dt = demo.parse_local_datetime(row.get("segment_start_local", ""))
            if candidate_dt is None:
                continue
            if abs((candidate_dt - start_dt).total_seconds()) <= 90:
                nearby_conflicts += 1
        score += min(nearby_conflicts, 3)
    return score


def select_best_candidate(
    display_set: Dict,
    candidate_rows: Sequence[Dict[str, str]],
    max_match_seconds: float,
) -> Optional[Tuple[Dict[str, str], float]]:
    expected_number = str(display_set["display_name"]).strip()
    if not expected_number:
        return None
    anchor_text = display_set.get("first_photo_local") or display_set.get("performance_start_local", "")
    anchor_dt = demo.parse_local_datetime(anchor_text)
    if anchor_dt is None:
        return None
    matches: List[Tuple[Tuple[int, float, str, str], Dict[str, str], float]] = []
    for row in candidate_rows:
        if row.get("performance_number", "").strip() != expected_number:
            continue
        candidate_dt = demo.parse_local_datetime(row.get("segment_start_local", ""))
        if candidate_dt is None:
            continue
        delta = (candidate_dt - anchor_dt).total_seconds()
        abs_delta = abs(delta)
        if abs_delta > max_match_seconds:
            continue
        sort_key = (1 if delta > 0 else 0, abs_delta, row.get("stream_id", ""), row.get("filename", ""))
        matches.append((sort_key, row, delta))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    _sort_key, row, delta = matches[0]
    return row, delta


def build_case_window(
    day_name: str,
    candidate_row: Dict[str, str],
    clip_segments: Dict[Tuple[str, str], List[Dict]],
    clip_meta: Dict[Tuple[str, str], Dict[str, str]],
    keyword_before: int,
    keyword_after: int,
    trim_adjacent_trigger_windows: bool,
) -> Optional[Dict]:
    key = (candidate_row["stream_id"], candidate_row["filename"])
    segments = clip_segments.get(key)
    meta = clip_meta.get(key)
    if not segments or not meta:
        return None
    try:
        segment_index = int(candidate_row["segment_index"])
    except (TypeError, ValueError):
        return None
    positions = {segment["segment_index"]: position for position, segment in enumerate(segments)}
    position = positions.get(segment_index)
    if position is None:
        return None
    start_index = max(0, position - keyword_before)
    end_index = min(len(segments) - 1, position + keyword_after)
    if trim_adjacent_trigger_windows:
        for scan_index in range(position - 1, start_index - 1, -1):
            if segments[scan_index]["trigger_types"]:
                start_index = scan_index + 1
                break
        for scan_index in range(position + 1, end_index + 1):
            if segments[scan_index]["trigger_types"]:
                end_index = scan_index - 1
                break
        if start_index > end_index:
            start_index = position
            end_index = position
    window = demo.make_window(
        day_name,
        candidate_row["stream_id"],
        candidate_row["filename"],
        meta["clip_path"],
        meta["transcript_path"],
        segments,
        start_index,
        end_index,
        segments[position]["trigger_types"] or ["keyword"],
        segments[position]["matched_terms"],
        [segment_index],
    )
    window["window_id"] = (
        f"{candidate_row['stream_id']}:{Path(candidate_row['filename']).stem}:"
        f"{window['segment_start_index']:04d}:{window['segment_end_index']:04d}"
    )
    window["prompt"] = demo.build_prompt(window)
    return window


def build_benchmark_prompt(case: Dict, prompt_profile: str) -> str:
    transcript_lines = []
    for segment in case["segments"]:
        transcript_lines.append(
            f"- {segment['segment_start_local']} -> {segment['segment_end_local']} | "
            f"segment {segment['segment_index']}: {segment['text']}"
        )
    transcript = "\n".join(transcript_lines)
    if prompt_profile == "compact":
        schema_text = (
            "{\n"
            '  "window_id": string,\n'
            '  "detections": [\n'
            "    {\n"
            '      "event_type": "performance_announcement" | "ceremony" | "other",\n'
            '      "start_number": integer | null,\n'
            '      "category_ordinal": integer | null\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        rules = (
            "Return exactly one JSON object and nothing else.\n"
            "Keep the response short.\n"
            f"{ANNOUNCEMENT_WINDOW_GUIDANCE}\n"
            "Use start_number only for the explicit competitor number.\n"
            "Use category_ordinal only for order phrases such as pierwszy, drugi, trzeci, piaty, or szosty.\n"
            "Never use the ordinal as start_number.\n"
            f"{OFFICIAL_CATEGORY_GUIDANCE}\n"
            f"{POLISH_NUMBER_GUIDANCE}\n"
            "Do not infer category_ordinal from class labels, genre names, age labels, roman numerals, or words like One or Two inside category titles.\n"
            "Roman numerals or words like One, Two, I, II, III inside class names such as 'Junior Solo Contemporary I' or 'Children's Solo Acro One' are part of the class name and must not become category_ordinal.\n"
            "If the transcript contains a pattern like 'numer startowy N', 'start number N', or even a mistranscribed '* startowy N', treat N as the explicit competitor start_number.\n"
            "A short line such as 'Numer 221' is already a valid explicit announcement if it appears as an announcement line.\n"
            "A phrase like 'numer z karty 383' contains an explicit competitor number and must set start_number to that number.\n"
            "If the local transcript says 'Numer 152. Przepraszam. Numer 151.', prefer the corrected later number.\n"
            "If there is no real competition announcement, return an empty detections list.\n"
        )
    elif prompt_profile == "standard":
        schema_text = (
            "{\n"
            '  "window_id": string,\n'
            '  "detections": [\n'
            "    {\n"
            '      "event_type": "performance_announcement" | "ceremony" | "other",\n'
            '      "start_number": integer | null,\n'
            '      "category_ordinal": integer | null,\n'
            '      "evidence": string | null\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        rules = (
            "Return exactly one JSON object and nothing else.\n"
            f"{ANNOUNCEMENT_WINDOW_GUIDANCE}\n"
            "Use start_number only for the explicit competitor number.\n"
            "Use category_ordinal only for order phrases such as pierwszy, drugi, trzeci, piaty, or szosty.\n"
            "Never use the ordinal as start_number.\n"
            f"{OFFICIAL_CATEGORY_GUIDANCE}\n"
            f"{POLISH_NUMBER_GUIDANCE}\n"
            "Do not infer category_ordinal from class labels, genre names, age labels, roman numerals, or words like One or Two inside category titles.\n"
            "Roman numerals or words like One, Two, I, II, III inside class names such as 'Junior Solo Contemporary I' or 'Children's Solo Acro One' are part of the class name and must not become category_ordinal.\n"
            "If the transcript contains a pattern like 'numer startowy N', 'start number N', or even a mistranscribed '* startowy N', treat N as the explicit competitor start_number.\n"
            "A short line such as 'Numer 221' is already a valid explicit announcement if it appears as an announcement line.\n"
            "A phrase like 'numer z karty 383' contains an explicit competitor number and must set start_number to that number.\n"
            "If the local transcript says 'Numer 152. Przepraszam. Numer 151.', prefer the corrected later number.\n"
            "If there is no real competition announcement, return an empty detections list.\n"
            "Evidence must be a short direct quote from the transcript.\n"
        )
    else:
        schema_text = (
            "{\n"
            '  "window_id": string,\n'
            '  "detections": [\n'
            "    {\n"
            '      "event_type": "performance_announcement" | "ceremony" | "other",\n'
            '      "start_number": integer | null,\n'
            '      "category_ordinal": integer | null,\n'
            '      "title": string | null,\n'
            '      "evidence": string | null,\n'
            '      "notes": string | null,\n'
            '      "confidence": float | null\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        rules = (
            "Return exactly one JSON object and nothing else.\n"
            f"{ANNOUNCEMENT_WINDOW_GUIDANCE}\n"
            "Use start_number only for the explicit competitor number.\n"
            "Use category_ordinal only for order phrases such as pierwszy, drugi, trzeci, piaty, or szosty.\n"
            "Never use the ordinal as start_number.\n"
            f"{OFFICIAL_CATEGORY_GUIDANCE}\n"
            f"{POLISH_NUMBER_GUIDANCE}\n"
            "Do not infer category_ordinal from class labels, genre names, age labels, roman numerals, or words like One or Two inside category titles.\n"
            "Roman numerals or words like One, Two, I, II, III inside class names such as 'Junior Solo Contemporary I' or 'Children's Solo Acro One' are part of the class name and must not become category_ordinal.\n"
            "If the transcript contains a pattern like 'numer startowy N', 'start number N', or even a mistranscribed '* startowy N', treat N as the explicit competitor start_number.\n"
            "A short line such as 'Numer 221' is already a valid explicit announcement if it appears as an announcement line.\n"
            "A phrase like 'numer z karty 383' contains an explicit competitor number and must set start_number to that number.\n"
            "If the local transcript says 'Numer 152. Przepraszam. Numer 151.', prefer the corrected later number.\n"
            "If there is no real competition announcement, return an empty detections list.\n"
            "Title may be null if absent.\n"
            "Evidence must be a short direct quote from the transcript.\n"
            "Notes may be null.\n"
            "Confidence may be null.\n"
        )
    return (
        "Extract only real dance competition announcements from this local transcript window.\n"
        "Return valid JSON with this schema:\n"
        f"{schema_text}\n"
        "Rules:\n"
        f"{rules}\n"
        f"{PROMPT_EXAMPLES}\n"
        f"Window ID: {case['window_id']}\n"
        f"Stream: {case['stream_id']}\n"
        f"Filename: {case['filename']}\n"
        f"Window start: {case['window_start_local']}\n"
        f"Window end: {case['window_end_local']}\n"
        f"Trigger types: {case['trigger_types']}\n"
        "Transcript lines:\n"
        f"{transcript}\n"
    )


def build_benchmark_response_format(response_format_mode: str, prompt_profile: str) -> Optional[Dict]:
    if response_format_mode == "none":
        return None
    if response_format_mode == "json_object":
        return {"type": "json_object"}
    if prompt_profile == "compact":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "window_id": {"type": "string"},
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "event_type": {
                                "type": "string",
                                "enum": ["performance_announcement", "ceremony", "other"],
                            },
                            "start_number": {"type": ["integer", "null"]},
                            "category_ordinal": {"type": ["integer", "null"]},
                        },
                        "required": ["event_type", "start_number", "category_ordinal"],
                    },
                },
            },
            "required": ["window_id", "detections"],
        }
    elif prompt_profile == "standard":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "window_id": {"type": "string"},
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "event_type": {
                                "type": "string",
                                "enum": ["performance_announcement", "ceremony", "other"],
                            },
                            "start_number": {"type": ["integer", "null"]},
                            "category_ordinal": {"type": ["integer", "null"]},
                            "evidence": {"type": ["string", "null"]},
                        },
                        "required": ["event_type", "start_number", "category_ordinal", "evidence"],
                    },
                },
            },
            "required": ["window_id", "detections"],
        }
    else:
        return demo.build_openai_response_format("json_schema")
    return {"type": "json_schema", "schema": schema}


def build_ollama_extra_body(args: argparse.Namespace) -> Dict:
    extra_body: Dict[str, object] = {}
    if not is_ollama_api_base_url(args.api_base_url):
        return extra_body
    if args.ollama_think != "inherit":
        reasoning_effort = "none" if args.ollama_think == "false" else args.ollama_think
        extra_body["reasoning_effort"] = reasoning_effort
        extra_body["reasoning"] = {"effort": reasoning_effort}
    options: Dict[str, int] = {}
    if args.ollama_num_predict is not None:
        options["num_predict"] = args.ollama_num_predict
    elif args.max_output_tokens:
        options["num_predict"] = args.max_output_tokens
    if args.ollama_num_ctx is not None:
        options["num_ctx"] = args.ollama_num_ctx
    if options:
        extra_body["options"] = options
    return extra_body


def extract_backend_metrics(backend_payload: Dict) -> Dict[str, object]:
    metrics = {
        "finish_reason": "",
        "content_length": 0,
        "reasoning_length": 0,
        "prompt_eval_count": "",
        "eval_count": "",
    }
    if not isinstance(backend_payload, dict):
        return metrics
    choices = backend_payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        metrics["finish_reason"] = str(choice.get("finish_reason", "") or "")
        message = choice.get("message", {})
        if isinstance(message, dict):
            metrics["content_length"] = len(str(message.get("content", "") or ""))
            reasoning_text = ""
            for key in ("reasoning", "reasoning_content", "thinking"):
                value = message.get(key)
                if value:
                    reasoning_text = str(value)
                    break
            metrics["reasoning_length"] = len(reasoning_text)
    usage = backend_payload.get("usage", {})
    if isinstance(usage, dict):
        prompt_eval_count = usage.get("prompt_eval_count")
        eval_count = usage.get("eval_count")
        if prompt_eval_count not in (None, ""):
            metrics["prompt_eval_count"] = prompt_eval_count
        if eval_count not in (None, ""):
            metrics["eval_count"] = eval_count
    return metrics


def strip_v1_suffix(api_base_url: str) -> str:
    text = api_base_url.rstrip("/")
    if text.endswith("/v1"):
        return text[:-3]
    return text


def is_ollama_api_base_url(api_base_url: str) -> bool:
    parsed = urllib.parse.urlparse(api_base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return False
    if parsed.port != 11434:
        return False
    return True


def ollama_post_json(base_url: str, path: str, payload: Dict, timeout_seconds: float) -> Dict:
    request = urllib.request.Request(
        strip_v1_suffix(base_url) + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_get_json(base_url: str, path: str, timeout_seconds: float) -> Dict:
    with urllib.request.urlopen(strip_v1_suffix(base_url) + path, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_stop_model(model_name: str) -> None:
    subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True, check=False)


def ollama_ps_names(base_url: str, timeout_seconds: float) -> List[str]:
    payload = ollama_get_json(base_url, "/api/ps", timeout_seconds)
    return [str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")]


def prepare_ollama_model(
    api_base_url: str,
    model_name: str,
    benchmark_models: Sequence[str],
    keep_alive: str,
    poll_seconds: float,
    poll_attempts: int,
    timeout_seconds: float,
) -> None:
    for candidate in benchmark_models:
        ollama_stop_model(candidate)
    ollama_post_json(
        api_base_url,
        "/api/generate",
        {
            "model": model_name,
            "prompt": "ok",
            "stream": False,
            "keep_alive": keep_alive,
        },
        timeout_seconds,
    )
    for _attempt in range(poll_attempts):
        time.sleep(poll_seconds)
        loaded_models = ollama_ps_names(api_base_url, timeout_seconds)
        if model_name in loaded_models:
            return
    raise RuntimeError(f'failed to confirm Ollama model load for "{model_name}"')


def unload_ollama_model(
    api_base_url: str,
    model_name: str,
    poll_seconds: float,
    poll_attempts: int,
    timeout_seconds: float,
) -> None:
    ollama_stop_model(model_name)
    for _attempt in range(poll_attempts):
        time.sleep(poll_seconds)
        loaded_models = ollama_ps_names(api_base_url, timeout_seconds)
        if model_name not in loaded_models:
            return
    raise RuntimeError(f'failed to confirm Ollama model unload for "{model_name}"')


def limit_cases(cases: Sequence[Dict], max_cases: int, per_day_limit: Optional[int]) -> List[Dict]:
    if max_cases <= 0:
        return []
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    for case in cases:
        by_day[case["day"]].append(case)
    selected: List[Dict] = []
    effective_per_day_limit = per_day_limit
    if effective_per_day_limit is None and len(by_day) > 1:
        effective_per_day_limit = max(1, math.ceil(max_cases / len(by_day)))
    for day_name in sorted(by_day):
        rows = sorted(
            by_day[day_name],
            key=lambda item: (-int(item["difficulty_score"]), item["window_start_local"], item["case_id"]),
        )
        if effective_per_day_limit is not None:
            rows = rows[:effective_per_day_limit]
        selected.extend(rows)
    selected.sort(key=lambda item: (-int(item["difficulty_score"]), item["day"], item["window_start_local"], item["case_id"]))
    return selected[:max_cases]


def parse_set_range(value: str) -> Tuple[int, int]:
    parts = value.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f'invalid set range "{value}"')
    start_text, end_text = parts[0].strip(), parts[1].strip()
    if not start_text.isdigit() or not end_text.isdigit():
        raise ValueError(f'invalid set range "{value}"')
    start_value = int(start_text)
    end_value = int(end_text)
    if end_value <= start_value:
        raise ValueError(f'invalid set range "{value}": expected N2 > N1')
    return start_value, end_value


def parse_set_list(value: str) -> List[int]:
    raw_items = value.replace(",", " ").split()
    if not raw_items:
        raise ValueError('invalid set list: expected comma-separated numbers, for example "160,152,235"')
    parsed: List[int] = []
    for item in raw_items:
        if not item.isdigit():
            raise ValueError(f'invalid set number "{item}" in set list')
        parsed.append(int(item))
    seen = set()
    ordered_unique: List[int] = []
    for number in parsed:
        if number in seen:
            continue
        seen.add(number)
        ordered_unique.append(number)
    return ordered_unique


def sort_cases_by_difficulty(cases: Sequence[Dict]) -> List[Dict]:
    return sorted(
        cases,
        key=lambda item: (-int(item["difficulty_score"]), item["day"], int(item["expected_start_number"]), item["case_id"]),
    )


def stratified_pick(cases: Sequence[Dict], limit: int, rng: random.Random) -> List[Dict]:
    if limit <= 0 or not cases:
        return []
    ordered = sort_cases_by_difficulty(cases)
    if limit >= len(ordered):
        return ordered
    bucket_count = min(3, len(ordered))
    buckets: List[List[Dict]] = []
    for bucket_index in range(bucket_count):
        start_index = math.floor(bucket_index * len(ordered) / bucket_count)
        end_index = math.floor((bucket_index + 1) * len(ordered) / bucket_count)
        buckets.append(ordered[start_index:end_index])
    target_per_bucket = max(1, math.ceil(limit / bucket_count))
    selected: List[Dict] = []
    selected_ids = set()
    for bucket in buckets:
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        for case in bucket_copy[:target_per_bucket]:
            if case["case_id"] in selected_ids:
                continue
            selected.append(case)
            selected_ids.add(case["case_id"])
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        remainder = [case for case in ordered if case["case_id"] not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    return selected[:limit]


def select_cases(
    cases: Sequence[Dict],
    max_cases: int,
    per_day_limit: Optional[int],
    sampling: str,
    random_seed: int,
    set_range: Optional[str],
    set_list: Optional[str],
) -> List[Dict]:
    if max_cases <= 0:
        return []
    filtered_cases = list(cases)
    if sampling == "range":
        if not set_range:
            raise ValueError("--set-range is required when --sampling range is used")
        start_value, end_value = parse_set_range(set_range)
        filtered_cases = [
            case
            for case in filtered_cases
            if start_value <= int(case["expected_start_number"]) <= end_value
        ]
    elif sampling == "list":
        if not set_list:
            raise ValueError("--set-list is required when --sampling list is used")
        requested = parse_set_list(set_list)
        allowed = set(requested)
        filtered_cases = [
            case
            for case in filtered_cases
            if int(case["expected_start_number"]) in allowed
        ]
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    for case in filtered_cases:
        by_day[case["day"]].append(case)
    effective_per_day_limit = per_day_limit
    if sampling == "list":
        effective_per_day_limit = None
    if effective_per_day_limit is None and len(by_day) > 1:
        effective_per_day_limit = max(1, math.ceil(max_cases / len(by_day)))
    rng = random.Random(random_seed)
    selected: List[Dict] = []
    for day_name in sorted(by_day):
        day_cases = list(by_day[day_name])
        if sampling == "hardest":
            day_selected = sort_cases_by_difficulty(day_cases)
        elif sampling == "random":
            day_selected = list(day_cases)
            rng.shuffle(day_selected)
        elif sampling == "range":
            day_selected = sorted(day_cases, key=lambda item: (int(item["expected_start_number"]), item["window_start_local"], item["case_id"]))
        elif sampling == "stratified":
            local_limit = effective_per_day_limit if effective_per_day_limit is not None else len(day_cases)
            day_selected = stratified_pick(day_cases, local_limit, rng)
        elif sampling == "list":
            day_selected = list(day_cases)
        else:
            raise ValueError(f"unsupported sampling mode: {sampling}")
        if effective_per_day_limit is not None and sampling != "stratified":
            day_selected = day_selected[:effective_per_day_limit]
        selected.extend(day_selected)
    if sampling == "random":
        rng.shuffle(selected)
    elif sampling == "hardest":
        selected = sort_cases_by_difficulty(selected)
    elif sampling == "range":
        selected = sorted(selected, key=lambda item: (item["day"], int(item["expected_start_number"]), item["window_start_local"], item["case_id"]))
    elif sampling == "stratified":
        selected = sorted(selected, key=lambda item: (-int(item["difficulty_score"]), item["day"], item["window_start_local"], item["case_id"]))
    elif sampling == "list":
        requested = parse_set_list(set_list or "")
        order = {number: index for index, number in enumerate(requested)}
        selected = sorted(
            selected,
            key=lambda item: (
                order.get(int(item["expected_start_number"]), 10**9),
                item["day"],
                item["window_start_local"],
                item["case_id"],
            ),
        )
    return selected[:max_cases]


def prepare_cases(args: argparse.Namespace) -> int:
    if args.workspace_dir and len(args.day_dirs) != 1:
        console.print("[red]Error: --workspace-dir can only be used with one day directory.[/red]")
        return 1
    all_cases: List[Dict] = []
    summary_rows: List[List[str]] = []
    for day_value in args.day_dirs:
        day_dir = Path(day_value).resolve()
        if not day_dir.exists() or not day_dir.is_dir():
            console.print(f"[red]Error: {day_value} is not a directory.[/red]")
            return 1
        workspace_dir = resolve_day_workspace(day_dir, args.workspace_dir)
        index_json = resolve_workspace_file(
            workspace_dir,
            args.index_json,
            ("performance_proxy_index_semantic.json", "performance_proxy_index.json"),
        )
        state_json = resolve_workspace_file(
            workspace_dir,
            args.state_json,
            ("review_state_semantic.json", "review_state.json"),
        )
        announcements_csv = resolve_workspace_file(
            workspace_dir,
            args.announcements_csv,
            ("announcement_candidates_semantic.csv", "announcement_candidates.csv"),
        )
        merged_csv = resolve_workspace_file(
            workspace_dir,
            args.merged_csv,
            ("merged_video_synced.csv",),
        )
        if not index_json or not index_json.exists():
            console.print(f"[red]Error: reviewed index JSON not found for {day_dir}.[/red]")
            return 1
        if not state_json or not state_json.exists():
            console.print(f"[red]Error: review state JSON not found for {day_dir}.[/red]")
            return 1
        if not announcements_csv or not announcements_csv.exists():
            console.print(f"[red]Error: announcement candidates CSV not found for {day_dir}.[/red]")
            return 1
        if not merged_csv or not merged_csv.exists():
            console.print(f"[red]Error: merged synced video CSV not found for {day_dir}.[/red]")
            return 1

        raw_index = reviewed_sets.load_json(index_json)
        raw_performances = raw_index.get("performances", [])
        review_state = reviewed_sets.load_review_state(state_json)
        display_sets = reviewed_sets.rebuild_display_sets(raw_performances, review_state)
        candidate_rows = read_candidate_rows(announcements_csv, args.streams)
        clip_segments, clip_meta = build_clip_segments(merged_csv, candidate_rows, args.min_word_score)

        total_numeric_sets = 0
        matched_sets = 0
        skipped_sets = 0
        day_cases: List[Dict] = []
        for display_set in display_sets:
            if not is_numeric_set_name(display_set.get("display_name", "")):
                continue
            if int(display_set.get("photo_count", 0)) < args.min_photo_count:
                continue
            total_numeric_sets += 1
            match = select_best_candidate(display_set, candidate_rows, args.max_match_seconds)
            if match is None:
                skipped_sets += 1
                continue
            candidate_row, delta_seconds = match
            window = build_case_window(
                day_dir.name,
                candidate_row,
                clip_segments,
                clip_meta,
                args.keyword_window_before,
                args.keyword_window_after,
                args.trim_adjacent_trigger_windows,
            )
            if window is None:
                skipped_sets += 1
                continue
            expected_start_number = int(display_set["display_name"])
            case = {
                "case_id": f"{day_dir.name}:{display_set['display_name']}:{window['window_id']}",
                "day": day_dir.name,
                "stream_id": window["stream_id"],
                "filename": window["filename"],
                "window_id": window["window_id"],
                "window_start_local": window["window_start_local"],
                "window_end_local": window["window_end_local"],
                "expected_event_type": "performance_announcement",
                "expected_start_number": expected_start_number,
                "expected_category_ordinal": infer_ordinal_from_text(window["text"]),
                "expected_set_name": display_set["display_name"],
                "expected_set_id": display_set["set_id"],
                "difficulty_score": difficulty_score(window, candidate_rows, expected_start_number),
                "anchor_delta_seconds": round(delta_seconds, 3),
                "trigger_types": ",".join(window["trigger_types"]),
                "segment_text": window["text"],
                "prompt": window["prompt"],
                "segments": window["segments"],
                "clip_path": window["clip_path"],
                "transcript_path": window["transcript_path"],
            }
            day_cases.append(case)
            matched_sets += 1
        summary_rows.append(
            [
                day_dir.name,
                str(total_numeric_sets),
                str(matched_sets),
                str(skipped_sets),
                index_json.name,
                state_json.name,
                announcements_csv.name,
            ]
        )
        all_cases.extend(day_cases)

    try:
        selected_cases = select_cases(
            all_cases,
            args.max_cases,
            args.per_day_limit,
            args.sampling,
            args.random_seed,
            args.set_range,
            args.set_list,
        )
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        return 1
    output_csv = Path(args.output_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    csv_rows = []
    for case in selected_cases:
        row = {key: case.get(key, "") for key in CASE_HEADERS}
        if row["expected_category_ordinal"] is None:
            row["expected_category_ordinal"] = ""
        csv_rows.append(row)
    written_csv = demo.write_csv(output_csv, CASE_HEADERS, csv_rows)
    written_jsonl = demo.write_jsonl(output_jsonl, selected_cases)

    table = Table(title="Semantic Benchmark Prepare Summary", expand=False)
    table.add_column("Day", style="green")
    table.add_column("Numeric sets", justify="right", style="magenta")
    table.add_column("Matched", justify="right", style="cyan")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Index", style="white")
    table.add_column("State", style="white")
    table.add_column("Candidates", style="white")
    for row in summary_rows:
        table.add_row(*row)
    console.print(table)
    console.print(f"[green]Wrote {written_csv} benchmark cases to {output_csv}[/green]")
    console.print(f"[green]Wrote {written_jsonl} benchmark cases to {output_jsonl}[/green]")
    return 0


def classify_case(
    backend_args: argparse.Namespace,
    model_name: str,
    case: Dict,
    codex_output_dir: Path,
    codex_schema: Path,
    codex_config_overrides: Sequence[str],
    response_format: Optional[Dict],
    extra_body: Optional[Dict],
) -> Tuple[str, str, Dict, str, List[Dict], float, Dict[str, object]]:
    start_time = time.perf_counter()
    response_status = "ok"
    response_error = ""
    backend_payload: Dict = {}
    raw_response_text = ""
    detections: List[Dict] = []
    prompt_text = build_benchmark_prompt(case, backend_args.prompt_profile)
    try:
        if backend_args.backend in {"openai-compatible", "local-openai"}:
            payload = demo.post_chat_completion(
                backend_args.api_base_url,
                backend_args.api_key,
                model_name,
                backend_args.system_prompt,
                prompt_text,
                backend_args.temperature,
                backend_args.timeout_seconds,
                backend_args.max_output_tokens,
                response_format,
                extra_body,
            )
            backend_payload = payload
            choice = payload["choices"][0]
            raw_response_text = str(choice["message"].get("content", ""))
        else:
            raw_response_text = demo.run_codex_exec(
                backend_args.system_prompt,
                prompt_text,
                codex_output_dir / demo.sanitize_window_id(case["case_id"]),
                codex_schema,
                model_name,
                backend_args.timeout_seconds,
                codex_config_overrides,
            )
        parsed = demo.extract_json_payload(raw_response_text, expected_window_id=case["window_id"])
        raw_detections = parsed.get("detections", [])
        if isinstance(raw_detections, list):
            detections = [demo.normalize_detection(item) for item in raw_detections if isinstance(item, dict)]
            detections = [item for item in detections if not demo.is_junk_detection(item)]
        else:
            response_status = "error"
            response_error = "response field 'detections' is not a list"
        elapsed = time.perf_counter() - start_time
        return response_status, response_error, backend_payload, raw_response_text, detections, elapsed, extract_backend_metrics(backend_payload)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        response_status = "error"
        response_error = str(exc)
        return response_status, response_error, backend_payload, raw_response_text, detections, elapsed, extract_backend_metrics(backend_payload)


def score_case(case: Dict, detections: Sequence[Dict]) -> Dict[str, bool]:
    expected_event_type = case["expected_event_type"]
    expected_number = case.get("expected_start_number")
    expected_ordinal = case.get("expected_category_ordinal")
    event_ok = any(detection.get("event_type") == expected_event_type for detection in detections)
    start_number_ok = False
    category_ordinal_ok = expected_ordinal in ("", None)
    if expected_number not in ("", None):
        start_number_ok = any(detection.get("start_number") == expected_number for detection in detections)
    if expected_ordinal not in ("", None):
        category_ordinal_ok = any(detection.get("category_ordinal") == expected_ordinal for detection in detections)
    strict_ok = event_ok and start_number_ok and category_ordinal_ok
    return {
        "event_ok": event_ok,
        "start_number_ok": start_number_ok,
        "category_ordinal_ok": category_ordinal_ok,
        "strict_ok": strict_ok,
    }


def stringify_values(values: Iterable[Optional[int]]) -> str:
    cleaned = [str(value) for value in values if value not in ("", None)]
    return "|".join(cleaned)


def run_benchmark(args: argparse.Namespace) -> int:
    args = demo.apply_backend_preset(args)
    input_jsonl = Path(args.input_jsonl).resolve()
    if not input_jsonl.exists():
        console.print(f"[red]Error: benchmark cases JSONL not found: {input_jsonl}[/red]")
        return 1
    cases = demo.read_jsonl(input_jsonl)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        console.print("[red]Error: no benchmark cases selected.[/red]")
        return 1
    output_csv = Path(args.output_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    codex_output_dir = Path(args.codex_output_dir).resolve()
    codex_schema = Path(args.codex_schema).resolve()
    if args.backend == "codex-exec" and not codex_schema.exists():
        console.print(f"[red]Error: codex exec output schema not found: {codex_schema}[/red]")
        return 1
    if args.backend in {"openai-compatible", "local-openai"} and not args.api_base_url:
        console.print("[red]Error: missing --api-base-url.[/red]")
        return 1

    response_format = build_benchmark_response_format(args.response_format_mode, args.prompt_profile)
    extra_body = build_ollama_extra_body(args)
    debug_dir = Path(args.debug_response_dir).resolve()
    codex_config_overrides = ['model_reasoning_effort="medium"']
    codex_config_overrides.extend(args.codex_config)
    manage_ollama_models = (
        args.manage_model_loading
        and args.backend in {"openai-compatible", "local-openai"}
        and is_ollama_api_base_url(args.api_base_url)
    )

    result_rows: List[Dict[str, object]] = []
    summary_rows: List[List[str]] = []

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
        models_task = progress.add_task("Models".ljust(25), total=len(args.models))
        for model_name in args.models:
            cases_task = progress.add_task(f"{model_name}".ljust(25), total=len(cases))
            model_results: List[Dict[str, object]] = []
            model_prepare_error = ""
            if manage_ollama_models:
                try:
                    prepare_ollama_model(
                        args.api_base_url,
                        model_name,
                        args.models,
                        args.ollama_keep_alive,
                        args.ollama_load_poll_seconds,
                        args.ollama_load_poll_attempts,
                        max(args.timeout_seconds, 120.0),
                    )
                except Exception as exc:
                    model_prepare_error = str(exc)
            for case in cases:
                if model_prepare_error:
                    response_status = "error"
                    response_error = model_prepare_error
                    backend_payload = {}
                    raw_response_text = ""
                    detections = []
                    elapsed = 0.0
                    backend_metrics = extract_backend_metrics(backend_payload)
                    scores = {
                        "event_ok": False,
                        "start_number_ok": False,
                        "category_ordinal_ok": case.get("expected_category_ordinal") in ("", None),
                        "strict_ok": False,
                    }
                else:
                    response_status, response_error, backend_payload, raw_response_text, detections, elapsed, backend_metrics = classify_case(
                        args,
                        model_name,
                        case,
                        codex_output_dir,
                        codex_schema,
                        codex_config_overrides,
                        response_format,
                        extra_body,
                    )
                    scores = score_case(case, detections)
                row = {
                    "model": model_name,
                    "case_id": case["case_id"],
                    "day": case["day"],
                    "expected_set_name": case["expected_set_name"],
                    "stream_id": case["stream_id"],
                    "filename": case["filename"],
                    "window_id": case["window_id"],
                    "expected_event_type": case["expected_event_type"],
                    "expected_start_number": case.get("expected_start_number", ""),
                    "expected_category_ordinal": case.get("expected_category_ordinal", ""),
                    "difficulty_score": case.get("difficulty_score", ""),
                    "predicted_event_types": "|".join(str(d.get("event_type", "")) for d in detections if d.get("event_type")),
                    "predicted_start_numbers": stringify_values(d.get("start_number") for d in detections),
                    "predicted_category_ordinals": stringify_values(d.get("category_ordinal") for d in detections),
                    "detection_count": len(detections),
                    "event_ok": scores["event_ok"],
                    "start_number_ok": scores["start_number_ok"],
                    "category_ordinal_ok": scores["category_ordinal_ok"],
                    "strict_ok": scores["strict_ok"],
                    "response_status": response_status,
                    "response_error": response_error,
                    "finish_reason": backend_metrics.get("finish_reason", ""),
                    "content_length": backend_metrics.get("content_length", 0),
                    "reasoning_length": backend_metrics.get("reasoning_length", 0),
                    "prompt_eval_count": backend_metrics.get("prompt_eval_count", ""),
                    "eval_count": backend_metrics.get("eval_count", ""),
                    "elapsed_seconds": round(elapsed, 3),
                }
                payload_row = dict(row)
                payload_row["detections"] = detections
                payload_row["backend_payload"] = backend_payload
                payload_row["raw_response_text"] = raw_response_text
                payload_row["prompt"] = case["prompt"]
                demo.write_debug_response_files(
                    debug_dir / demo.sanitize_window_id(model_name),
                    {
                        "window_id": case["case_id"],
                        "backend": args.backend,
                        "model": model_name,
                        "response_status": response_status,
                        "response_error": response_error,
                        "backend_payload": backend_payload,
                        "parsed_response": {"window_id": case["window_id"], "detections": detections},
                        "detections": detections,
                        "raw_response_text": raw_response_text,
                        "prompt": build_benchmark_prompt(case, args.prompt_profile),
                    },
                )
                result_rows.append(payload_row)
                model_results.append(row)
                progress.advance(cases_task)
            if manage_ollama_models and not model_prepare_error:
                try:
                    unload_ollama_model(
                        args.api_base_url,
                        model_name,
                        args.ollama_load_poll_seconds,
                        args.ollama_load_poll_attempts,
                        max(args.timeout_seconds, 60.0),
                    )
                except Exception:
                    pass
            progress.remove_task(cases_task)
            strict_correct = sum(1 for row in model_results if row["strict_ok"])
            event_correct = sum(1 for row in model_results if row["event_ok"])
            number_correct = sum(1 for row in model_results if row["start_number_ok"])
            errors = sum(1 for row in model_results if row["response_status"] != "ok")
            elapsed_values = [float(row["elapsed_seconds"]) for row in model_results]
            total_elapsed = sum(elapsed_values) if elapsed_values else 0.0
            summary_rows.append(
                [
                    model_name,
                    str(len(model_results)),
                    str(strict_correct),
                    f"{(strict_correct / len(model_results) * 100.0) if model_results else 0.0:.1f}%",
                    str(event_correct),
                    str(number_correct),
                    str(errors),
                    f"{total_elapsed:.2f}s",
                ]
            )
            progress.advance(models_task)

    flat_rows = []
    for row in result_rows:
        flat_rows.append({key: row.get(key, "") for key in RESULT_HEADERS})
    written_csv = demo.write_csv(output_csv, RESULT_HEADERS, flat_rows)
    written_jsonl = demo.write_jsonl(output_jsonl, result_rows)

    table = Table(title="Semantic Benchmark Summary", expand=False)
    table.add_column("Model", style="green")
    table.add_column("X/N", justify="right", style="cyan")
    table.add_column("T", justify="right", style="yellow")
    for model_name, case_count, strict_correct, _accuracy, _event_ok, _number_ok, _errors, _avg_time in summary_rows:
        table.add_row(model_name, f"{strict_correct}/{case_count}", _avg_time)
    console.print(table)
    if args.details:
        details_table = Table(title="Semantic Benchmark Details", expand=False)
        details_table.add_column("Model", style="green")
        details_table.add_column("Set", style="cyan")
        details_table.add_column("Day", style="magenta")
        details_table.add_column("Diff", justify="right", style="yellow")
        details_table.add_column("OK", justify="center", style="white")
        details_table.add_column("Event", justify="center", style="blue")
        details_table.add_column("Num", justify="center", style="red")
        details_table.add_column("Ord", justify="center", style="white")
        details_table.add_column("Status", style="white")
        details_table.add_column("Error", style="white")
        detail_rows = sorted(
            flat_rows,
            key=lambda row: (
                row.get("model", ""),
                row.get("strict_ok", "") == "True",
                -int(row.get("difficulty_score") or 0),
                row.get("day", ""),
                int(row.get("expected_start_number") or 0),
            ),
        )
        for row in detail_rows:
            details_table.add_row(
                str(row.get("model", "")),
                str(row.get("expected_set_name", "")),
                str(row.get("day", "")),
                str(row.get("difficulty_score", "")),
                "OK" if str(row.get("strict_ok", "")) == "True" else "NOK",
                "Y" if str(row.get("event_ok", "")) == "True" else "N",
                "Y" if str(row.get("start_number_ok", "")) == "True" else "N",
                "Y" if str(row.get("category_ordinal_ok", "")) == "True" else "N",
                str(row.get("response_status", "")),
                str(row.get("response_error", ""))[:80],
            )
        console.print(details_table)
    console.print(f"[green]Wrote {written_csv} benchmark result rows to {output_csv}[/green]")
    console.print(f"[green]Wrote {written_jsonl} benchmark payload rows to {output_jsonl}[/green]")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare_cases(args)
    if args.command == "run":
        return run_benchmark(args)
    console.print(f"[red]Error: unsupported command: {args.command}[/red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
