#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta
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


console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent

DAY_PATTERN = re.compile(r"^\d{8}$")
NUMBER_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_PREPARE_CSV = "semantic_announcement_windows.csv"
DEFAULT_PREPARE_JSONL = "semantic_announcement_windows.jsonl"
DEFAULT_CLASSIFY_JSONL = "semantic_announcement_classification.jsonl"
DEFAULT_CLASSIFY_CSV = "semantic_announcement_classification.csv"
DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_API_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_CODEX_SCHEMA = str(SCRIPT_DIR / "codex_exec_announcement_output_schema.json")
DEFAULT_CODEX_BATCH_SCHEMA = str(SCRIPT_DIR / "codex_exec_announcement_batch_output_schema.json")
DEFAULT_DEBUG_RESPONSE_DIR = "/tmp/vocatio"
OFFICIAL_CATEGORY_GUIDANCE = (
    "Official DWC class and genre names may appear in Polish or English, for example: "
    "Ballet, Repertoire, National and Folklore, Lyrical, Showstopper, Jazz, Contemporary, "
    "Acro, Tap, Step, Song and Dance, Street Dance, Commercial, balet, balet repertuarowy, "
    "taniec narodowy i ludowy, taniec liryczny, taniec wspolczesny, wystep wokalno-taneczny, "
    "taniec uliczny, taniec komercyjny."
)

KEYWORD_TERMS = {
    "numer",
    "nr",
    "startowy",
    "startowa",
    "startowe",
    "startowych",
}
CATEGORY_TERMS = {
    "kategoria",
    "kategorii",
    "category",
    "children",
    "junior",
    "adult",
    "solo",
    "duo",
    "trio",
    "formation",
    "formacja",
}
JUNK_EVIDENCE_PATTERNS = (
    re.compile(r"\bdziekuj[ea]\b", re.IGNORECASE),
    re.compile(r"\bdzieki za ogladanie\b", re.IGNORECASE),
    re.compile(r"\bmuzyka\b", re.IGNORECASE),
    re.compile(r"\bnapisy by\b", re.IGNORECASE),
    re.compile(r"\bzdjecia i montaz\b", re.IGNORECASE),
    re.compile(r"\bbardzo prosimy o cisze\b", re.IGNORECASE),
)
EXPLICIT_NUMBER_PATTERN = re.compile(r"\b(?:numer(?:\s+startowy)?|nr)\s+(\d{1,3})\b", re.IGNORECASE)

PREPARE_HEADERS = [
    "day",
    "stream_id",
    "filename",
    "window_id",
    "window_start_local",
    "window_end_local",
    "window_duration_seconds",
    "segment_count",
    "trigger_types",
    "text_preview",
]

CLASSIFY_HEADERS = [
    "day",
    "stream_id",
    "filename",
    "window_id",
    "window_start_local",
    "window_end_local",
    "trigger_types",
    "detection_index",
    "event_type",
    "start_number",
    "category_ordinal",
    "title",
    "confidence",
    "evidence",
    "notes",
    "response_status",
    "response_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and classify transcript windows for semantic announcement experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Build local transcript windows around likely announcement content.",
    )
    prepare_parser.add_argument("day_dir", help="Path to a single day directory like /data/20260324")
    prepare_parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video_synced.csv. Default: DAY/_workspace",
    )
    prepare_parser.add_argument(
        "--merged-csv",
        help="Synced video CSV path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    prepare_parser.add_argument(
        "--transcripts-root",
        help="Transcript root directory. Default: DAY/_workspace/transcripts",
    )
    prepare_parser.add_argument(
        "--streams",
        nargs="*",
        help='Specific transcript stream IDs to parse, for example "v-pocket3" "v-gh7"',
    )
    prepare_parser.add_argument(
        "--all-streams",
        action="store_true",
        help="Parse transcripts from every available stream directory",
    )
    prepare_parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available transcript stream IDs and exit",
    )
    prepare_parser.add_argument(
        "--filenames",
        nargs="*",
        help="Optional exact video filenames to include, for example clip.mov",
    )
    prepare_parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit for the number of transcript JSON files to parse after filtering",
    )
    prepare_parser.add_argument(
        "--keyword-window-before",
        type=int,
        default=2,
        help="Segments to include before a keyword-triggered segment. Default: 2",
    )
    prepare_parser.add_argument(
        "--keyword-window-after",
        type=int,
        default=2,
        help="Segments to include after a keyword-triggered segment. Default: 2",
    )
    prepare_parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=5,
        help="Segment count for sliding windows. Default: 5",
    )
    prepare_parser.add_argument(
        "--sliding-step",
        type=int,
        default=2,
        help="Step between sliding windows. Default: 2",
    )
    prepare_parser.add_argument(
        "--max-windows",
        type=int,
        help="Optional limit for total emitted windows after de-duplication",
    )
    prepare_parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Build only keyword/category-triggered windows and skip generic sliding windows",
    )
    prepare_parser.set_defaults(trim_adjacent_trigger_windows=True)
    prepare_parser.add_argument(
        "--trim-adjacent-trigger-windows",
        action=argparse.BooleanOptionalAction,
        help="Trim keyword/category windows so they stop before adjacent trigger segments and focus on one local announcement. Default: enabled",
    )
    prepare_parser.add_argument(
        "--summary-output",
        default=DEFAULT_PREPARE_CSV,
        help=f"Summary CSV filename inside workspace or absolute path. Default: {DEFAULT_PREPARE_CSV}",
    )
    prepare_parser.add_argument(
        "--jsonl-output",
        default=DEFAULT_PREPARE_JSONL,
        help=f"Full JSONL filename inside workspace or absolute path. Default: {DEFAULT_PREPARE_JSONL}",
    )

    classify_parser = subparsers.add_parser(
        "classify",
        help="Send prepared transcript windows to an OpenAI-compatible endpoint.",
    )
    classify_parser.add_argument("day_dir", help="Path to a single day directory like /data/20260324")
    classify_parser.add_argument(
        "--workspace-dir",
        help="Workspace directory. Default: DAY/_workspace",
    )
    classify_parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_PREPARE_JSONL,
        help=f"Prepared windows JSONL inside workspace or absolute path. Default: {DEFAULT_PREPARE_JSONL}",
    )
    classify_parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_CLASSIFY_JSONL,
        help=f"Classification JSONL inside workspace or absolute path. Default: {DEFAULT_CLASSIFY_JSONL}",
    )
    classify_parser.add_argument(
        "--output-csv",
        default=DEFAULT_CLASSIFY_CSV,
        help=f"Flattened classification CSV inside workspace or absolute path. Default: {DEFAULT_CLASSIFY_CSV}",
    )
    classify_parser.add_argument(
        "--debug-response-dir",
        default=DEFAULT_DEBUG_RESPONSE_DIR,
        help=f"Directory for per-window raw response dumps. Default: {DEFAULT_DEBUG_RESPONSE_DIR}",
    )
    classify_parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        help='Model name for the selected backend. Default: OPENAI_MODEL or "gpt-5.4"',
    )
    classify_parser.add_argument(
        "--backend",
        choices=("openai-compatible", "local-openai", "codex-exec", "qwen-local"),
        default="codex-exec",
        help="Classification backend. Use local-openai for the local OpenAI-compatible preset. qwen-local is kept as a legacy alias. Default: codex-exec",
    )
    classify_parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        help=f"Base URL for the OpenAI-compatible endpoint. Default: OPENAI_BASE_URL or {DEFAULT_API_BASE_URL}",
    )
    classify_parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for the OpenAI-compatible endpoint. Default: OPENAI_API_KEY",
    )
    classify_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Chat completion temperature. Default: 0",
    )
    classify_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Request timeout in seconds. Default: 120",
    )
    classify_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for the OpenAI-compatible backend. Default: 1024",
    )
    classify_parser.add_argument(
        "--response-format-mode",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help="Structured output mode for the OpenAI-compatible backend. Default: json_schema",
    )
    classify_parser.add_argument(
        "--max-windows",
        type=int,
        help="Optional limit for the number of windows to classify",
    )
    classify_parser.add_argument(
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
        help="System prompt passed to the chat completion request.",
    )
    classify_parser.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help='Extra codex exec -c key=value options. Default includes model_reasoning_effort="medium".',
    )
    classify_parser.add_argument(
        "--codex-output-dir",
        help="Directory for temporary codex exec output files. Default: WORKSPACE/codex_exec_compare",
    )
    classify_parser.add_argument(
        "--codex-schema",
        default=DEFAULT_CODEX_SCHEMA,
        help=f"Output schema path for codex exec. Default: {DEFAULT_CODEX_SCHEMA}",
    )
    classify_parser.add_argument(
        "--codex-batch-schema",
        default=DEFAULT_CODEX_BATCH_SCHEMA,
        help=f"Batch output schema path for codex exec. Default: {DEFAULT_CODEX_BATCH_SCHEMA}",
    )
    classify_parser.add_argument(
        "--codex-batch-size",
        type=int,
        default=10,
        help="Number of windows per codex exec request. Default: 10",
    )
    classify_parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable structured response_format in the chat completion request.",
    )
    return parser.parse_args()


def resolve_output_path(workspace_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace_dir / path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_local_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="milliseconds" if value.microsecond else "seconds")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_text(value: str) -> str:
    lowered = strip_accents(value.lower())
    lowered = lowered.replace("№", " nr ")
    tokens = NUMBER_TOKEN_PATTERN.findall(lowered)
    return " ".join(tokens)


def detect_stream_ids(transcripts_root: Path) -> List[str]:
    if not transcripts_root.exists():
        return []
    return sorted(path.name for path in transcripts_root.iterdir() if path.is_dir())


def select_stream_ids(
    available_streams: Sequence[str],
    explicit_streams: Optional[Sequence[str]],
    all_streams: bool,
) -> List[str]:
    if explicit_streams:
        missing = [stream_id for stream_id in explicit_streams if stream_id not in available_streams]
        if missing:
            console.print(f"[red]Error: unknown transcript streams: {', '.join(missing)}[/red]")
            raise SystemExit(1)
        return list(explicit_streams)
    if all_streams:
        return list(available_streams)
    return list(available_streams)


def load_video_index(merged_rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in merged_rows:
        key = (row["stream_id"], Path(row["filename"]).stem)
        index[key] = row
    return index


def collect_transcript_jobs(
    transcripts_root: Path,
    stream_ids: Sequence[str],
    max_files: Optional[int],
) -> List[Tuple[str, Path]]:
    jobs: List[Tuple[str, Path]] = []
    for stream_id in stream_ids:
        stream_dir = transcripts_root / stream_id
        for path in sorted(stream_dir.glob("*.json")):
            jobs.append((stream_id, path))
    if max_files is not None:
        jobs = jobs[:max_files]
    return jobs


def text_trigger_types(text: str) -> Tuple[List[str], List[str]]:
    normalized = normalize_text(text)
    if not normalized:
        return [], []
    tokens = normalized.split()
    trigger_types: List[str] = []
    matched_terms: List[str] = []
    if any(token in KEYWORD_TERMS for token in tokens):
        trigger_types.append("keyword")
        matched_terms.extend(sorted({token for token in tokens if token in KEYWORD_TERMS}))
    if any(token in CATEGORY_TERMS for token in tokens):
        trigger_types.append("category")
        matched_terms.extend(sorted({token for token in tokens if token in CATEGORY_TERMS}))
    return trigger_types, matched_terms


def build_prompt(window: Dict) -> str:
    lines = []
    for segment in window["segments"]:
        lines.append(
            f"- {segment['segment_start_local']} -> {segment['segment_end_local']} | "
            f"segment {segment['segment_index']}: {segment['text']}"
        )
    transcript = "\n".join(lines)
    return (
        "Extract only real dance competition announcements from this local transcript window.\n"
        "Return only valid JSON with this schema:\n"
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
        '      "confidence": float\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Ignore song lyrics, background singing, subtitle-like noise, and watermark-like text.\n"
        "- Ignore generic phrases such as 'dziekuje', 'dzieki za ogladanie', 'muzyka', and organizational speech such as 'prosmy o cisze'.\n"
        "- Prefer explicit phrases like 'numer startowy 279' or 'numer 279' over category ordinals like 'szosty w tej kategorii'.\n"
        "- category_ordinal is the order within a category. start_number is the competitor number.\n"
        "- A word like 'piaty' or a phrase like 'numeru piatego w tej samej kategorii' means category_ordinal=5 and does not mean start_number=5.\n"
        "- If the transcript contains both an ordinal and an explicit competitor number, use the ordinal only for category_ordinal and use the explicit competitor number for start_number.\n"
        f"- {OFFICIAL_CATEGORY_GUIDANCE}\n"
        "- Only use category_ordinal for explicit ordering phrases such as pierwszy, drugi, trzeci, czwarty, piaty, szosty, first, second, third, fourth, fifth, or sixth.\n"
        "- Do not infer category_ordinal from class labels, genre names, age labels, roman numerals, or words like One or Two inside category titles.\n"
        "- Roman numerals or words like One, Two, I, II, III inside class names such as 'Junior Solo Contemporary I' or 'Children's Solo Acro One' are part of the class name and must not become category_ordinal.\n"
        "- If the local transcript contains a pattern like 'numer startowy N', 'start number N', or even a mistranscribed '* startowy N', treat N as the explicit competitor start_number.\n"
        "- Ceremony result reading is not a performance announcement.\n"
        "- If there is no explicit announcement, return an empty detections list.\n"
        "- If the window contains more than one useful announcement, return one detection per announcement in chronological order.\n"
        "- Only set start_number when an actual competitor start number is stated or clearly implied by the local transcript.\n"
        "- Never use timestamps, segment numbers, or line numbers as start_number.\n"
        "- Only set title if the title is explicitly present in the transcript.\n"
        "- Evidence must be a short direct quote from the transcript.\n"
        "- If there is no useful announcement-like content, return an empty detections list.\n\n"
        "Negative examples that should return empty detections:\n"
        '- "Dziekuje."\n'
        '- "Dzieki za ogladanie!"\n'
        '- "Bardzo prosimy juz o cisze."\n'
        '- "Popular, you\'re gonna be popular."\n\n'
        "Examples:\n"
        '- Transcript: "Przechodzimy do kolejnego numeru, numeru piatego w tej samej kategorii. Numer startowy 278."\n'
        '  Correct JSON detection: {"event_type":"performance_announcement","start_number":278,"category_ordinal":5,"title":null,"evidence":"Numer startowy 278.","notes":null,"confidence":0.95}\n'
        '- Transcript: "szosty w tej kategorii"\n'
        '  Correct JSON detection: {"event_type":"performance_announcement","start_number":null,"category_ordinal":6,"title":null,"evidence":"szosty w tej kategorii","notes":"No explicit competitor number.","confidence":0.6}\n\n'
        f"Window ID: {window['window_id']}\n"
        f"Stream: {window['stream_id']}\n"
        f"Filename: {window['filename']}\n"
        f"Window start: {window['window_start_local']}\n"
        f"Window end: {window['window_end_local']}\n"
        f"Trigger types: {', '.join(window['trigger_types'])}\n"
        "Transcript lines:\n"
        f"{transcript}\n"
    )


def build_batch_prompt(windows: Sequence[Dict]) -> str:
    blocks: List[str] = []
    for window in windows:
        lines = []
        for segment in window["segments"]:
            lines.append(
                f"- {segment['segment_start_local']} -> {segment['segment_end_local']} | "
                f"segment {segment['segment_index']}: {segment['text']}"
            )
        transcript = "\n".join(lines)
        blocks.append(
            f"Window ID: {window['window_id']}\n"
            f"Stream: {window['stream_id']}\n"
            f"Filename: {window['filename']}\n"
            f"Window start: {window['window_start_local']}\n"
            f"Window end: {window['window_end_local']}\n"
            f"Trigger types: {', '.join(window['trigger_types'])}\n"
            "Transcript lines:\n"
            f"{transcript}"
        )
    joined_blocks = "\n\n---\n\n".join(blocks)
    return (
        "Extract only real dance competition announcements from these local transcript windows.\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "results": [\n'
        "    {\n"
        '      "window_id": string,\n'
        '      "detections": [\n'
        "        {\n"
        '          "event_type": "performance_announcement" | "ceremony" | "other",\n'
        '          "start_number": integer | null,\n'
        '          "category_ordinal": integer | null,\n'
        '          "title": string | null,\n'
        '          "evidence": string | null,\n'
        '          "notes": string | null,\n'
        '          "confidence": float | null\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Return exactly one result object per provided window_id.\n"
        "- Ignore song lyrics, background singing, subtitle-like noise, and watermark-like text.\n"
        "- Ignore generic phrases such as 'dziekuje', 'dzieki za ogladanie', 'muzyka', and organizational speech such as 'prosmy o cisze'.\n"
        "- Prefer explicit phrases like 'numer startowy 279' or 'numer 279' over category ordinals like 'szosty w tej kategorii'.\n"
        "- category_ordinal is the order within a category. start_number is the competitor number.\n"
        "- A word like 'piaty' or a phrase like 'numeru piatego w tej samej kategorii' means category_ordinal=5 and does not mean start_number=5.\n"
        "- If a window contains both an ordinal and an explicit competitor number, use the ordinal only for category_ordinal and use the explicit competitor number for start_number.\n"
        f"- {OFFICIAL_CATEGORY_GUIDANCE}\n"
        "- Only use category_ordinal for explicit ordering phrases such as pierwszy, drugi, trzeci, czwarty, piaty, szosty, first, second, third, fourth, fifth, or sixth.\n"
        "- Do not infer category_ordinal from class labels, genre names, age labels, roman numerals, or words like One or Two inside category titles.\n"
        "- Roman numerals or words like One, Two, I, II, III inside class names such as 'Junior Solo Contemporary I' or 'Children's Solo Acro One' are part of the class name and must not become category_ordinal.\n"
        "- If the local transcript contains a pattern like 'numer startowy N', 'start number N', or even a mistranscribed '* startowy N', treat N as the explicit competitor start_number.\n"
        "- Ceremony result reading is not a performance announcement, but a ceremony detection is allowed if the window clearly contains ceremony-only speech.\n"
        "- If there is no explicit announcement in a window, return that window with an empty detections list.\n"
        "- If a window contains more than one useful announcement, return one detection per announcement in chronological order.\n"
        "- Only set start_number when an actual competitor start number is stated or clearly implied by the local transcript.\n"
        "- Never use timestamps, segment numbers, line numbers, or window ids as start_number.\n"
        "- Only set title if the title is explicitly present in the transcript.\n"
        "- Evidence must be a short direct quote from the transcript.\n\n"
        "Windows:\n"
        f"{joined_blocks}\n"
    )


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in row_list:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(row_list)


def sanitize_window_id(window_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", window_id).strip("_")


def write_debug_response_files(debug_dir: Path, result_payload: Dict) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    base_name = sanitize_window_id(str(result_payload.get("window_id", "window")))
    payload_path = debug_dir / f"{base_name}.response.json"
    content_path = debug_dir / f"{base_name}.content.txt"
    prompt_path = debug_dir / f"{base_name}.prompt.txt"
    payload_to_write = {
        "window_id": result_payload.get("window_id", ""),
        "backend": result_payload.get("backend", ""),
        "model": result_payload.get("model", ""),
        "response_status": result_payload.get("response_status", ""),
        "response_error": result_payload.get("response_error", ""),
        "backend_payload": result_payload.get("backend_payload"),
        "parsed_response": result_payload.get("parsed_response"),
        "detections": result_payload.get("detections", []),
    }
    payload_path.write_text(json.dumps(payload_to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    content_path.write_text(str(result_payload.get("raw_response_text", "")), encoding="utf-8")
    prompt_path.write_text(str(result_payload.get("prompt", "")), encoding="utf-8")


def build_detection_json_schema() -> Dict:
    return {
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
                        "title": {"type": ["string", "null"]},
                        "evidence": {"type": ["string", "null"]},
                        "notes": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "event_type",
                        "start_number",
                        "category_ordinal",
                        "title",
                        "evidence",
                        "notes",
                        "confidence",
                    ],
                },
            },
        },
        "required": ["window_id", "detections"],
    }


def build_openai_response_format(response_format_mode: str) -> Optional[Dict]:
    if response_format_mode == "none":
        return None
    if response_format_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "schema": build_detection_json_schema(),
    }


def apply_backend_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.backend not in {"local-openai", "qwen-local"}:
        return args
    args.backend = "local-openai"
    if getattr(args, "api_base_url", "") in ("", DEFAULT_API_BASE_URL):
        args.api_base_url = DEFAULT_LOCAL_API_BASE_URL
    if hasattr(args, "response_format_mode"):
        args.response_format_mode = "json_schema"
    if hasattr(args, "max_output_tokens") and args.max_output_tokens == 1024:
        args.max_output_tokens = 4096
    return args


def strip_markdown_fences(text: str) -> str:
    stripped = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"```", "", stripped)
    return stripped


def strip_think_blocks(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"</think>\s*", "", stripped, flags=re.IGNORECASE)
    return stripped


def extract_all_json_objects(text: str) -> List[Dict]:
    decoder = json.JSONDecoder()
    results: List[Dict] = []
    index = 0
    text_length = len(text)
    while index < text_length:
        if text[index] != "{":
            index += 1
            continue
        try:
            obj, offset = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(obj, dict):
            results.append(obj)
        index += offset
    return results


def is_valid_single_payload(payload: Dict, expected_window_id: Optional[str] = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if "window_id" not in payload or "detections" not in payload:
        return False
    if not isinstance(payload["window_id"], str):
        return False
    if expected_window_id and payload["window_id"] != expected_window_id:
        return False
    if not isinstance(payload["detections"], list):
        return False
    return True


def is_valid_batch_payload(payload: Dict) -> bool:
    if not isinstance(payload, dict):
        return False
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    for item in results:
        if not isinstance(item, dict):
            return False
        if "window_id" not in item or "detections" not in item:
            return False
        if not isinstance(item["window_id"], str):
            return False
        if not isinstance(item["detections"], list):
            return False
    return True


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def dedupe_windows(windows: Sequence[Dict]) -> List[Dict]:
    index: Dict[Tuple[str, str, int, int], Dict] = {}
    for window in windows:
        key = (
            window["stream_id"],
            window["filename"],
            window["segment_start_index"],
            window["segment_end_index"],
        )
        existing = index.get(key)
        if existing is None:
            index[key] = window
            continue
        trigger_types = sorted(set(existing["trigger_types"]) | set(window["trigger_types"]))
        matched_terms = sorted(set(existing["matched_terms"]) | set(window["matched_terms"]))
        center_segment_indices = sorted(
            set(existing["center_segment_indices"]) | set(window["center_segment_indices"])
        )
        existing["trigger_types"] = trigger_types
        existing["matched_terms"] = matched_terms
        existing["center_segment_indices"] = center_segment_indices
    windows_out = list(index.values())
    windows_out.sort(
        key=lambda item: (
            item["stream_id"],
            item["window_start_local"],
            item["filename"],
            item["segment_start_index"],
            item["segment_end_index"],
        )
    )
    return windows_out


def build_windows_for_clip(
    day_name: str,
    stream_id: str,
    filename: str,
    clip_path: str,
    transcript_path: str,
    segments: Sequence[Dict],
    keyword_before: int,
    keyword_after: int,
    sliding_window_size: int,
    sliding_step: int,
    keyword_only: bool,
    trim_adjacent_trigger_windows: bool,
) -> List[Dict]:
    raw_windows: List[Dict] = []
    segment_count = len(segments)
    if not segment_count:
        return raw_windows

    for position, segment in enumerate(segments):
        trigger_types = list(segment["trigger_types"])
        matched_terms = list(segment["matched_terms"])
        if not trigger_types:
            continue
        start_index = max(0, position - keyword_before)
        end_index = min(segment_count - 1, position + keyword_after)
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
        raw_windows.append(
            make_window(
                day_name,
                stream_id,
                filename,
                clip_path,
                transcript_path,
                segments,
                start_index,
                end_index,
                trigger_types,
                matched_terms,
                [segment["segment_index"]],
            )
        )

    if not keyword_only and sliding_window_size > 0 and sliding_step > 0:
        for start_index in range(0, segment_count, sliding_step):
            end_index = min(segment_count - 1, start_index + sliding_window_size - 1)
            raw_windows.append(
                make_window(
                    day_name,
                    stream_id,
                    filename,
                    clip_path,
                    transcript_path,
                    segments,
                    start_index,
                    end_index,
                    ["sliding"],
                    [],
                    [],
                )
            )
            if end_index == segment_count - 1:
                break
    return dedupe_windows(raw_windows)


def make_window(
    day_name: str,
    stream_id: str,
    filename: str,
    clip_path: str,
    transcript_path: str,
    segments: Sequence[Dict],
    start_index: int,
    end_index: int,
    trigger_types: Sequence[str],
    matched_terms: Sequence[str],
    center_segment_indices: Sequence[int],
) -> Dict:
    window_segments = list(segments[start_index : end_index + 1])
    start_local = window_segments[0]["segment_start_local"]
    end_local = window_segments[-1]["segment_end_local"]
    start_dt = parse_local_datetime(start_local)
    end_dt = parse_local_datetime(end_local)
    duration_seconds = 0.0
    if start_dt is not None and end_dt is not None:
        duration_seconds = (end_dt - start_dt).total_seconds()
    return {
        "day": day_name,
        "stream_id": stream_id,
        "filename": filename,
        "clip_path": clip_path,
        "transcript_path": transcript_path,
        "segment_start_index": window_segments[0]["segment_index"],
        "segment_end_index": window_segments[-1]["segment_index"],
        "center_segment_indices": list(center_segment_indices),
        "trigger_types": list(trigger_types),
        "matched_terms": list(matched_terms),
        "window_start_local": start_local,
        "window_end_local": end_local,
        "window_duration_seconds": round(duration_seconds, 3),
        "segment_count": len(window_segments),
        "segments": window_segments,
        "text": " ".join(segment["text"] for segment in window_segments).strip(),
    }


def build_prepare_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Bielik Window Prepare Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Segments", justify="right", style="cyan")
    table.add_column("Windows", justify="right", style="yellow")
    table.add_column("Keyword", justify="right", style="red")
    table.add_column("Category", justify="right", style="blue")
    table.add_column("Sliding", justify="right", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def build_classify_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Bielik Window Classification Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Windows", justify="right", style="magenta")
    table.add_column("Detections", justify="right", style="cyan")
    table.add_column("Announcements", justify="right", style="yellow")
    table.add_column("Ceremony", justify="right", style="red")
    table.add_column("Other", justify="right", style="blue")
    table.add_column("Errors", justify="right", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def prepare_windows(args: argparse.Namespace) -> int:
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260324, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video_synced.csv"
    transcripts_root = Path(args.transcripts_root).resolve() if args.transcripts_root else workspace_dir / "transcripts"
    summary_output = resolve_output_path(workspace_dir, args.summary_output)
    jsonl_output = resolve_output_path(workspace_dir, args.jsonl_output)

    if not merged_csv.exists():
        console.print(f"[red]Error: merged synced video CSV not found: {merged_csv}[/red]")
        return 1
    if not transcripts_root.exists() or not transcripts_root.is_dir():
        console.print(f"[red]Error: transcripts root not found: {transcripts_root}[/red]")
        return 1

    available_streams = detect_stream_ids(transcripts_root)
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

    selected_streams = select_stream_ids(available_streams, args.streams, args.all_streams)
    jobs = collect_transcript_jobs(transcripts_root, selected_streams, args.max_files)
    if not jobs:
        console.print("[red]Error: no transcript JSON files selected.[/red]")
        return 1
    filename_filters = set(args.filenames or [])

    merged_rows = read_csv_rows(merged_csv)
    video_index = load_video_index(merged_rows)

    clip_segments: Dict[Tuple[str, str], List[Dict]] = {}
    clip_meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    file_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    segment_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}

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
            for _, transcript_path in [job for job in jobs if job[0] == stream_id]:
                transcript_stem = transcript_path.stem
                video_row = video_index.get((stream_id, transcript_stem))
                if video_row is None:
                    progress.advance(files_task)
                    continue
                if filename_filters and video_row.get("filename", "") not in filename_filters:
                    progress.advance(files_task)
                    continue
                start_synced = parse_local_datetime(video_row.get("start_synced", ""))
                if start_synced is None:
                    progress.advance(files_task)
                    continue
                try:
                    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
                except Exception:
                    progress.advance(files_task)
                    continue

                key = (stream_id, video_row["filename"])
                file_counts[stream_id] += 1
                clip_meta[key] = {
                    "clip_path": video_row.get("path", ""),
                    "transcript_path": str(transcript_path),
                    "filename": video_row["filename"],
                }

                segments = payload.get("segments", [])
                if not isinstance(segments, list):
                    segments = []
                for segment_index, segment in enumerate(segments, 1):
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
                    segment_start_local = start_synced + timedelta(seconds=segment_start_seconds)
                    segment_end_local = start_synced + timedelta(seconds=segment_end_seconds)
                    trigger_types, matched_terms = text_trigger_types(text)
                    clip_segments.setdefault(key, []).append(
                        {
                            "segment_index": segment_index,
                            "segment_start_seconds": segment_start_seconds,
                            "segment_end_seconds": segment_end_seconds,
                            "segment_start_local": format_datetime(segment_start_local),
                            "segment_end_local": format_datetime(segment_end_local),
                            "text": text,
                            "trigger_types": trigger_types,
                            "matched_terms": matched_terms,
                        }
                    )
                    segment_counts[stream_id] += 1
                progress.advance(files_task)
            progress.advance(streams_task)

    windows: List[Dict] = []
    summary_counts: Dict[str, Dict[str, int]] = {
        stream_id: {"windows": 0, "keyword": 0, "category": 0, "sliding": 0}
        for stream_id in selected_streams
    }

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
        clips = sorted(clip_segments.items(), key=lambda item: (item[0][0], item[0][1]))
        clips_task = progress.add_task("Clips".ljust(25), total=len(clips))
        for (stream_id, filename), segments in clips:
            meta = clip_meta[(stream_id, filename)]
            clip_windows = build_windows_for_clip(
                day_dir.name,
                stream_id,
                filename,
                meta["clip_path"],
                meta["transcript_path"],
                sorted(segments, key=lambda item: item["segment_index"]),
                args.keyword_window_before,
                args.keyword_window_after,
                args.sliding_window_size,
                args.sliding_step,
                args.keyword_only,
                args.trim_adjacent_trigger_windows,
            )
            for clip_window_index, window in enumerate(clip_windows, 1):
                window["window_id"] = (
                    f"{stream_id}:{Path(filename).stem}:{window['segment_start_index']:04d}:"
                    f"{window['segment_end_index']:04d}"
                )
                window["clip_window_index"] = clip_window_index
                window["prompt"] = build_prompt(window)
                windows.append(window)
                summary_counts[stream_id]["windows"] += 1
                for trigger_type in window["trigger_types"]:
                    if trigger_type in summary_counts[stream_id]:
                        summary_counts[stream_id][trigger_type] += 1
            progress.advance(clips_task)

    windows = dedupe_windows(windows)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]

    prepare_rows: List[Dict[str, str]] = []
    for window in windows:
        text_preview = window["text"][:180]
        if len(window["text"]) > 180:
            text_preview += "..."
        prepare_rows.append(
            {
                "day": window["day"],
                "stream_id": window["stream_id"],
                "filename": window["filename"],
                "window_id": window["window_id"],
                "window_start_local": window["window_start_local"],
                "window_end_local": window["window_end_local"],
                "window_duration_seconds": f"{window['window_duration_seconds']:.3f}",
                "segment_count": str(window["segment_count"]),
                "trigger_types": ",".join(window["trigger_types"]),
                "text_preview": text_preview,
            }
        )

    summary_rows: List[List[str]] = []
    for stream_id in selected_streams:
        counts = summary_counts.get(stream_id, {})
        summary_rows.append(
            [
                stream_id,
                str(file_counts.get(stream_id, 0)),
                str(segment_counts.get(stream_id, 0)),
                str(sum(1 for window in windows if window["stream_id"] == stream_id)),
                str(sum(1 for window in windows if "keyword" in window["trigger_types"])),
                str(sum(1 for window in windows if "category" in window["trigger_types"])),
                str(sum(1 for window in windows if "sliding" in window["trigger_types"])),
            ]
        )

    written_csv = write_csv(summary_output, PREPARE_HEADERS, prepare_rows)
    written_jsonl = write_jsonl(jsonl_output, windows)
    console.print(build_prepare_summary_table(summary_rows))
    console.print(f"[green]Wrote {written_csv} Bielik window rows to {summary_output}[/green]")
    console.print(f"[green]Wrote {written_jsonl} Bielik window payloads to {jsonl_output}[/green]")
    return 0


def extract_json_payload(text: str, expected_window_id: Optional[str] = None, expect_batch: bool = False) -> Dict:
    stripped = strip_think_blocks(strip_markdown_fences(text.strip()))
    try:
        payload = json.loads(stripped)
        if expect_batch and is_valid_batch_payload(payload):
            return payload
        if not expect_batch and is_valid_single_payload(payload, expected_window_id):
            return payload
    except json.JSONDecodeError:
        pass

    candidates = extract_all_json_objects(stripped)
    if expect_batch:
        valid_candidates = [candidate for candidate in candidates if is_valid_batch_payload(candidate)]
    else:
        valid_candidates = [
            candidate for candidate in candidates if is_valid_single_payload(candidate, expected_window_id)
        ]
    if valid_candidates:
        return valid_candidates[-1]
    raise ValueError("response did not contain a valid target JSON payload")


def post_chat_completion(
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    timeout_seconds: float,
    max_output_tokens: int,
    response_format: Optional[Dict],
    extra_body: Optional[Dict] = None,
) -> Dict:
    base_url = api_base_url.rstrip("/")
    if not base_url.endswith("/chat/completions"):
        base_url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw_body = response.read().decode("utf-8")
        payload = json.loads(raw_body)
    return payload


def run_codex_exec(
    system_prompt: str,
    prompt: str,
    output_dir: Path,
    schema_path: Path,
    model: str,
    timeout_seconds: float,
    config_overrides: Sequence[str],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "last_message.json"
    command = [
        "codex",
        "exec",
        "-m",
        model,
        "-s",
        "read-only",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    for override in config_overrides:
        command.extend(["-c", override])
    completed = subprocess.run(
        command,
        input=f"{system_prompt}\n\n{prompt}",
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or f"codex exec exit code {completed.returncode}"
        raise RuntimeError(error_text)
    if not output_path.exists():
        raise RuntimeError("codex exec did not produce an output message file")
    return output_path.read_text(encoding="utf-8")


def parse_optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def parse_optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_explicit_start_number(text: str) -> Optional[int]:
    if not text:
        return None
    match = EXPLICIT_NUMBER_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1))


def normalize_detection(detection: Dict) -> Dict:
    event_type = str(detection.get("event_type", "")).strip().lower()
    if event_type not in {"performance_announcement", "ceremony", "other"}:
        event_type = "other"
    title = detection.get("title")
    evidence = detection.get("evidence")
    notes = detection.get("notes")
    normalized_evidence = None if evidence in (None, "") else str(evidence).strip()
    start_number = parse_optional_int(detection.get("start_number"))
    if normalized_evidence:
        explicit_start_number = extract_explicit_start_number(normalized_evidence)
        if explicit_start_number is not None and (start_number is None or start_number > 999):
            start_number = explicit_start_number
    if start_number is not None and start_number > 999:
        start_number = None
    category_ordinal = parse_optional_int(detection.get("category_ordinal"))
    if category_ordinal is not None and category_ordinal > 999:
        category_ordinal = None
    if event_type != "performance_announcement":
        start_number = None
    return {
        "event_type": event_type,
        "start_number": start_number,
        "category_ordinal": category_ordinal,
        "title": None if title in (None, "") else str(title).strip(),
        "evidence": normalized_evidence,
        "notes": None if notes in (None, "") else str(notes).strip(),
        "confidence": parse_optional_float(detection.get("confidence")),
    }


def is_junk_detection(detection: Dict) -> bool:
    evidence = detection.get("evidence") or ""
    title = detection.get("title") or ""
    combined = f"{evidence} {title}".strip()
    if not combined and detection.get("start_number") is None:
        return True
    if detection.get("event_type") != "performance_announcement":
        return False
    if detection.get("start_number") is None:
        for pattern in JUNK_EVIDENCE_PATTERNS:
            if pattern.search(combined):
                return True
    return False


def classify_windows(args: argparse.Namespace) -> int:
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260324, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    input_jsonl = resolve_output_path(workspace_dir, args.input_jsonl)
    output_jsonl = resolve_output_path(workspace_dir, args.output_jsonl)
    output_csv = resolve_output_path(workspace_dir, args.output_csv)
    codex_output_dir = (
        Path(args.codex_output_dir).resolve()
        if args.codex_output_dir
        else workspace_dir / "codex_exec_compare"
    )
    codex_schema = Path(args.codex_schema).resolve()
    codex_batch_schema = Path(args.codex_batch_schema).resolve()

    if not input_jsonl.exists():
        console.print(f"[red]Error: prepared windows JSONL not found: {input_jsonl}[/red]")
        return 1
    if not args.model:
        console.print("[red]Error: missing --model or OPENAI_MODEL.[/red]")
        return 1
    if args.backend in {"openai-compatible", "local-openai"} and not args.api_base_url:
        console.print("[red]Error: missing --api-base-url or OPENAI_BASE_URL.[/red]")
        return 1
    if args.backend == "codex-exec" and not codex_schema.exists():
        console.print(f"[red]Error: codex exec output schema not found: {codex_schema}[/red]")
        return 1
    if args.backend == "codex-exec" and args.codex_batch_size > 1 and not codex_batch_schema.exists():
        console.print(f"[red]Error: codex exec batch output schema not found: {codex_batch_schema}[/red]")
        return 1

    windows = read_jsonl(input_jsonl)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        console.print("[red]Error: no windows selected for classification.[/red]")
        return 1

    result_rows: List[Dict] = []
    csv_rows: List[Dict[str, str]] = []
    codex_config_overrides = ['model_reasoning_effort="medium"']
    codex_config_overrides.extend(args.codex_config)
    pending_windows: List[Dict] = list(windows)
    debug_response_dir = Path(args.debug_response_dir).expanduser().resolve()
    response_format = build_openai_response_format("none" if args.no_json_mode else args.response_format_mode)

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

            if args.backend == "codex-exec" and len(batch) > 1:
                response_status = "ok"
                response_error = ""
                raw_response_text = ""
                response_payload: Dict = {}
                backend_payload: Dict = {}
                batch_results: Dict[str, List[Dict]] = {window["window_id"]: [] for window in batch}
                try:
                    raw_response_text = run_codex_exec(
                        args.system_prompt,
                        build_batch_prompt(batch),
                        codex_output_dir / f"batch_{batch[0]['window_id'].replace(':', '_')}",
                        codex_batch_schema,
                        args.model,
                        args.timeout_seconds,
                        codex_config_overrides,
                    )
                    response_payload = extract_json_payload(raw_response_text, expect_batch=True)
                    raw_results = response_payload.get("results", [])
                    if not isinstance(raw_results, list):
                        response_status = "error"
                        response_error = "response field 'results' is not a list"
                    else:
                        for item in raw_results:
                            if not isinstance(item, dict):
                                continue
                            window_id = str(item.get("window_id", "")).strip()
                            if window_id not in batch_results:
                                continue
                            raw_detections = item.get("detections", [])
                            if not isinstance(raw_detections, list):
                                continue
                            detections = [
                                normalize_detection(detection)
                                for detection in raw_detections
                                if isinstance(detection, dict)
                            ]
                            batch_results[window_id] = [
                                detection for detection in detections if not is_junk_detection(detection)
                            ]
                except Exception as exc:
                    response_status = "error"
                    response_error = str(exc)

                for window in batch:
                    detections = batch_results.get(window["window_id"], [])
                    result_payload = {
                        "day": window["day"],
                        "stream_id": window["stream_id"],
                        "filename": window["filename"],
                        "window_id": window["window_id"],
                        "window_start_local": window["window_start_local"],
                        "window_end_local": window["window_end_local"],
                        "trigger_types": window["trigger_types"],
                        "matched_terms": window.get("matched_terms", []),
                        "prompt": window["prompt"],
                        "text": window["text"],
                        "segments": window["segments"],
                        "backend": args.backend,
                        "model": args.model,
                        "response_status": response_status,
                        "response_error": response_error,
                        "raw_response_text": raw_response_text,
                        "backend_payload": backend_payload,
                        "parsed_response": response_payload,
                        "detections": detections,
                    }
                    result_rows.append(result_payload)
                    write_debug_response_files(debug_response_dir, result_payload)

                    if detections:
                        for detection_index, detection in enumerate(detections, 1):
                            csv_rows.append(
                                {
                                    "day": window["day"],
                                    "stream_id": window["stream_id"],
                                    "filename": window["filename"],
                                    "window_id": window["window_id"],
                                    "window_start_local": window["window_start_local"],
                                    "window_end_local": window["window_end_local"],
                                    "trigger_types": ",".join(window["trigger_types"]),
                                    "detection_index": str(detection_index),
                                    "event_type": str(detection.get("event_type", "")),
                                    "start_number": "" if detection.get("start_number") is None else str(detection.get("start_number")),
                                    "category_ordinal": "" if detection.get("category_ordinal") is None else str(detection.get("category_ordinal")),
                                    "title": "" if detection.get("title") is None else str(detection.get("title")),
                                    "confidence": "" if detection.get("confidence") is None else str(detection.get("confidence")),
                                    "evidence": "" if detection.get("evidence") is None else str(detection.get("evidence")),
                                    "notes": "" if detection.get("notes") is None else str(detection.get("notes")),
                                    "response_status": response_status,
                                    "response_error": response_error,
                                }
                            )
                    else:
                        csv_rows.append(
                            {
                                "day": window["day"],
                                "stream_id": window["stream_id"],
                                "filename": window["filename"],
                                "window_id": window["window_id"],
                                "window_start_local": window["window_start_local"],
                                "window_end_local": window["window_end_local"],
                                "trigger_types": ",".join(window["trigger_types"]),
                                "detection_index": "",
                                "event_type": "",
                                "start_number": "",
                                "category_ordinal": "",
                                "title": "",
                                "confidence": "",
                                "evidence": "",
                                "notes": "",
                                "response_status": response_status,
                                "response_error": response_error,
                            }
                        )
                    progress.advance(windows_task)
                continue

            window = batch[0]
            response_status = "ok"
            response_error = ""
            response_payload: Dict = {}
            backend_payload: Dict = {}
            raw_response_text = ""
            detections: List[Dict] = []
            try:
                if args.backend in {"openai-compatible", "local-openai"}:
                    payload = post_chat_completion(
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
                    backend_payload = payload
                    choice = payload["choices"][0]
                    message = choice["message"]
                    raw_response_text = str(message.get("content", ""))
                else:
                    raw_response_text = run_codex_exec(
                        args.system_prompt,
                        window["prompt"],
                        codex_output_dir / window["window_id"].replace(":", "_"),
                        codex_schema,
                        args.model,
                        args.timeout_seconds,
                        codex_config_overrides,
                    )
                response_payload = extract_json_payload(raw_response_text, expected_window_id=window["window_id"])
                raw_detections = response_payload.get("detections", [])
                if isinstance(raw_detections, list):
                    detections = [
                        normalize_detection(item)
                        for item in raw_detections
                        if isinstance(item, dict)
                    ]
                    detections = [item for item in detections if not is_junk_detection(item)]
                else:
                    response_status = "error"
                    response_error = "response field 'detections' is not a list"
            except urllib.error.HTTPError as exc:
                response_status = "error"
                try:
                    response_error = exc.read().decode("utf-8").strip()
                except Exception:
                    response_error = str(exc)
            except Exception as exc:
                response_status = "error"
                response_error = str(exc)

            result_payload = {
                "day": window["day"],
                "stream_id": window["stream_id"],
                "filename": window["filename"],
                "window_id": window["window_id"],
                "window_start_local": window["window_start_local"],
                "window_end_local": window["window_end_local"],
                "trigger_types": window["trigger_types"],
                "matched_terms": window.get("matched_terms", []),
                "prompt": window["prompt"],
                "text": window["text"],
                "segments": window["segments"],
                "backend": args.backend,
                "model": args.model,
                "response_status": response_status,
                "response_error": response_error,
                "raw_response_text": raw_response_text,
                "backend_payload": backend_payload,
                "parsed_response": response_payload,
                "detections": detections,
            }
            result_rows.append(result_payload)
            write_debug_response_files(debug_response_dir, result_payload)

            if detections:
                for detection_index, detection in enumerate(detections, 1):
                    csv_rows.append(
                        {
                            "day": window["day"],
                            "stream_id": window["stream_id"],
                            "filename": window["filename"],
                            "window_id": window["window_id"],
                            "window_start_local": window["window_start_local"],
                            "window_end_local": window["window_end_local"],
                            "trigger_types": ",".join(window["trigger_types"]),
                            "detection_index": str(detection_index),
                            "event_type": str(detection.get("event_type", "")),
                            "start_number": "" if detection.get("start_number") is None else str(detection.get("start_number")),
                            "category_ordinal": "" if detection.get("category_ordinal") is None else str(detection.get("category_ordinal")),
                            "title": "" if detection.get("title") is None else str(detection.get("title")),
                            "confidence": "" if detection.get("confidence") is None else str(detection.get("confidence")),
                            "evidence": "" if detection.get("evidence") is None else str(detection.get("evidence")),
                            "notes": "" if detection.get("notes") is None else str(detection.get("notes")),
                            "response_status": response_status,
                            "response_error": response_error,
                        }
                    )
            else:
                csv_rows.append(
                    {
                        "day": window["day"],
                        "stream_id": window["stream_id"],
                        "filename": window["filename"],
                        "window_id": window["window_id"],
                        "window_start_local": window["window_start_local"],
                        "window_end_local": window["window_end_local"],
                        "trigger_types": ",".join(window["trigger_types"]),
                        "detection_index": "",
                        "event_type": "",
                        "start_number": "",
                        "category_ordinal": "",
                        "title": "",
                        "confidence": "",
                        "evidence": "",
                        "notes": "",
                        "response_status": response_status,
                        "response_error": response_error,
                    }
                )
            progress.advance(windows_task)

    summary_map: Dict[str, Dict[str, int]] = {}
    for row in result_rows:
        stream_summary = summary_map.setdefault(
            row["stream_id"],
            {"windows": 0, "detections": 0, "performance_announcement": 0, "ceremony": 0, "other": 0, "errors": 0},
        )
        stream_summary["windows"] += 1
        if row["response_status"] != "ok":
            stream_summary["errors"] += 1
        detections = row.get("detections", [])
        stream_summary["detections"] += len(detections)
        for detection in detections:
            event_type = str(detection.get("event_type", ""))
            if event_type in stream_summary:
                stream_summary[event_type] += 1

    summary_rows = []
    for stream_id in sorted(summary_map):
        counts = summary_map[stream_id]
        summary_rows.append(
            [
                stream_id,
                str(counts["windows"]),
                str(counts["detections"]),
                str(counts["performance_announcement"]),
                str(counts["ceremony"]),
                str(counts["other"]),
                str(counts["errors"]),
            ]
        )

    written_jsonl = write_jsonl(output_jsonl, result_rows)
    written_csv = write_csv(output_csv, CLASSIFY_HEADERS, csv_rows)
    console.print(build_classify_summary_table(summary_rows))
    console.print(f"[green]Wrote {written_jsonl} classification payload rows to {output_jsonl}[/green]")
    console.print(f"[green]Wrote {written_csv} flattened classification rows to {output_csv}[/green]")
    return 0


def main() -> int:
    args = parse_args()
    args = apply_backend_preset(args)
    if args.command == "prepare":
        return prepare_windows(args)
    if args.command == "classify":
        return classify_windows(args)
    console.print(f"[red]Error: unsupported command: {args.command}[/red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
