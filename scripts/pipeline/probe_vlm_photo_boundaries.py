#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

from lib.image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1
from lib.pipeline_io import atomic_write_json
from lib.photo_pre_model_annotations import (
    DEFAULT_OUTPUT_DIRNAME as DEFAULT_PHOTO_PRE_MODEL_DIRNAME,
    build_annotation_output_path,
    load_photo_pre_model_annotations_by_relative_path,
)


console = Console()

PHOTO_EMBEDDED_MANIFEST_FILENAME = "photo_embedded_manifest.csv"
PHOTO_MANIFEST_FILENAME = "photo_manifest.csv"
PHOTO_BOUNDARY_SCORES_FILENAME = "photo_boundary_scores.csv"
DEFAULT_OUTPUT_FILENAME = "vlm_boundary_test.csv"
RUN_METADATA_DIRNAME = "vlm_runs"
DEFAULT_IMAGE_VARIANT = "preview"
DEFAULT_WINDOW_SIZE = 10
DEFAULT_OVERLAP = 2
DEFAULT_BOUNDARY_GAP_SECONDS = 10
DEFAULT_MAX_BATCHES = 10
DEFAULT_MODEL_NAME = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_KEEP_ALIVE = "15m"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_THINK = "inherit"
DEFAULT_RESPONSE_SCHEMA_MODE = "off"
DEFAULT_JSON_VALIDATION_MODE = "strict"
DEFAULT_PHOTO_PRE_MODEL_DIR = DEFAULT_PHOTO_PRE_MODEL_DIRNAME
SEGMENT_TYPES = ("dance", "ceremony", "audience", "rehearsal", "other")
EMBEDDED_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "thumb_path", "preview_path"})
PHOTO_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "start_epoch_ms", "start_local", "photo_order_index"})
PHOTO_BOUNDARY_SCORES_REQUIRED_COLUMNS = frozenset(
    {
        "left_relative_path",
        "right_relative_path",
        "time_gap_seconds",
        "dino_cosine_distance",
        "boundary_score",
    }
)
OUTPUT_HEADERS = [
    "generated_at",
    "run_id",
    "config_hash",
    "image_variant",
    "model",
    "temperature",
    "batch_index",
    "start_row",
    "end_row",
    "window_size",
    "overlap",
    "relative_paths_json",
    "filenames_json",
    "image_paths_json",
    "delta_from_first_seconds_json",
    "delta_from_previous_seconds_json",
    "decision",
    "cut_after_local_index",
    "cut_after_global_row",
    "cut_left_relative_path",
    "cut_right_relative_path",
    "reason",
    "response_status",
    "raw_response",
]
MID_OUTPUT_HEADERS = [header for header in OUTPUT_HEADERS if header != "config_hash"]
LEGACY_OUTPUT_HEADERS = [header for header in OUTPUT_HEADERS if header not in {"run_id", "config_hash"}]
RESUME_CONFIG_KEYS = (
    "embedded_manifest_csv",
    "photo_manifest_csv",
    "image_variant",
    "window_size",
    "overlap",
    "boundary_gap_seconds",
    "model",
    "ollama_base_url",
    "ollama_num_ctx",
    "ollama_num_predict",
    "ollama_keep_alive",
    "timeout_seconds",
    "temperature",
    "ollama_think",
    "response_schema_mode",
    "json_validation_mode",
    "photo_pre_model_dir",
    "effective_extra_instructions",
)

SYSTEM_PROMPT = (
    "You analyze consecutive stage performance photos. "
    "Choose at most one boundary between consecutive frames. "
    "Return only valid JSON with keys decision, frame_notes, primary_evidence, and summary."
)


def build_system_prompt(response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE) -> str:
    if response_schema_mode == "on":
        return (
            "You analyze consecutive stage performance photos. "
            "Choose at most one boundary between consecutive frames. "
            "Return only valid JSON with keys boundary_after_frame, left_segment_type, right_segment_type, "
            "frame_notes, primary_evidence, and summary."
        )
    return SYSTEM_PROMPT


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def non_negative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a local VLM on overlapping stage-photo windows and write one CSV summary row per batch."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--run-id",
        help="Continue a specific existing VLM run by run_id instead of auto-selecting the latest compatible run.",
    )
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--embedded-manifest-csv",
        default=PHOTO_EMBEDDED_MANIFEST_FILENAME,
        help=f"Embedded manifest filename or absolute path. Default: {PHOTO_EMBEDDED_MANIFEST_FILENAME}",
    )
    parser.add_argument(
        "--photo-manifest-csv",
        default=PHOTO_MANIFEST_FILENAME,
        help=f"Photo manifest filename or absolute path. Default: {PHOTO_MANIFEST_FILENAME}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output CSV filename or absolute path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--image-variant",
        choices=("thumb", "preview"),
        default=DEFAULT_IMAGE_VARIANT,
        help=f"Embedded image variant to send to the VLM. Default: {DEFAULT_IMAGE_VARIANT}",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int_arg,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Number of consecutive images per VLM batch. Default: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int_arg,
        default=DEFAULT_OVERLAP,
        help=f"Number of images shared between adjacent windows. Default: {DEFAULT_OVERLAP}",
    )
    parser.add_argument(
        "--boundary-gap-seconds",
        type=non_negative_int_arg,
        default=DEFAULT_BOUNDARY_GAP_SECONDS,
        help=(
            "Only evaluate candidate boundaries where the time gap between consecutive photos "
            f"exceeds this many seconds. Default: {DEFAULT_BOUNDARY_GAP_SECONDS}"
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=positive_int_arg,
        default=DEFAULT_MAX_BATCHES,
        help=f"Maximum number of windows to evaluate. Default: {DEFAULT_MAX_BATCHES}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API base URL. Default: {DEFAULT_OLLAMA_BASE_URL}",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=positive_int_arg,
        help="Optional Ollama num_ctx override.",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=positive_int_arg,
        help="Optional Ollama num_predict override.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default=DEFAULT_OLLAMA_KEEP_ALIVE,
        help=f"Ollama keep_alive value. Default: {DEFAULT_OLLAMA_KEEP_ALIVE}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=non_negative_float_arg,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout for each Ollama request. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--temperature",
        type=non_negative_float_arg,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--ollama-think",
        choices=("inherit", "false", "low", "medium", "high"),
        default=DEFAULT_OLLAMA_THINK,
        help=f"Ollama reasoning effort override. Default: {DEFAULT_OLLAMA_THINK}",
    )
    parser.add_argument(
        "--response-schema-mode",
        choices=("off", "on"),
        default=DEFAULT_RESPONSE_SCHEMA_MODE,
        help=f"Enable or disable Ollama JSON Schema format enforcement. Default: {DEFAULT_RESPONSE_SCHEMA_MODE}",
    )
    parser.add_argument(
        "--json-validation-mode",
        choices=("strict", "relaxed"),
        default=DEFAULT_JSON_VALIDATION_MODE,
        help=f"Parser validation strictness for JSON responses. Default: {DEFAULT_JSON_VALIDATION_MODE}",
    )
    parser.add_argument(
        "--photo-pre-model-dir",
        default=DEFAULT_PHOTO_PRE_MODEL_DIR,
        help=(
            "Optional pre-model annotation directory relative to workspace or absolute. "
            f"Default: {DEFAULT_PHOTO_PRE_MODEL_DIR}"
        ),
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Optional extra instructions appended to the default VLM prompt.",
    )
    parser.add_argument(
        "--extra-instructions-file",
        help="Optional text file appended to the default VLM prompt.",
    )
    parser.add_argument(
        "--dump-debug-dir",
        help="Optional directory for per-batch prompt/request/response debug dumps.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a new VLM run instead of continuing the latest compatible run.",
    )
    return parser.parse_args(argv)


def validate_required_columns(name: str, fieldnames: Optional[Sequence[str]], required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(fieldnames or ()))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def read_embedded_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(EMBEDDED_MANIFEST_REQUIRED_COLUMNS))
        return [dict(row) for row in reader]


def read_photo_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(PHOTO_MANIFEST_REQUIRED_COLUMNS))
        rows = [dict(row) for row in reader]
    return {str(row["relative_path"]): row for row in rows}


def read_joined_rows(
    *,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    image_variant: str,
) -> List[Dict[str, str]]:
    embedded_rows = read_embedded_manifest(embedded_manifest_csv)
    photo_rows_by_relative_path = read_photo_manifest(photo_manifest_csv)
    image_column = "thumb_path" if image_variant == "thumb" else "preview_path"
    joined_rows: List[Dict[str, str]] = []
    for embedded_row in embedded_rows:
        relative_path = str(embedded_row.get("relative_path", "") or "").strip()
        if not relative_path:
            raise ValueError(f"{embedded_manifest_csv.name} row missing relative_path")
        photo_row = photo_rows_by_relative_path.get(relative_path)
        if photo_row is None:
            raise ValueError(f"{photo_manifest_csv.name} missing row for {relative_path}")
        image_value = str(embedded_row.get(image_column, "") or "").strip()
        if not image_value:
            raise ValueError(f"{embedded_manifest_csv.name} row missing {image_column} for {relative_path}")
        preview_value = str(embedded_row.get("preview_path", "") or "").strip()
        if not preview_value:
            raise ValueError(f"{embedded_manifest_csv.name} row missing preview_path for {relative_path}")
        image_path = resolve_path(workspace_dir, image_value).resolve()
        if not image_path.exists():
            raise ValueError(f"Image variant file does not exist: {image_path}")
        preview_path = resolve_path(workspace_dir, preview_value).resolve()
        if not preview_path.exists():
            raise ValueError(f"Preview image file does not exist: {preview_path}")
        joined_rows.append(
            {
                "relative_path": relative_path,
                "source_path": relative_path,
                "filename": Path(relative_path).name,
                "image_path": str(image_path),
                "image_relative_path": image_value,
                "preview_path": str(preview_path),
                "preview_relative_path": preview_value,
                "start_epoch_ms": str(photo_row.get("start_epoch_ms", "") or "").strip(),
                "start_local": str(photo_row.get("start_local", "") or "").strip(),
                "photo_order_index": str(photo_row.get("photo_order_index", "") or "").strip(),
                "stream_id": Path(relative_path).parts[0] if Path(relative_path).parts else "",
                "device": Path(relative_path).parts[0] if Path(relative_path).parts else "",
            }
        )
    if not joined_rows:
        raise ValueError(f"{embedded_manifest_csv.name} contains no rows")
    return joined_rows


def build_window_start_indexes(total_rows: int, window_size: int, overlap: int) -> List[int]:
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    stride = window_size - overlap
    starts = list(range(0, total_rows - window_size + 1, stride))
    final_start = total_rows - window_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_candidate_windows(
    rows: Sequence[Mapping[str, str]],
    window_size: int,
    overlap: int,
    boundary_gap_seconds: int,
) -> List[Dict[str, int]]:
    total_rows = len(rows)
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    final_start = total_rows - window_size
    candidates_by_start: Dict[int, Dict[str, int]] = {}
    for cut_index in range(total_rows - 1):
        left_epoch_ms = int(str(rows[cut_index]["start_epoch_ms"]))
        right_epoch_ms = int(str(rows[cut_index + 1]["start_epoch_ms"]))
        time_gap_seconds = rounded_seconds(right_epoch_ms - left_epoch_ms)
        if time_gap_seconds <= boundary_gap_seconds:
            continue
        start_index = cut_index - overlap + 1
        if start_index < 0:
            start_index = 0
        if start_index > final_start:
            start_index = final_start
        existing = candidates_by_start.get(start_index)
        candidate = {
            "start_index": start_index,
            "cut_index": cut_index,
            "time_gap_seconds": time_gap_seconds,
        }
        if existing is None or candidate["time_gap_seconds"] > existing["time_gap_seconds"]:
            candidates_by_start[start_index] = candidate
    return [candidates_by_start[start_index] for start_index in sorted(candidates_by_start)]


def build_candidate_window_start_indexes(
    rows: Sequence[Mapping[str, str]],
    window_size: int,
    overlap: int,
    boundary_gap_seconds: int,
) -> List[int]:
    return [
        int(candidate["start_index"])
        for candidate in build_candidate_windows(
            rows,
            window_size=window_size,
            overlap=overlap,
            boundary_gap_seconds=boundary_gap_seconds,
        )
    ]


def rounded_seconds(delta_ms: int) -> int:
    return int(round(delta_ms / 1000.0))


def build_temporal_lines(rows: Sequence[Mapping[str, str]]) -> List[str]:
    first_epoch_ms = int(str(rows[0]["start_epoch_ms"]))
    previous_epoch_ms = first_epoch_ms
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        current_epoch_ms = int(str(row["start_epoch_ms"]))
        t_from_first = rounded_seconds(current_epoch_ms - first_epoch_ms)
        delta_from_previous = rounded_seconds(current_epoch_ms - previous_epoch_ms) if index > 1 else 0
        lines.append(
            f"frame_{index:02d}: t_from_first={t_from_first}s, delta_from_previous={delta_from_previous}s"
        )
        previous_epoch_ms = current_epoch_ms
    return lines


def classify_time_gap_level(seconds: int) -> str:
    if seconds < 10:
        return "low"
    if seconds < 60:
        return "medium"
    return "high"


def classify_visual_distance_level(value: float) -> str:
    if value < 0.10:
        return "low"
    if value < 0.35:
        return "medium"
    return "high"


def classify_boundary_score_level(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    return "high"


def format_prompt_float(value: float) -> str:
    return f"{value:.3f}"


def read_boundary_scores_by_pair(path: Path) -> Dict[tuple[str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(PHOTO_BOUNDARY_SCORES_REQUIRED_COLUMNS))
        rows = [dict(row) for row in reader]
    return {
        (
            str(row.get("left_relative_path", "") or "").strip(),
            str(row.get("right_relative_path", "") or "").strip(),
        ): row
        for row in rows
    }


def build_gap_hint_lines(
    rows: Sequence[Mapping[str, str]],
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
) -> List[str]:
    lines: List[str] = []
    for index in range(1, len(rows)):
        left_row = rows[index - 1]
        right_row = rows[index]
        pair = (str(left_row["relative_path"]), str(right_row["relative_path"]))
        boundary_row = boundary_rows_by_pair.get(pair)
        if boundary_row is None:
            visual_distance_text = "unknown"
            visual_distance_level = "unknown"
            boundary_score_text = "unknown"
            boundary_score_level = "unknown"
        else:
            visual_distance = float(str(boundary_row.get("dino_cosine_distance", "") or "0"))
            boundary_score = float(str(boundary_row.get("boundary_score", "") or "0"))
            visual_distance_text = format_prompt_float(visual_distance)
            visual_distance_level = classify_visual_distance_level(visual_distance)
            boundary_score_text = format_prompt_float(boundary_score)
            boundary_score_level = classify_boundary_score_level(boundary_score)
        lines.append(
            f"gap_{index:02d}_{index + 1:02d}: "
            f"visual_distance={visual_distance_text} ({visual_distance_level}), "
            f"heuristic_boundary_score={boundary_score_text} ({boundary_score_level})"
        )
    return lines


def load_extra_instructions(inline_value: str, file_value: Optional[str]) -> str:
    parts: List[str] = []
    if inline_value.strip():
        parts.append(inline_value.strip())
    if file_value:
        file_path = Path(file_value).expanduser()
        parts.append(file_path.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in parts if part)


def build_valid_decisions(window_size: int) -> tuple[str, ...]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    return tuple(["no_cut"] + [f"cut_after_{index}" for index in range(1, window_size)])


def build_boundary_after_frame_values(window_size: int) -> List[Optional[str]]:
    return [None] + [f"frame_{index:02d}" for index in range(1, window_size)]


def build_response_schema(window_size: int) -> Dict[str, Any]:
    frame_properties = {f"frame_{index:02d}": {"type": "string"} for index in range(1, window_size + 1)}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "boundary_after_frame": {
                "type": ["string", "null"],
                "enum": build_boundary_after_frame_values(window_size),
            },
            "left_segment_type": {
                "type": "string",
                "enum": list(SEGMENT_TYPES),
            },
            "right_segment_type": {
                "type": "string",
                "enum": list(SEGMENT_TYPES),
            },
            "frame_notes": {
                "type": "object",
                "additionalProperties": False,
                "properties": frame_properties,
                "required": list(frame_properties.keys()),
            },
            "primary_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "summary": {"type": "string"},
        },
        "required": [
            "boundary_after_frame",
            "left_segment_type",
            "right_segment_type",
            "frame_notes",
            "primary_evidence",
            "summary",
        ],
    }


def build_user_prompt(
    window_size: int,
    gap_hint_lines: Sequence[str],
    extra_instructions: str = "",
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    pre_model_lines: Sequence[str] = (),
) -> str:
    frames_block = "\n".join([f"frame_{index:02d} = attached image {index}" for index in range(1, window_size + 1)])
    hints_block = "\n".join(gap_hint_lines) if gap_hint_lines else "No heuristic gap hints are available."
    if response_schema_mode == "on":
        decisions_block = "\n".join([f'- {value!r}' if value is not None else "- null" for value in build_boundary_after_frame_values(window_size)])
        response_header = "Allowed boundary_after_frame values:\n"
        response_rules = (
            '- Use null if there is no boundary in the window.\n'
            '- Use "frame_NN" only if the boundary is between frame_NN and the next frame.\n'
        )
        response_schema_hint = f'<one of: {", ".join(["null"] + [f"frame_{index:02d}" for index in range(1, window_size)])}>'
        response_object = (
            '{\n'
            f'  "boundary_after_frame": "{response_schema_hint}",\n'
            f'  "left_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            f'  "right_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            '  "frame_notes": {\n'
            + ",\n".join(
                [f'    "frame_{index:02d}": "<short note>"' for index in range(1, window_size + 1)]
            )
            + '\n  },\n'
            '  "primary_evidence": [\n'
            '    "<short evidence item>",\n'
            '    "<short evidence item>"\n'
            "  ],\n"
            '  "summary": "<one short sentence>"\n'
            '}'
        )
    else:
        decisions_block = "\n".join([f'- "{decision}"' for decision in build_valid_decisions(window_size)])
        response_header = "Allowed decisions:\n"
        response_rules = (
            f'- Choose "no_cut" if all {window_size} frames most likely belong to the same segment.\n'
            '- If there is no clear evidence for a boundary, choose "no_cut".\n'
            f'- Choose "cut_after_N" only if frames 1..N and frames N+1..{window_size} most likely belong to different segments.\n'
        )
        response_object = (
            '{\n'
            f'  "decision": "<one of: {", ".join(build_valid_decisions(window_size))}>",\n'
            '  "frame_notes": {\n'
            + ",\n".join(
                [f'    "frame_{index:02d}": "<short note>"' for index in range(1, window_size + 1)]
            )
            + '\n  },\n'
            '  "primary_evidence": [\n'
            '    "<short evidence item>",\n'
            '    "<short evidence item>"\n'
            "  ],\n"
            '  "summary": "<one short sentence>"\n'
            '}'
        )
    prompt = (
        f"You will receive {window_size} consecutive stage-event photos.\n\n"
        "Order:\n"
        f"{frames_block}\n\n"
        "Important:\n"
        "The frames are consecutive photos from one chronological sequence.\n"
        "They are not random examples.\n"
        "Reason about continuity from left to right:\n"
        f"frame_01 -> frame_02 -> ... -> frame_{window_size:02d}\n\n"
        "Task:\n"
        "Choose at most one boundary between consecutive frames.\n\n"
        "A boundary means that the photos before and after it most likely belong to different segments.\n\n"
        "Possible segment types:\n"
        "- dance performance / act\n"
        "- audience or backstage insert\n"
        "- floor rehearsal / floor test / stage test\n"
        "- ceremony / award / host / result reading\n\n"
        "Segment type labels:\n"
        "- dance = active staged dance performance\n"
        "- ceremony = awards, medals, result reading, host speaking, formal presentation\n"
        "- audience = viewers, crowd, or audience-facing non-performance shots\n"
        "- rehearsal = floor test, stage test, or non-performance rehearsal\n"
        "- other = does not clearly fit the categories above\n\n"
        "Create a boundary only if at least one positive boundary condition below is clearly true.\n"
        "If none of the positive boundary conditions is clearly true, return null.\n\n"
        "Positive boundary conditions:\n"
        "- the person on the left and the person on the right are not the same dancer\n"
        "- the dancers on the left and the dancers on the right do not belong to the same group\n"
        "- the costume on the left and the costume on the right do not belong to the same costume set\n"
        "- the segment type changes:\n"
        "  - dance -> ceremony\n"
        "  - dance -> audience\n"
        "  - dance -> rehearsal\n"
        "  - ceremony -> dance\n"
        "  - audience -> dance\n"
        "  - rehearsal -> dance\n\n"
        "Forbidden reasons for a boundary:\n"
        "- pose change\n"
        "- motion change\n"
        "- choreography phrase change\n"
        "- a new movement phrase inside the same act\n"
        "- framing change\n"
        "- lighting change alone\n"
        "- background change alone\n"
        "- a background change caused by camera angle, shooting direction, crop, or zoom on the same stage\n"
        "- a group shot followed by a solo shot of one dancer from the same group\n"
        "- a solo shot followed by a wider group shot from the same ongoing act\n"
        "- a change in visible performer count caused only by framing, crop, zoom, or partial visibility\n\n"
        "Continuity reminders:\n"
        "- Group performance -> single dancer from the same group can still be the same segment.\n"
        "- Single dancer -> wider group view can still be the same segment.\n"
        "- If costume identity and act identity still match, do not create a boundary only because fewer or more dancers are visible.\n"
        "- Ignore background differences if they can be explained by camera direction, framing, crop, zoom, or a different shooting angle on the same stage.\n\n"
        "If the change is only a new pose, new movement phrase, or another choreography moment within the same act, you must return null.\n\n"
        "Decision priority:\n"
        "1. images\n"
        "2. heuristic hints\n\n"
        "If images clearly contradict heuristic hints, trust the images first.\n\n"
        + (
            "Optional pre-model per-image annotations:\n"
            + "\n".join(pre_model_lines)
            + "\n\n"
            if pre_model_lines
            else ""
        )
        +
        "Hint interpretation:\n"
        "- low = weak signal\n"
        "- medium = ambiguous signal\n"
        "- high = strong signal\n"
        "- unknown = unavailable hint\n"
        "- larger visual_distance means more visual change between adjacent frames\n"
        "- heuristic_boundary_score near 0 suggests continuity, near 1 suggests a likely boundary\n\n"
        "Heuristic hints for consecutive gaps:\n"
        f"{hints_block}\n\n"
        f"{response_header}"
        f"{decisions_block}\n\n"
        "Rules:\n"
        f"{response_rules}"
        f"- If more than one real boundary appears inside the {window_size}-frame window, choose the single strongest and clearest boundary.\n"
        f"- Always assign left_segment_type and right_segment_type using only these values: {'|'.join(SEGMENT_TYPES)}.\n"
        "- Keep frame notes short and concrete.\n"
        '- If there is no boundary, primary_evidence should describe continuity evidence.\n'
        "- Output only valid JSON.\n"
        "- Do not output markdown.\n"
        "- Do not output any text before or after JSON.\n\n"
        "Return exactly one JSON object with this structure:\n"
        f"{response_object}"
    )
    if extra_instructions.strip():
        prompt += f"\n\nAdditional instructions:\n{extra_instructions.strip()}"
    return prompt


def encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


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


def build_ollama_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if not is_ollama_api_base_url(args.ollama_base_url):
        return extra_body
    if args.ollama_think != "inherit":
        reasoning_effort = "none" if args.ollama_think == "false" else args.ollama_think
        extra_body["reasoning_effort"] = reasoning_effort
        extra_body["reasoning"] = {"effort": reasoning_effort}
        extra_body["think"] = False if args.ollama_think == "false" else True
    options: Dict[str, Any] = {}
    if args.ollama_num_predict is not None:
        options["num_predict"] = args.ollama_num_predict
    if args.ollama_num_ctx is not None:
        options["num_ctx"] = args.ollama_num_ctx
    if options:
        extra_body["options"] = options
    return extra_body


def ollama_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        strip_v1_suffix(base_url) + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def build_ollama_payload(
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    keep_alive: str,
    temperature: float,
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    response_schema: Optional[Mapping[str, Any]] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {"temperature": temperature}
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
        "messages": [
            {"role": "system", "content": build_system_prompt(response_schema_mode)},
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image_base64(path) for path in image_paths],
            },
        ],
    }
    if extra_body:
        merged_options = dict(extra_body.get("options", {}))
        merged_options.update(payload["options"])
        payload.update({key: value for key, value in extra_body.items() if key != "options"})
        payload["options"] = merged_options
    if response_schema is not None:
        payload["format"] = dict(response_schema)
    return payload


def extract_response_text(response_payload: Mapping[str, Any]) -> str:
    message = response_payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    return ""


def extract_json_object_text(raw_response: str) -> str:
    text = raw_response.strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("JSON object not found in model response")
    return text[start : end + 1]


def parse_model_response(
    raw_response: str,
    *,
    window_size: int,
    json_validation_mode: str = DEFAULT_JSON_VALIDATION_MODE,
) -> Dict[str, str]:
    valid_decisions = build_valid_decisions(window_size)
    try:
        payload = json.loads(extract_json_object_text(raw_response))
    except Exception as error:
        return {
            "decision": "invalid_response",
            "reason": f"JSON parse error: {error}",
            "response_status": "invalid_response",
        }
    boundary_after_frame = payload.get("boundary_after_frame", "__missing__")
    if boundary_after_frame != "__missing__":
        if boundary_after_frame is None:
            decision = "no_cut"
        else:
            boundary_text = str(boundary_after_frame).strip()
            if boundary_text.lower() == "null":
                boundary_text = ""
            valid_frames = {f"frame_{index:02d}": f"cut_after_{index}" for index in range(1, window_size)}
            decision = "no_cut" if not boundary_text else valid_frames.get(boundary_text, "")
    else:
        decision = str(payload.get("decision", "") or "").strip()
    if decision not in valid_decisions:
        return {
            "decision": "invalid_response",
            "reason": f"Invalid decision value: {decision}",
            "response_status": "invalid_response",
        }
    left_segment_type = str(payload.get("left_segment_type", "") or "").strip()
    right_segment_type = str(payload.get("right_segment_type", "") or "").strip()
    if boundary_after_frame != "__missing__":
        if json_validation_mode == "strict" and (
            left_segment_type not in SEGMENT_TYPES or right_segment_type not in SEGMENT_TYPES
        ):
            return {
                "decision": "invalid_response",
                "reason": "Missing or invalid segment type value",
                "response_status": "invalid_response",
            }
    frame_notes = payload.get("frame_notes")
    if not isinstance(frame_notes, Mapping):
        return {
            "decision": "invalid_response",
            "reason": "Missing frame_notes object",
            "response_status": "invalid_response",
        }
    expected_keys = [f"frame_{index:02d}" for index in range(1, window_size + 1)]
    normalized_frame_notes: List[str] = []
    for key in expected_keys:
        value = str(frame_notes.get(key, "") or "").strip()
        if not value:
            return {
                "decision": "invalid_response",
                "reason": f"Missing frame_notes value for {key}",
                "response_status": "invalid_response",
            }
        normalized_frame_notes.append(f"{key}={value}")
    primary_evidence = payload.get("primary_evidence")
    if not isinstance(primary_evidence, list):
        return {
            "decision": "invalid_response",
            "reason": "Missing primary_evidence list",
            "response_status": "invalid_response",
        }
    normalized_evidence = [str(item).strip() for item in primary_evidence if str(item).strip()]
    if not normalized_evidence:
        return {
            "decision": "invalid_response",
            "reason": "Missing primary_evidence values",
            "response_status": "invalid_response",
        }
    summary = str(payload.get("summary", "") or "").strip()
    if not summary:
        return {
            "decision": "invalid_response",
            "reason": "Missing summary value",
            "response_status": "invalid_response",
        }
    segment_type_reason = ""
    if boundary_after_frame != "__missing__" and left_segment_type in SEGMENT_TYPES and right_segment_type in SEGMENT_TYPES:
        segment_type_reason = (
            f"Left segment type: {left_segment_type} | Right segment type: {right_segment_type} | "
        )
    reason = (
        f"{segment_type_reason}"
        f"Frame notes: {'; '.join(normalized_frame_notes)} | "
        f"Primary evidence: {'; '.join(normalized_evidence)} | "
        f"Summary: {summary}"
    )
    return {"decision": decision, "reason": reason, "response_status": "ok"}


def build_result_row(
    *,
    generated_at: str,
    run_id: str,
    config_hash: str,
    image_variant: str,
    batch_index: int,
    start_row: int,
    end_row: int,
    rows: Sequence[Mapping[str, str]],
    window_size: int,
    overlap: int,
    raw_response: str,
    parsed_response: Mapping[str, str],
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, str]:
    decision = str(parsed_response["decision"])
    cut_after_local_index = ""
    cut_after_global_row = ""
    cut_left_relative_path = ""
    cut_right_relative_path = ""
    if decision.startswith("cut_after_"):
        cut_after_local_index = decision.removeprefix("cut_after_")
        local_index = int(cut_after_local_index)
        cut_after_global_row = str(start_row + local_index - 1)
        cut_left_relative_path = str(rows[local_index - 1]["relative_path"])
        cut_right_relative_path = str(rows[local_index]["relative_path"])
    first_epoch_ms = int(str(rows[0]["start_epoch_ms"]))
    previous_epoch_ms = first_epoch_ms
    delta_from_first_seconds: List[int] = []
    delta_from_previous_seconds: List[int] = []
    for index, row in enumerate(rows):
        current_epoch_ms = int(str(row["start_epoch_ms"]))
        delta_from_first_seconds.append(rounded_seconds(current_epoch_ms - first_epoch_ms))
        delta_from_previous_seconds.append(0 if index == 0 else rounded_seconds(current_epoch_ms - previous_epoch_ms))
        previous_epoch_ms = current_epoch_ms
    return {
        "generated_at": generated_at,
        "run_id": run_id,
        "config_hash": config_hash,
        "image_variant": image_variant,
        "model": model,
        "temperature": str(temperature),
        "batch_index": str(batch_index),
        "start_row": str(start_row),
        "end_row": str(end_row),
        "window_size": str(window_size),
        "overlap": str(overlap),
        "relative_paths_json": json.dumps([str(row["relative_path"]) for row in rows], ensure_ascii=True),
        "filenames_json": json.dumps([str(row["filename"]) for row in rows], ensure_ascii=True),
        "image_paths_json": json.dumps([str(row["image_path"]) for row in rows], ensure_ascii=True),
        "delta_from_first_seconds_json": json.dumps(delta_from_first_seconds, ensure_ascii=True),
        "delta_from_previous_seconds_json": json.dumps(delta_from_previous_seconds, ensure_ascii=True),
        "decision": decision,
        "cut_after_local_index": cut_after_local_index,
        "cut_after_global_row": cut_after_global_row,
        "cut_left_relative_path": cut_left_relative_path,
        "cut_right_relative_path": cut_right_relative_path,
        "reason": str(parsed_response["reason"]),
        "response_status": str(parsed_response["response_status"]),
        "raw_response": raw_response,
    }


def append_result_rows(output_csv: Path, rows: Sequence[Mapping[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_csv.exists() else "w"
    write_header = not output_csv.exists()
    with output_csv.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({header: str(row.get(header, "")) for header in OUTPUT_HEADERS})


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_run_id(datetime_text: Optional[str] = None) -> str:
    if datetime_text:
        moment = datetime.fromisoformat(datetime_text)
    else:
        moment = datetime.now(timezone.utc).astimezone()
    return f"vlm-{moment.strftime('%Y%m%d%H%M%S')}"


def build_config_hash(args_payload: Mapping[str, Any]) -> str:
    filtered_payload = {
        key: (DEFAULT_RESPONSE_SCHEMA_MODE if key == "response_schema_mode" else args_payload.get(key))
        for key in RESUME_CONFIG_KEYS
    }
    encoded = json.dumps(filtered_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()


def build_user_prompt_template(
    window_size: int,
    extra_instructions: str = "",
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
) -> str:
    gap_hint_lines = [
        f"gap_{index:02d}_{index + 1:02d}: visual_distance=<value> (<level>), heuristic_boundary_score=<value> (<level>)"
        for index in range(1, window_size)
    ]
    return build_user_prompt(
        window_size=window_size,
        gap_hint_lines=gap_hint_lines,
        extra_instructions=extra_instructions,
        response_schema_mode=response_schema_mode,
    )


def build_photo_pre_model_lines(
    rows: Sequence[Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
) -> List[str]:
    if photo_pre_model_dir is None or not photo_pre_model_dir.exists():
        return []
    relative_paths = [str(row["relative_path"]) for row in rows]
    annotations_by_relative_path = load_photo_pre_model_annotations_by_relative_path(photo_pre_model_dir, relative_paths)
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        annotation = annotations_by_relative_path.get(str(row["relative_path"]))
        if not annotation:
            continue
        dominant_colors = annotation.get("dominant_colors")
        props = annotation.get("props")
        dominant_colors_text = (
            "|".join(str(value).strip() for value in dominant_colors if str(value).strip())
            if isinstance(dominant_colors, list)
            else ""
        )
        props_text = (
            "|".join(str(value).strip() for value in props if str(value).strip())
            if isinstance(props, list)
            else ""
        )
        parts = [
            f"people_count={annotation.get('people_count', '')}",
            f"performer_view={annotation.get('performer_view', '')}",
            f"upper_garment={annotation.get('upper_garment', '')}",
            f"lower_garment={annotation.get('lower_garment', '')}",
            f"sleeves={annotation.get('sleeves', '')}",
            f"leg_coverage={annotation.get('leg_coverage', '')}",
            f"dominant_colors={dominant_colors_text}",
            f"headwear={annotation.get('headwear', '')}",
            f"footwear={annotation.get('footwear', '')}",
            f"props={props_text}",
            f"dance_style_hint={annotation.get('dance_style_hint', '')}",
        ]
        lines.append(f"frame_{index:02d}: " + ", ".join(parts))
    return lines


def write_run_metadata(
    *,
    workspace_dir: Path,
    run_id: str,
    generated_at: str,
    config_hash: str,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    output_csv: Path,
    args_payload: Mapping[str, Any],
    system_prompt: str,
    user_prompt_template: str,
    response_schema: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    runs_dir = workspace_dir / RUN_METADATA_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = runs_dir / f"{run_id}.json"
    metadata = {
        "run_id": run_id,
        "generated_at": generated_at,
        "config_hash": config_hash,
        "embedded_manifest_csv": str(embedded_manifest_csv),
        "photo_manifest_csv": str(photo_manifest_csv),
        "output_csv": str(output_csv),
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "response_schema": dict(response_schema) if response_schema is not None else None,
        "args": dict(args_payload),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def read_latest_run_metadata(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    runs_dir = workspace_dir / RUN_METADATA_DIRNAME
    if not runs_dir.exists():
        return None
    metadata_paths = sorted(runs_dir.glob("vlm-*.json"))
    if not metadata_paths:
        return None
    return json.loads(metadata_paths[-1].read_text(encoding="utf-8"))


def read_run_metadata_by_id(workspace_dir: Path, run_id: str) -> Optional[Dict[str, Any]]:
    runs_dir = workspace_dir / RUN_METADATA_DIRNAME
    metadata_path = runs_dir / f"{run_id}.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def dump_debug_artifacts(
    *,
    debug_dir: Path,
    run_id: str,
    batch_index: int,
    prompt: str,
    request_payload: Mapping[str, Any],
    response_payload: Optional[Mapping[str, Any]],
    error_text: Optional[str],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"vlm_probe_{run_id}_batch_{batch_index:03d}"
    (debug_dir / f"{stem}_prompt.txt").write_text(prompt, encoding="utf-8")
    (debug_dir / f"{stem}_request.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if response_payload is not None:
        (debug_dir / f"{stem}_response.json").write_text(
            json.dumps(response_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if error_text is not None:
        (debug_dir / f"{stem}_error.txt").write_text(error_text, encoding="utf-8")


def build_gui_index_payload(
    *,
    day_name: str,
    workspace_dir: Path,
    image_variant: str,
    run_id: str,
    ordered_rows: Sequence[Mapping[str, str]],
    result_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    cut_reasons_by_pair: Dict[tuple[str, str], List[str]] = {}
    for result_row in result_rows:
        if str(result_row.get("response_status", "")) != "ok":
            continue
        left_relative_path = str(result_row.get("cut_left_relative_path", "") or "")
        right_relative_path = str(result_row.get("cut_right_relative_path", "") or "")
        if not left_relative_path or not right_relative_path:
            continue
        pair = (left_relative_path, right_relative_path)
        cut_reasons_by_pair.setdefault(pair, []).append(str(result_row.get("reason", "") or "").strip())

    segments: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    for row in ordered_rows:
        relative_path = str(row["relative_path"])
        is_new_segment_start = False
        split_pair: Optional[tuple[str, str]] = None
        if current_rows:
            previous_relative_path = str(current_rows[-1]["relative_path"])
            candidate_pair = (previous_relative_path, relative_path)
            if candidate_pair in cut_reasons_by_pair:
                is_new_segment_start = True
                split_pair = candidate_pair
        if is_new_segment_start:
            reasons = cut_reasons_by_pair.get(split_pair or ("", ""), [])
            segments.append(
                {
                    "rows": current_rows,
                    "cut_hits": len(reasons),
                    "cut_reasons": reasons,
                }
            )
            current_rows = [row]
            continue
        current_rows.append(row)
    if current_rows:
        segments.append(
            {
                "rows": current_rows,
                "cut_hits": 0,
                "cut_reasons": [],
            }
        )

    performances: List[Dict[str, Any]] = []
    total_photo_count = len(ordered_rows)
    for segment_index, segment in enumerate(segments, start=1):
        rows = list(segment["rows"])
        if not rows:
            continue
        photos: List[Dict[str, Any]] = []
        for row in rows:
            photos.append(
                {
                    "relative_path": str(row["relative_path"]),
                    "source_path": str(row["source_path"]),
                    "proxy_path": str(row["preview_relative_path"]),
                    "proxy_exists": True,
                    "filename": str(row["filename"]),
                    "photo_start_local": str(row.get("start_local", "") or ""),
                    "assignment_status": "",
                    "assignment_reason": "",
                    "seconds_to_nearest_boundary": "",
                    "stream_id": str(row.get("stream_id", "") or ""),
                    "device": str(row.get("device", "") or ""),
                }
            )
        display_name = f"VLM{segment_index:04d}"
        performances.append(
            {
                "set_id": f"vlm-set-{segment_index:04d}",
                "base_set_id": f"vlm-set-{segment_index:04d}",
                "display_name": display_name,
                "original_performance_number": display_name,
                "performance_number": display_name,
                "segment_index": str(segment_index - 1),
                "timeline_status": f"vlm_probe:{int(segment['cut_hits'])}_hits",
                "duplicate_status": "normal",
                "performance_start_local": str(rows[0].get("start_local", "") or ""),
                "performance_end_local": str(rows[-1].get("start_local", "") or ""),
                "segment_confidence": image_variant,
                "vlm_boundary_hits": str(int(segment["cut_hits"])),
                "vlm_boundary_reasons": list(segment["cut_reasons"]),
                "photos": photos,
            }
        )
    return {
        "day": day_name,
        "workspace_dir": str(workspace_dir),
        "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
        "vlm_run_id": run_id,
        "vlm_image_variant": image_variant,
        "performance_count": len(performances),
        "photo_count": total_photo_count,
        "performances": performances,
    }


def read_result_rows(output_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header_row = next(reader)
        except StopIteration:
            return rows
        header = [str(value) for value in header_row]
        if header == OUTPUT_HEADERS:
            for row in reader:
                if not row:
                    continue
                padded = list(row) + [""] * max(0, len(OUTPUT_HEADERS) - len(row))
                rows.append({key: padded[index] for index, key in enumerate(OUTPUT_HEADERS)})
            return rows
        if header == MID_OUTPUT_HEADERS:
            for row in reader:
                if not row:
                    continue
                if len(row) == len(MID_OUTPUT_HEADERS):
                    rows.append({key: row[index] for index, key in enumerate(MID_OUTPUT_HEADERS)} | {"config_hash": ""})
                    continue
                if len(row) == len(OUTPUT_HEADERS):
                    rows.append({key: row[index] for index, key in enumerate(OUTPUT_HEADERS)})
                    continue
                raise ValueError(
                    f"{output_csv.name} has unsupported row width {len(row)} for mid header; expected "
                    f"{len(MID_OUTPUT_HEADERS)} or {len(OUTPUT_HEADERS)} columns"
                )
            return rows
        if header == LEGACY_OUTPUT_HEADERS:
            for row in reader:
                if not row:
                    continue
                if len(row) == len(LEGACY_OUTPUT_HEADERS):
                    rows.append(
                        {key: row[index] for index, key in enumerate(LEGACY_OUTPUT_HEADERS)}
                        | {"run_id": "", "config_hash": ""}
                    )
                    continue
                if len(row) == len(MID_OUTPUT_HEADERS):
                    rows.append(
                        {key: row[index] for index, key in enumerate(MID_OUTPUT_HEADERS)}
                        | {"config_hash": ""}
                    )
                    continue
                if len(row) == len(OUTPUT_HEADERS):
                    rows.append({key: row[index] for index, key in enumerate(OUTPUT_HEADERS)})
                    continue
                raise ValueError(
                    f"{output_csv.name} has unsupported row width {len(row)} for legacy header; expected "
                    f"{len(LEGACY_OUTPUT_HEADERS)}, {len(MID_OUTPUT_HEADERS)} or {len(OUTPUT_HEADERS)} columns"
                )
            return rows
    raise ValueError(
        f"{output_csv.name} has unsupported header; expected current or legacy VLM probe columns"
    )


def resolve_run_state(
    *,
    workspace_dir: Path,
    output_csv: Path,
    config_hash: str,
    new_run: bool,
    run_id: Optional[str],
) -> Dict[str, Any]:
    if new_run:
        return {"run_id": None, "completed_batches": 0, "metadata": None}
    selected_metadata: Optional[Dict[str, Any]]
    if run_id:
        selected_metadata = read_run_metadata_by_id(workspace_dir, run_id)
        if selected_metadata is None:
            raise ValueError(f"Requested run_id does not exist: {run_id}")
    else:
        selected_metadata = read_latest_run_metadata(workspace_dir)
    if selected_metadata is None:
        return {"run_id": None, "completed_batches": 0, "metadata": None}
    selected_hash = str(selected_metadata.get("config_hash", "") or "")
    selected_args = selected_metadata.get("args")
    compatible_hashes = {selected_hash}
    if isinstance(selected_args, Mapping):
        compatible_hashes.add(build_config_hash(selected_args))
    if config_hash not in compatible_hashes:
        raise ValueError(
            f"Run configuration mismatch for resume: run {selected_metadata.get('run_id', '')} has "
            f"config_hash={selected_hash}, current config_hash={config_hash}. Use --new-run to start a fresh run."
        )
    completed_batches = 0
    if output_csv.exists():
        completed_batches = sum(
            1
            for row in read_result_rows(output_csv)
            if str(row.get("run_id", "") or "") == str(selected_metadata.get("run_id", "") or "")
        )
    return {
        "run_id": str(selected_metadata.get("run_id", "") or ""),
        "completed_batches": completed_batches,
        "metadata": selected_metadata,
    }


def build_resume_message(
    *,
    run_id: str,
    completed_batches: int,
    total_batches: int,
    requested_batches: int,
) -> str:
    next_batch = completed_batches + 1
    remaining_batches = max(0, total_batches - completed_batches)
    return (
        f"Continuing VLM run {run_id} from batch {next_batch}; "
        f"{remaining_batches} remaining, running up to {requested_batches} more batch(es)."
    )


def build_run_start_message(*, run_id: str, total_batches: int, boundary_gap_seconds: int) -> str:
    return (
        f"Starting VLM run {run_id}; {total_batches} candidate batch(es) "
        f"from gaps > {boundary_gap_seconds}s."
    )


def format_start_local_time(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        return text.split("T", 1)[1]
    return text


def build_candidate_batch_message(
    *,
    batch_index: int,
    total_batches: int,
    time_gap_seconds: int,
    start_row: int,
    end_row: int,
    left_start_local: str,
    right_start_local: str,
) -> str:
    return (
        f"Batch {batch_index}/{total_batches} | gap {time_gap_seconds}s | rows {start_row}..{end_row} | "
        f"{format_start_local_time(left_start_local)} -> {format_start_local_time(right_start_local)}"
    )


def build_batch_result_message(
    *,
    batch_index: int,
    decision: str,
    cut_after_global_row: str,
    cut_left_start_local: str,
    cut_right_start_local: str,
    cuts: int,
    no_cut: int,
    invalid: int,
) -> str:
    outcome = decision
    if cut_after_global_row:
        outcome = (
            f"{decision} -> global row {cut_after_global_row} "
            f"({format_start_local_time(cut_left_start_local)} -> {format_start_local_time(cut_right_start_local)})"
        )
    return (
        f"Result batch {batch_index}: {outcome} | "
        f"cuts={cuts} no_cut={no_cut} invalid={invalid}"
    )


def probe_vlm_photo_boundaries(
    *,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    output_csv: Path,
    image_variant: str,
    window_size: int,
    overlap: int,
    boundary_gap_seconds: int,
    max_batches: int,
    model: str,
    ollama_base_url: str,
    ollama_num_ctx: Optional[int],
    ollama_num_predict: Optional[int],
    ollama_keep_alive: str,
    timeout_seconds: float,
    temperature: float,
    ollama_think: str,
    extra_instructions: str,
    dump_debug_dir: Optional[Path],
    photo_pre_model_dir: Optional[Path] = None,
    json_validation_mode: str = DEFAULT_JSON_VALIDATION_MODE,
    args_payload: Mapping[str, Any],
    new_run: bool,
) -> int:
    config_hash = build_config_hash(args_payload)
    joined_rows = read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )
    boundary_rows_by_pair = read_boundary_scores_by_pair(workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME)
    all_candidates = build_candidate_windows(
        joined_rows,
        window_size=window_size,
        overlap=overlap,
        boundary_gap_seconds=boundary_gap_seconds,
    )
    run_state = resolve_run_state(
        workspace_dir=workspace_dir,
        output_csv=output_csv,
        config_hash=config_hash,
        new_run=new_run,
        run_id=str(args_payload.get("run_id", "") or "") or None,
    )
    completed_batches = int(run_state["completed_batches"])
    if completed_batches > len(all_candidates):
        raise ValueError(
            f"Run {run_state['run_id']} already has {completed_batches} batches, but only {len(all_candidates)} windows exist"
        )
    candidate_windows = all_candidates[completed_batches : completed_batches + max_batches]
    if not candidate_windows:
        console.print(
            f"No remaining VLM windows for run {run_state['run_id'] or '<new>'}; already processed {completed_batches} batch(es)."
        )
        return 0
    if run_state["run_id"]:
        console.print(
            build_resume_message(
                run_id=str(run_state["run_id"]),
                completed_batches=completed_batches,
                total_batches=len(all_candidates),
                requested_batches=max_batches,
            )
        )
    generated_at = current_timestamp()
    run_id = str(run_state["run_id"] or "")
    if not run_id:
        run_id = build_run_id(generated_at)
        response_schema = build_response_schema(window_size) if str(args_payload.get("response_schema_mode", "off")) == "on" else None
        write_run_metadata(
            workspace_dir=workspace_dir,
            run_id=run_id,
            generated_at=generated_at,
            config_hash=config_hash,
            embedded_manifest_csv=embedded_manifest_csv,
            photo_manifest_csv=photo_manifest_csv,
            output_csv=output_csv,
            args_payload=args_payload,
            system_prompt=build_system_prompt(str(args_payload.get("response_schema_mode", "off"))),
            user_prompt_template=build_user_prompt_template(
                window_size=window_size,
                extra_instructions=extra_instructions,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
            ),
            response_schema=response_schema,
        )
        console.print(
            build_run_start_message(
                run_id=run_id,
                total_batches=len(all_candidates),
                boundary_gap_seconds=boundary_gap_seconds,
            )
        )
    existing_result_rows = [
        row
        for row in (read_result_rows(output_csv) if output_csv.exists() else [])
        if str(row.get("run_id", "") or "") == run_id
    ]
    cut_count = sum(1 for row in existing_result_rows if str(row.get("decision", "") or "").startswith("cut_after_"))
    no_cut_count = sum(1 for row in existing_result_rows if str(row.get("decision", "") or "") == "no_cut")
    invalid_count = sum(1 for row in existing_result_rows if str(row.get("response_status", "") or "") != "ok")
    response_schema = build_response_schema(window_size) if str(args_payload.get("response_schema_mode", "off")) == "on" else None
    request_args = argparse.Namespace(
        ollama_base_url=ollama_base_url,
        ollama_think=ollama_think,
        ollama_num_predict=ollama_num_predict,
        ollama_num_ctx=ollama_num_ctx,
    )
    extra_body = build_ollama_extra_body(request_args)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    ) as progress:
        task = progress.add_task("Probe VLM candidate gaps".ljust(25), total=len(candidate_windows))
        for batch_offset, candidate in enumerate(candidate_windows, start=1):
            batch_index = completed_batches + batch_offset
            start_index = int(candidate["start_index"])
            cut_index = int(candidate["cut_index"])
            time_gap_seconds = int(candidate["time_gap_seconds"])
            end_index = start_index + window_size
            window_rows = joined_rows[start_index:end_index]
            progress.console.print(
                build_candidate_batch_message(
                    batch_index=batch_index,
                    total_batches=len(all_candidates),
                    time_gap_seconds=time_gap_seconds,
                    start_row=start_index + 1,
                    end_row=end_index,
                    left_start_local=str(joined_rows[cut_index]["start_local"]),
                    right_start_local=str(joined_rows[cut_index + 1]["start_local"]),
                )
            )
            gap_hint_lines = build_gap_hint_lines(window_rows, boundary_rows_by_pair)
            pre_model_lines = build_photo_pre_model_lines(window_rows, photo_pre_model_dir)
            prompt = build_user_prompt(
                window_size=window_size,
                gap_hint_lines=gap_hint_lines,
                extra_instructions=extra_instructions,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
                pre_model_lines=pre_model_lines,
            )
            payload = build_ollama_payload(
                model=model,
                prompt=prompt,
                image_paths=[Path(str(row["image_path"])) for row in window_rows],
                keep_alive=ollama_keep_alive,
                temperature=temperature,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
                response_schema=response_schema,
                extra_body=extra_body,
            )
            try:
                response_payload = ollama_post_json(ollama_base_url, "/api/chat", payload, timeout_seconds)
            except Exception as error:
                if dump_debug_dir is not None:
                    dump_debug_artifacts(
                        debug_dir=dump_debug_dir,
                        run_id=run_id,
                        batch_index=batch_index,
                        prompt=prompt,
                        request_payload=payload,
                        response_payload=None,
                        error_text=str(error),
                    )
                raise
            if dump_debug_dir is not None:
                dump_debug_artifacts(
                    debug_dir=dump_debug_dir,
                    run_id=run_id,
                    batch_index=batch_index,
                    prompt=prompt,
                    request_payload=payload,
                    response_payload=response_payload,
                    error_text=None,
                )
            raw_response = extract_response_text(response_payload)
            parsed_response = parse_model_response(
                raw_response,
                window_size=window_size,
                json_validation_mode=json_validation_mode,
            )
            result_row = build_result_row(
                generated_at=generated_at,
                run_id=run_id,
                config_hash=config_hash,
                image_variant=image_variant,
                batch_index=batch_index,
                start_row=start_index + 1,
                end_row=end_index,
                rows=window_rows,
                window_size=window_size,
                overlap=overlap,
                raw_response=raw_response,
                parsed_response=parsed_response,
                model=model,
                temperature=temperature,
            )
            append_result_rows(output_csv, [result_row])
            if str(result_row["decision"]).startswith("cut_after_"):
                cut_count += 1
            elif str(result_row["decision"]) == "no_cut":
                no_cut_count += 1
            else:
                invalid_count += 1
            progress.console.print(
                build_batch_result_message(
                    batch_index=batch_index,
                    decision=str(result_row["decision"]),
                    cut_after_global_row=str(result_row["cut_after_global_row"]),
                    cut_left_start_local=(
                        str(joined_rows[int(result_row["cut_after_global_row"]) - 1]["start_local"])
                        if str(result_row["cut_after_global_row"])
                        else ""
                    ),
                    cut_right_start_local=(
                        str(joined_rows[int(result_row["cut_after_global_row"])]["start_local"])
                        if str(result_row["cut_after_global_row"])
                        else ""
                    ),
                    cuts=cut_count,
                    no_cut=no_cut_count,
                    invalid=invalid_count,
                )
            )
            progress.advance(task)
    return len(candidate_windows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    embedded_manifest_csv = resolve_path(workspace_dir, args.embedded_manifest_csv)
    photo_manifest_csv = resolve_path(workspace_dir, args.photo_manifest_csv)
    output_csv = resolve_path(workspace_dir, args.output)
    if not embedded_manifest_csv.exists():
        raise SystemExit(f"Embedded manifest CSV does not exist: {embedded_manifest_csv}")
    if not photo_manifest_csv.exists():
        raise SystemExit(f"Photo manifest CSV does not exist: {photo_manifest_csv}")
    args_payload = {
        "day_dir": str(day_dir),
        "run_id": args.run_id,
        "workspace_dir": str(workspace_dir),
        "embedded_manifest_csv": str(embedded_manifest_csv),
        "photo_manifest_csv": str(photo_manifest_csv),
        "output_csv": str(output_csv),
        "image_variant": args.image_variant,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "boundary_gap_seconds": args.boundary_gap_seconds,
        "max_batches": args.max_batches,
        "model": args.model,
        "ollama_base_url": args.ollama_base_url,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_num_predict": args.ollama_num_predict,
        "ollama_keep_alive": args.ollama_keep_alive,
        "timeout_seconds": args.timeout_seconds,
        "temperature": args.temperature,
        "ollama_think": args.ollama_think,
        "response_schema_mode": args.response_schema_mode,
        "json_validation_mode": args.json_validation_mode,
        "photo_pre_model_dir": str(resolve_path(workspace_dir, args.photo_pre_model_dir)),
        "extra_instructions": args.extra_instructions,
        "extra_instructions_file": args.extra_instructions_file,
        "effective_extra_instructions": load_extra_instructions(args.extra_instructions, args.extra_instructions_file),
        "dump_debug_dir": args.dump_debug_dir,
    }
    row_count = probe_vlm_photo_boundaries(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        output_csv=output_csv,
        image_variant=args.image_variant,
        window_size=args.window_size,
        overlap=args.overlap,
        boundary_gap_seconds=args.boundary_gap_seconds,
        max_batches=args.max_batches,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        ollama_num_ctx=args.ollama_num_ctx,
        ollama_num_predict=args.ollama_num_predict,
        ollama_keep_alive=args.ollama_keep_alive,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        ollama_think=args.ollama_think,
        extra_instructions=str(args_payload["effective_extra_instructions"]),
        dump_debug_dir=Path(args.dump_debug_dir).expanduser().resolve() if args.dump_debug_dir else None,
        photo_pre_model_dir=(
            resolve_path(workspace_dir, args.photo_pre_model_dir).resolve()
            if str(args.photo_pre_model_dir or "").strip()
            else None
        ),
        json_validation_mode=args.json_validation_mode,
        args_payload=args_payload,
        new_run=bool(args.new_run),
    )
    console.print(f"Wrote {row_count} VLM probe rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
