#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

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
from lib.ml_boundary_dataset import normalize_timestamp
from lib.ml_boundary_features import build_candidate_feature_row
from lib.ml_boundary_training_data import HEURISTIC_PAIR_JOIN_COLUMNS, HEURISTIC_VALUE_COLUMNS
from lib.media_manifest import read_media_manifest, select_photo_rows
from lib.pipeline_io import atomic_write_json
from lib.photo_pre_model_annotations import (
    DEFAULT_OUTPUT_DIRNAME as DEFAULT_PHOTO_PRE_MODEL_DIRNAME,
    load_photo_pre_model_annotations_by_relative_path,
)
from lib.vlm_transport import VlmRequest, build_provider_request_payload, get_vlm_capabilities, run_vlm_request
from lib.workspace_dir import load_vocatio_config, resolve_workspace_dir
from train_ml_boundary_verifier import (
    BOUNDARY_MODEL_DIRNAME,
    FEATURE_COLUMNS_FILENAME,
    SEGMENT_TYPE_MODEL_DIRNAME,
    TRAINING_METADATA_FILENAME,
    load_multimodal_predictor_class,
    load_tabular_predictor_class,
)

console = Console()

PHOTO_EMBEDDED_MANIFEST_FILENAME = "photo_embedded_manifest.csv"
PHOTO_MANIFEST_FILENAME = "media_manifest.csv"
PHOTO_BOUNDARY_SCORES_FILENAME = "photo_boundary_scores.csv"
DEFAULT_OUTPUT_FILENAME = "vlm_boundary_results.csv"
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
DEFAULT_PROVIDER = "ollama"
DEFAULT_ML_MODEL_RUN_ID = ""
SEGMENT_TYPES = ("dance", "ceremony", "audience", "rehearsal", "other")
ML_BOUNDARY_WINDOW_SIZE = 5
ML_BOUNDARY_WINDOW_RADIUS = 2
ML_SEGMENT_TYPE_TO_PROMPT_LABEL = {
    "performance": "dance",
    "ceremony": "ceremony",
    "warmup": "rehearsal",
}
ML_MODEL_ROOT_DIR_PARTS = ("ml_boundary_corpus", "ml_boundary_models")
EMBEDDED_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "thumb_path", "preview_path"})
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
    "provider",
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
    "effective_ml_model_run_id",
    "effective_extra_instructions",
)


class MlHintContext(NamedTuple):
    run_id: str
    mode: str
    model_dir: Path
    boundary_predictor: object
    segment_type_predictor: object
    boundary_feature_columns: list[str]
    segment_type_feature_columns: list[str]


class MlHintPrediction(NamedTuple):
    boundary_prediction: bool
    boundary_confidence: float
    boundary_positive_probability: float
    segment_type_prediction: str
    segment_type_confidence: float

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


def max_batches_arg(value: str) -> int:
    parsed = int(value)
    if parsed in (0, -1):
        return parsed
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1, or 0/-1 for all remaining batches")
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
        type=max_batches_arg,
        default=DEFAULT_MAX_BATCHES,
        help=(
            "Maximum number of windows to evaluate. "
            f"Use 0 or -1 for all remaining windows. Default: {DEFAULT_MAX_BATCHES}"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "llamacpp", "vllm"),
        default=DEFAULT_PROVIDER,
        help=f"VLM provider. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"VLM model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Provider API base URL. Default: {DEFAULT_OLLAMA_BASE_URL}",
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
        help=f"HTTP timeout for each VLM request. Default: {DEFAULT_TIMEOUT_SECONDS}",
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
        help=f"Provider reasoning override. Default: {DEFAULT_OLLAMA_THINK}",
    )
    parser.add_argument(
        "--response-schema-mode",
        choices=("off", "on"),
        default=DEFAULT_RESPONSE_SCHEMA_MODE,
        help=f"Enable or disable provider structured-output enforcement. Default: {DEFAULT_RESPONSE_SCHEMA_MODE}",
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
        "--ml-model-run-id",
        default=DEFAULT_ML_MODEL_RUN_ID,
        help=(
            "Optional ML boundary model run id used for prompt hints. "
            "Default: auto-select the latest available run in ml_boundary_corpus/ml_boundary_models."
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


def resolve_ml_model_run(
    workspace_dir: Path,
    requested_run_id: str,
) -> tuple[str, Optional[Path]]:
    model_root_dir = workspace_dir.joinpath(*ML_MODEL_ROOT_DIR_PARTS)
    normalized_run_id = requested_run_id.strip()
    if normalized_run_id:
        candidate_dir = model_root_dir / normalized_run_id
        if candidate_dir.is_dir():
            return normalized_run_id, candidate_dir.resolve()
        return normalized_run_id, None
    if not model_root_dir.is_dir():
        return "", None
    candidate_dirs = sorted(
        [
            path
            for path in model_root_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidate_dirs:
        return "", None
    selected_dir = candidate_dirs[-1]
    return selected_dir.name, selected_dir.resolve()


def build_default_output_filename(ml_model_run_id: str) -> str:
    normalized_run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", ml_model_run_id.strip()).strip("-")
    if normalized_run_id:
        return f"vlm_boundary_results.ml-{normalized_run_id}.csv"
    return "vlm_boundary_results.ml-hints.csv"


def read_embedded_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(EMBEDDED_MANIFEST_REQUIRED_COLUMNS))
        return [dict(row) for row in reader]


def read_photo_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    rows = select_photo_rows(read_media_manifest(path))
    return {str(row["relative_path"]): row for row in rows}


def apply_vocatio_defaults(args: argparse.Namespace, day_dir: Path) -> argparse.Namespace:
    config = load_vocatio_config(day_dir)

    def apply_string(attr: str, default_value: str, config_key: str) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, configured)

    def apply_optional_int(attr: str, config_key: str) -> None:
        if getattr(args, attr) is None:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, positive_int_arg(configured))

    def apply_int(attr: str, default_value: int, config_key: str, parser) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, parser(configured))

    def apply_float(attr: str, default_value: float, config_key: str) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, non_negative_float_arg(configured))

    apply_string("provider", DEFAULT_PROVIDER, "VLM_PROVIDER")
    apply_string("embedded_manifest_csv", PHOTO_EMBEDDED_MANIFEST_FILENAME, "VLM_EMBEDDED_MANIFEST_CSV")
    apply_string("photo_manifest_csv", PHOTO_MANIFEST_FILENAME, "VLM_PHOTO_MANIFEST_CSV")
    apply_string("image_variant", DEFAULT_IMAGE_VARIANT, "VLM_IMAGE_VARIANT")
    apply_int("window_size", DEFAULT_WINDOW_SIZE, "VLM_WINDOW_SIZE", positive_int_arg)
    apply_int("overlap", DEFAULT_OVERLAP, "VLM_OVERLAP", non_negative_int_arg)
    apply_int(
        "boundary_gap_seconds",
        DEFAULT_BOUNDARY_GAP_SECONDS,
        "VLM_BOUNDARY_GAP_SECONDS",
        non_negative_int_arg,
    )
    apply_int("max_batches", DEFAULT_MAX_BATCHES, "VLM_MAX_BATCHES", max_batches_arg)
    apply_string("model", DEFAULT_MODEL_NAME, "VLM_MODEL")
    apply_string("ollama_base_url", DEFAULT_OLLAMA_BASE_URL, "VLM_BASE_URL")
    apply_optional_int("ollama_num_ctx", "VLM_CONTEXT_TOKENS")
    apply_optional_int("ollama_num_predict", "VLM_MAX_OUTPUT_TOKENS")
    apply_string("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE, "VLM_KEEP_ALIVE")
    apply_float("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, "VLM_TIMEOUT_SECONDS")
    apply_float("temperature", DEFAULT_TEMPERATURE, "VLM_TEMPERATURE")
    apply_string("ollama_think", DEFAULT_OLLAMA_THINK, "VLM_REASONING_LEVEL")
    apply_string("response_schema_mode", DEFAULT_RESPONSE_SCHEMA_MODE, "VLM_RESPONSE_SCHEMA_MODE")
    apply_string("json_validation_mode", DEFAULT_JSON_VALIDATION_MODE, "VLM_JSON_VALIDATION_MODE")
    apply_string("photo_pre_model_dir", DEFAULT_PHOTO_PRE_MODEL_DIR, "VLM_PHOTO_PRE_MODEL_DIR")
    apply_string("ml_model_run_id", DEFAULT_ML_MODEL_RUN_ID, "VLM_ML_MODEL_RUN_ID")
    if args.dump_debug_dir is None:
        configured_dump_debug_dir = str(config.get("VLM_DUMP_DEBUG_DIR", "") or "").strip()
        if configured_dump_debug_dir:
            args.dump_debug_dir = configured_dump_debug_dir
    return args


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


def _load_required_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_plain_list(values: object) -> list[Any]:
    if hasattr(values, "tolist"):
        converted = values.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(values, list):
        return list(values)
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _to_model_frame(rows: list[dict[str, object]]) -> object:
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


def _normalize_boundary_prediction(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"unsupported boundary prediction value: {value!r}")


def _extract_positive_class_probability(probability_payload: object) -> float:
    if hasattr(probability_payload, "columns"):
        columns = list(probability_payload.columns)
        for candidate in (1, "1", True, "true", "True"):
            if candidate in columns:
                values = _to_plain_list(probability_payload[candidate])
                if len(values) != 1:
                    raise ValueError("boundary probability output must contain exactly one row")
                return float(values[0])
        if len(columns) == 2:
            values = _to_plain_list(probability_payload[columns[1]])
            if len(values) != 1:
                raise ValueError("boundary probability output must contain exactly one row")
            return float(values[0])
        raise ValueError("unable to resolve positive boundary probability column")
    values = _to_plain_list(probability_payload)
    if len(values) != 1:
        raise ValueError("boundary probability output must contain exactly one row")
    row = values[0]
    if isinstance(row, dict):
        for candidate in (1, "1", True, "true", "True"):
            if candidate in row:
                return float(row[candidate])
        raise ValueError("boundary probability dict is missing positive-class key")
    if isinstance(row, (list, tuple)):
        return float(row[-1])
    return float(row)


def _extract_label_probability(probability_payload: object, predicted_label: str) -> float:
    if hasattr(probability_payload, "columns"):
        columns = list(probability_payload.columns)
        if predicted_label not in columns:
            raise ValueError(f"segment_type probability output is missing label column {predicted_label!r}")
        values = _to_plain_list(probability_payload[predicted_label])
        if len(values) != 1:
            raise ValueError("segment_type probability output must contain exactly one row")
        return float(values[0])
    values = _to_plain_list(probability_payload)
    if len(values) != 1:
        raise ValueError("segment_type probability output must contain exactly one row")
    row = values[0]
    if isinstance(row, dict):
        if predicted_label not in row:
            raise ValueError(f"segment_type probability dict is missing label key {predicted_label!r}")
        return float(row[predicted_label])
    raise ValueError("unsupported segment_type probability output format")


def load_ml_hint_context(
    *,
    ml_model_run_id: str,
    ml_model_dir: Optional[Path],
) -> Optional[MlHintContext]:
    if ml_model_dir is None:
        return None
    training_metadata = _load_required_json(ml_model_dir / TRAINING_METADATA_FILENAME)
    feature_columns_payload = _load_required_json(ml_model_dir / FEATURE_COLUMNS_FILENAME)
    mode = str(training_metadata.get("mode", "tabular_only") or "tabular_only")
    predictor_class = load_multimodal_predictor_class() if mode == "tabular_plus_thumbnail" else load_tabular_predictor_class()
    if not hasattr(predictor_class, "load"):
        raise ValueError(f"{predictor_class.__name__} does not support load()")
    boundary_predictor = predictor_class.load(str(ml_model_dir / BOUNDARY_MODEL_DIRNAME))
    segment_type_predictor = predictor_class.load(str(ml_model_dir / SEGMENT_TYPE_MODEL_DIRNAME))
    return MlHintContext(
        run_id=ml_model_run_id,
        mode=mode,
        model_dir=ml_model_dir,
        boundary_predictor=boundary_predictor,
        segment_type_predictor=segment_type_predictor,
        boundary_feature_columns=[str(value) for value in feature_columns_payload.get("boundary_feature_columns", [])],
        segment_type_feature_columns=[str(value) for value in feature_columns_payload.get("segment_type_feature_columns", [])],
    )


def _build_ml_candidate_window_rows(
    joined_rows: Sequence[Mapping[str, str]],
    *,
    cut_index: int,
) -> Optional[list[Mapping[str, str]]]:
    window_start = cut_index - ML_BOUNDARY_WINDOW_RADIUS
    window_end = cut_index + ML_BOUNDARY_WINDOW_RADIUS + 1
    if window_start < 0 or window_end > len(joined_rows):
        return None
    rows = list(joined_rows[window_start:window_end])
    if len(rows) != ML_BOUNDARY_WINDOW_SIZE:
        return None
    return rows


def _build_ml_candidate_row(
    rows: Sequence[Mapping[str, str]],
    *,
    day_id: str,
) -> dict[str, object]:
    if len(rows) != ML_BOUNDARY_WINDOW_SIZE:
        raise ValueError(f"ML candidate rows must contain exactly {ML_BOUNDARY_WINDOW_SIZE} frames")
    normalized_day_id = day_id.strip()
    if normalized_day_id == "":
        raise ValueError("day_id must not be blank for ML prompt inference")
    candidate_row: dict[str, object] = {
        "day_id": normalized_day_id,
        "window_size": ML_BOUNDARY_WINDOW_SIZE,
        "center_left_photo_id": str(rows[2].get("photo_id", "") or rows[2].get("relative_path", "")).strip(),
        "center_right_photo_id": str(rows[3].get("photo_id", "") or rows[3].get("relative_path", "")).strip(),
        "boundary": False,
        "segment_type": "",
        "split_name": "",
    }
    for frame_index, row in enumerate(rows, start=1):
        suffix = f"{frame_index:02d}"
        photo_id = str(row.get("photo_id", "") or row.get("relative_path", "")).strip()
        relative_path = str(row.get("relative_path", "") or "").strip()
        if photo_id == "" or relative_path == "":
            raise ValueError("joined photo rows must include photo_id and relative_path for ML hints")
        timestamp_seconds = normalize_timestamp(row.get("start_epoch_ms")) / 1000.0
        candidate_row[f"frame_{suffix}_photo_id"] = photo_id
        candidate_row[f"frame_{suffix}_relpath"] = relative_path
        candidate_row[f"frame_{suffix}_timestamp"] = timestamp_seconds
        candidate_row[f"frame_{suffix}_thumb_path"] = str(row.get("thumb_path", "") or "").strip()
    return candidate_row


def _load_ml_candidate_descriptors(
    candidate_row: Mapping[str, object],
    *,
    photo_pre_model_dir: Optional[Path],
) -> dict[str, dict[str, object]]:
    if photo_pre_model_dir is None or not photo_pre_model_dir.exists():
        return {}
    relative_paths = [
        str(candidate_row.get(f"frame_{frame_index:02d}_relpath", "") or "").strip()
        for frame_index in range(1, ML_BOUNDARY_WINDOW_SIZE + 1)
    ]
    annotations_by_relative_path = load_photo_pre_model_annotations_by_relative_path(
        photo_pre_model_dir,
        relative_paths,
    )
    descriptors: dict[str, dict[str, object]] = {}
    for frame_index in range(1, ML_BOUNDARY_WINDOW_SIZE + 1):
        suffix = f"{frame_index:02d}"
        photo_id = str(candidate_row.get(f"frame_{suffix}_photo_id", "") or "").strip()
        relative_path = str(candidate_row.get(f"frame_{suffix}_relpath", "") or "").strip()
        if photo_id == "" or relative_path == "":
            continue
        annotation = annotations_by_relative_path.get(relative_path)
        if annotation is not None:
            descriptors[photo_id] = annotation
    return descriptors


def _build_ml_candidate_heuristic_features(
    candidate_row: Mapping[str, object],
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    heuristic_features: dict[str, dict[str, str]] = {}
    for pair_name, left_column, right_column in HEURISTIC_PAIR_JOIN_COLUMNS:
        pair_row = boundary_rows_by_pair.get(
            (
                str(candidate_row.get(left_column, "") or "").strip(),
                str(candidate_row.get(right_column, "") or "").strip(),
            )
        )
        if pair_row is None:
            continue
        heuristic_features[pair_name] = {
            column_name: str(pair_row.get(column_name, "") or "").strip()
            for column_name in HEURISTIC_VALUE_COLUMNS
        }
    return heuristic_features


def _build_predictor_feature_row(
    *,
    feature_columns: Sequence[str],
    candidate_row: Mapping[str, object],
    derived_features: Mapping[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {}
    for column_name in feature_columns:
        if column_name in derived_features:
            row[column_name] = derived_features[column_name]
            continue
        if column_name in candidate_row:
            row[column_name] = candidate_row[column_name]
            continue
        raise ValueError(f"missing ML predictor feature column {column_name!r} for prompt inference")
    return row


def predict_ml_hint_for_candidate(
    *,
    ml_hint_context: MlHintContext,
    candidate_row: Mapping[str, object],
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
) -> MlHintPrediction:
    descriptors = _load_ml_candidate_descriptors(candidate_row, photo_pre_model_dir=photo_pre_model_dir)
    heuristic_features = _build_ml_candidate_heuristic_features(candidate_row, boundary_rows_by_pair)
    derived_features = build_candidate_feature_row(
        candidate_row,
        descriptors=descriptors,
        embeddings=None,
        heuristic_features=heuristic_features,
    )
    segment_type_row = _build_predictor_feature_row(
        feature_columns=ml_hint_context.segment_type_feature_columns,
        candidate_row=candidate_row,
        derived_features=derived_features,
    )
    boundary_row = _build_predictor_feature_row(
        feature_columns=ml_hint_context.boundary_feature_columns,
        candidate_row=candidate_row,
        derived_features=derived_features,
    )
    segment_type_frame = _to_model_frame([segment_type_row])
    boundary_frame = _to_model_frame([boundary_row])

    segment_type_prediction = str(_to_plain_list(ml_hint_context.segment_type_predictor.predict(segment_type_frame))[0]).strip()
    segment_type_confidence = _extract_label_probability(
        ml_hint_context.segment_type_predictor.predict_proba(segment_type_frame),
        segment_type_prediction,
    )

    boundary_prediction = _normalize_boundary_prediction(
        _to_plain_list(ml_hint_context.boundary_predictor.predict(boundary_frame))[0]
    )
    positive_boundary_probability = _extract_positive_class_probability(
        ml_hint_context.boundary_predictor.predict_proba(boundary_frame)
    )
    boundary_confidence = positive_boundary_probability if boundary_prediction else 1.0 - positive_boundary_probability
    return MlHintPrediction(
        boundary_prediction=boundary_prediction,
        boundary_confidence=boundary_confidence,
        boundary_positive_probability=positive_boundary_probability,
        segment_type_prediction=segment_type_prediction,
        segment_type_confidence=segment_type_confidence,
    )


def build_ml_hint_lines(
    prediction: Optional[MlHintPrediction],
) -> List[str]:
    if prediction is None:
        return ["ML hints are unavailable for this window."]
    boundary_text = (
        "likely cut at the main candidate gap"
        if prediction.boundary_prediction
        else "likely no cut at the main candidate gap"
    )
    right_segment_label = ML_SEGMENT_TYPE_TO_PROMPT_LABEL.get(
        prediction.segment_type_prediction,
        prediction.segment_type_prediction,
    )
    return [
        f"ML hint for the main candidate gap in this window: {boundary_text} (confidence {prediction.boundary_confidence:.2f}).",
        f"ML hint for the likely segment on the right side of the candidate gap: {right_segment_label} (confidence {prediction.segment_type_confidence:.2f}).",
    ]


def build_ml_hint_lines_for_candidate(
    *,
    day_id: str,
    joined_rows: Sequence[Mapping[str, str]],
    cut_index: int,
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
    ml_hint_context: Optional[MlHintContext],
) -> List[str]:
    if ml_hint_context is None:
        return build_ml_hint_lines(None)
    candidate_rows = _build_ml_candidate_window_rows(joined_rows, cut_index=cut_index)
    if candidate_rows is None:
        return build_ml_hint_lines(None)
    prediction = predict_ml_hint_for_candidate(
        ml_hint_context=ml_hint_context,
        candidate_row=_build_ml_candidate_row(candidate_rows, day_id=day_id),
        boundary_rows_by_pair=boundary_rows_by_pair,
        photo_pre_model_dir=photo_pre_model_dir,
    )
    return build_ml_hint_lines(prediction)


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
    ml_hint_lines: Sequence[str],
    extra_instructions: str = "",
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
) -> str:
    frames_block = "\n".join([f"frame_{index:02d} = attached image {index}" for index in range(1, window_size + 1)])
    hints_block = "\n".join(ml_hint_lines) if ml_hint_lines else "ML hints are unavailable for this window."
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
        "2. ML hints\n\n"
        "If images clearly contradict ML hints, trust the images first.\n\n"
        "ML hint notes:\n"
        "- These hints come from a separate tabular model.\n"
        "- That model aggregates heuristic gap signals and per-photo visual annotations around the main candidate gap.\n"
        "- Treat the hints as advisory, not authoritative.\n"
        "- The images remain the final evidence.\n\n"
        "ML hints for the main candidate gap:\n"
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


def build_vlm_response_format(
    *,
    provider: str,
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    response_schema: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if response_schema_mode != "on" or response_schema is None:
        return None
    capabilities = get_vlm_capabilities(provider)
    if not capabilities.supports_json_schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "photo_boundary_probe",
            "schema": dict(response_schema),
        },
    }


def build_vlm_request(
    *,
    provider: str,
    base_url: str,
    response_schema_mode: str,
    prompt: str,
    image_paths: Sequence[Path],
    model: str,
    timeout_seconds: float,
    keep_alive: str,
    temperature: float,
    context_tokens: Optional[int],
    max_output_tokens: Optional[int],
    reasoning_level: str,
    response_schema: Optional[Mapping[str, Any]],
) -> VlmRequest:
    request_keep_alive: Optional[str] = None
    request_reasoning_level: Optional[str] = None
    if provider == "ollama":
        request_keep_alive = keep_alive
        request_reasoning_level = None if reasoning_level == "inherit" else reasoning_level
    return VlmRequest(
        provider=provider,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(response_schema_mode)},
            {"role": "user", "content": prompt},
        ],
        image_paths=list(image_paths),
        timeout_seconds=timeout_seconds,
        response_format=build_vlm_response_format(
            provider=provider,
            response_schema_mode=response_schema_mode,
            response_schema=response_schema,
        ),
        temperature=temperature,
        context_tokens=context_tokens,
        max_output_tokens=max_output_tokens,
        reasoning_level=request_reasoning_level,
        keep_alive=request_keep_alive,
    )


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
    ml_hint_lines = [
        "ML hint for the main candidate gap in this window: likely cut at the main candidate gap (confidence 0.82).",
        "ML hint for the likely segment on the right side of the candidate gap: dance (confidence 0.74).",
    ]
    return build_user_prompt(
        window_size=window_size,
        ml_hint_lines=ml_hint_lines,
        extra_instructions=extra_instructions,
        response_schema_mode=response_schema_mode,
    )


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
    requested_text = "all remaining" if requested_batches <= 0 else f"up to {requested_batches} more"
    return (
        f"Continuing VLM run {run_id} from batch {next_batch}; "
        f"{remaining_batches} remaining, running {requested_text} batch(es)."
    )


def build_run_start_message(*, run_id: str, total_batches: int, boundary_gap_seconds: int) -> str:
    return (
        f"Starting VLM run {run_id}; {total_batches} candidate batch(es) "
        f"from gaps > {boundary_gap_seconds}s."
    )


def build_candidate_batch_message(
    *,
    batch_index: int,
    total_batches: int,
    time_gap_seconds: int,
    start_row: int,
    end_row: int,
) -> str:
    return f"Batch {batch_index}/{total_batches} | gap {time_gap_seconds}s | rows {start_row}..{end_row}"


def build_batch_result_message(
    *,
    batch_index: int,
    total_batches: int,
    time_gap_seconds: int,
    start_row: int,
    end_row: int,
    decision: str,
    cuts: int,
    no_cut: int,
    invalid: int,
) -> str:
    return (
        f"Batch {batch_index}/{total_batches} | gap {time_gap_seconds}s | rows {start_row}..{end_row} | {decision} | "
        f"cuts={cuts} no_cut={no_cut} invalid={invalid}"
    )


def probe_vlm_photo_boundaries(
    *,
    day_id: str,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    output_csv: Path,
    provider: str,
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
    ml_model_run_id: str = "",
    ml_model_dir: Optional[Path] = None,
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
    try:
        ml_hint_context = load_ml_hint_context(
            ml_model_run_id=ml_model_run_id,
            ml_model_dir=ml_model_dir,
        )
    except Exception:
        ml_hint_context = None
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
    effective_max_batches = len(all_candidates) if max_batches <= 0 else max_batches
    candidate_windows = all_candidates[completed_batches : completed_batches + effective_max_batches]
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
            ml_hint_lines = build_ml_hint_lines_for_candidate(
                day_id=day_id,
                joined_rows=joined_rows,
                cut_index=cut_index,
                boundary_rows_by_pair=boundary_rows_by_pair,
                photo_pre_model_dir=photo_pre_model_dir,
                ml_hint_context=ml_hint_context,
            )
            prompt = build_user_prompt(
                window_size=window_size,
                ml_hint_lines=ml_hint_lines,
                extra_instructions=extra_instructions,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
            )
            request = build_vlm_request(
                provider=provider,
                base_url=ollama_base_url,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
                prompt=prompt,
                image_paths=[Path(str(row["image_path"])) for row in window_rows],
                model=model,
                timeout_seconds=timeout_seconds,
                keep_alive=ollama_keep_alive,
                temperature=temperature,
                context_tokens=ollama_num_ctx,
                max_output_tokens=ollama_num_predict,
                reasoning_level=ollama_think,
                response_schema=response_schema,
            )
            request_payload = build_provider_request_payload(request) if dump_debug_dir is not None else None
            try:
                response = run_vlm_request(request)
            except Exception as error:
                if dump_debug_dir is not None:
                    dump_debug_artifacts(
                        debug_dir=dump_debug_dir,
                        run_id=run_id,
                        batch_index=batch_index,
                        prompt=prompt,
                        request_payload=request_payload or {},
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
                    request_payload=request_payload or {},
                    response_payload=response.raw_response,
                    error_text=None,
                )
            raw_response = response.text
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
                    total_batches=len(all_candidates),
                    time_gap_seconds=time_gap_seconds,
                    start_row=start_index + 1,
                    end_row=end_index,
                    decision=str(result_row["decision"]),
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
    args = apply_vocatio_defaults(args, day_dir)
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    effective_ml_model_run_id, resolved_ml_model_dir = resolve_ml_model_run(
        workspace_dir,
        args.ml_model_run_id,
    )
    if args.output == DEFAULT_OUTPUT_FILENAME:
        args.output = build_default_output_filename(effective_ml_model_run_id)
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
        "provider": args.provider,
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
        "requested_ml_model_run_id": args.ml_model_run_id,
        "effective_ml_model_run_id": effective_ml_model_run_id,
        "ml_model_dir": str(resolved_ml_model_dir) if resolved_ml_model_dir is not None else "",
        "extra_instructions": args.extra_instructions,
        "extra_instructions_file": args.extra_instructions_file,
        "effective_extra_instructions": load_extra_instructions(args.extra_instructions, args.extra_instructions_file),
        "dump_debug_dir": args.dump_debug_dir,
    }
    row_count = probe_vlm_photo_boundaries(
        day_id=day_dir.name,
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        output_csv=output_csv,
        provider=args.provider,
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
        ml_model_run_id=effective_ml_model_run_id,
        ml_model_dir=resolved_ml_model_dir,
        json_validation_mode=args.json_validation_mode,
        args_payload=args_payload,
        new_run=bool(args.new_run),
    )
    console.print(f"Wrote {row_count} VLM probe rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
