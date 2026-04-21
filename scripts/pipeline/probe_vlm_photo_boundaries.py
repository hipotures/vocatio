#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
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

from lib import manual_vlm_models
from lib import vlm_prompt_templates
from lib import window_schema as window_schema_lib
from lib.image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1
from lib.ml_boundary_dataset import normalize_timestamp
from lib.ml_boundary_features import build_candidate_feature_row
from lib.ml_boundary_training_data import (
    HEURISTIC_VALUE_COLUMNS,
    heuristic_pair_join_columns_for_window_radius,
    image_feature_columns_for_mode as training_image_feature_columns_for_mode,
)
from lib.media_manifest import read_media_manifest, select_photo_rows
from lib.pipeline_io import atomic_write_json
from lib.photo_pre_model_annotations import (
    DEFAULT_OUTPUT_DIRNAME as DEFAULT_PHOTO_PRE_MODEL_DIRNAME,
    load_photo_pre_model_annotations_by_relative_path,
)
from lib.vlm_transport import VlmRequest, build_provider_request_payload, get_vlm_capabilities, run_vlm_request
from lib.window_radius_contract import (
    DEFAULT_WINDOW_RADIUS,
    build_centered_window_bounds,
    build_window_start_indexes,
    positive_window_radius_arg,
    window_radius_to_window_size,
)
from lib.workspace_dir import load_vocatio_config, resolve_workspace_dir
from train_ml_boundary_verifier import (
    BOUNDARY_MODEL_DIRNAME,
    FEATURE_COLUMNS_FILENAME,
    LEFT_SEGMENT_TYPE_MODEL_DIRNAME,
    RIGHT_SEGMENT_TYPE_MODEL_DIRNAME,
    TRAINING_METADATA_FILENAME,
    load_multimodal_predictor_class,
    load_tabular_predictor_class,
)

console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTO_EMBEDDED_MANIFEST_FILENAME = "photo_embedded_manifest.csv"
PHOTO_MANIFEST_FILENAME = "media_manifest.csv"
PHOTO_BOUNDARY_SCORES_FILENAME = "photo_boundary_scores.csv"
DEFAULT_OUTPUT_FILENAME = "vlm_boundary_results.csv"
RUN_METADATA_DIRNAME = "vlm_runs"
DEFAULT_IMAGE_VARIANT = "preview"
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
DEFAULT_PROMPT_TEMPLATE_ID = "group_compare_long"
DEFAULT_RESPONSE_CONTRACT_ID = "grouped_v1"
DEFAULT_PHOTO_PRE_MODEL_DIR = DEFAULT_PHOTO_PRE_MODEL_DIRNAME
DEFAULT_PROVIDER = "ollama"
DEFAULT_ML_MODEL_RUN_ID = ""
DEFAULT_WINDOW_SCHEMA = window_schema_lib.DEFAULT_WINDOW_SCHEMA
DEFAULT_WINDOW_SCHEMA_SEED = window_schema_lib.DEFAULT_WINDOW_SCHEMA_SEED
VLM_MODELS_CONFIG_PATH = REPO_ROOT / "conf" / "vlm_models.yaml"
PROMPT_TEMPLATES_CONFIG_PATH = REPO_ROOT / "conf" / "vlm_prompt_templates.yaml"
VLM_WORKFLOW_CONFIG_KEYS = frozenset(
    {
        "VLM_EMBEDDED_MANIFEST_CSV",
        "VLM_PHOTO_MANIFEST_CSV",
        "VLM_IMAGE_VARIANT",
        "VLM_WINDOW_RADIUS",
        "VLM_WINDOW_SCHEMA",
        "VLM_WINDOW_SCHEMA_SEED",
        "VLM_BOUNDARY_GAP_SECONDS",
        "VLM_MAX_BATCHES",
        "VLM_PHOTO_PRE_MODEL_DIR",
        "VLM_PROMPT_TEMPLATE_ID",
        "VLM_PROMPT_TEMPLATE_FILE",
        "VLM_ML_MODEL_RUN_ID",
        "VLM_DUMP_DEBUG_DIR",
    }
)
VLM_IGNORED_LEGACY_CONFIG_KEYS = frozenset({"VLM_WINDOW_SIZE", "VLM_OVERLAP"})
VLM_MODEL_ARG_SPECS = (
    ("VLM_PROVIDER", "provider", DEFAULT_PROVIDER),
    ("VLM_MODEL", "model", DEFAULT_MODEL_NAME),
    ("VLM_BASE_URL", "ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
    ("VLM_CONTEXT_TOKENS", "ollama_num_ctx", None),
    ("VLM_MAX_OUTPUT_TOKENS", "ollama_num_predict", None),
    ("VLM_KEEP_ALIVE", "ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE),
    ("VLM_TIMEOUT_SECONDS", "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
    ("VLM_TEMPERATURE", "temperature", DEFAULT_TEMPERATURE),
    ("VLM_REASONING_LEVEL", "ollama_think", DEFAULT_OLLAMA_THINK),
    ("VLM_RESPONSE_SCHEMA_MODE", "response_schema_mode", DEFAULT_RESPONSE_SCHEMA_MODE),
    ("VLM_JSON_VALIDATION_MODE", "json_validation_mode", DEFAULT_JSON_VALIDATION_MODE),
)
SEGMENT_TYPES = ("dance", "ceremony", "audience", "rehearsal", "other")
ML_SEGMENT_TYPE_TO_PROMPT_LABEL = {
    "performance": "dance",
    "ceremony": "ceremony",
    "warmup": "rehearsal",
}
GUI_TYPE_CODE_BY_SEGMENT_TYPE = {
    "performance": "P",
    "ceremony": "C",
    "warmup": "W",
    "dance": "D",
    "audience": "A",
    "rehearsal": "R",
    "other": "O",
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
    "window_radius",
    "relative_paths_json",
    "filenames_json",
    "image_paths_json",
    "delta_from_first_seconds_json",
    "delta_from_previous_seconds_json",
    "decision",
    "semantic_decision",
    "semantic_group_a_segment_type",
    "semantic_group_b_segment_type",
    "response_contract_id",
    "cut_after_local_index",
    "cut_after_global_row",
    "cut_left_relative_path",
    "cut_right_relative_path",
    "reason",
    "response_status",
    "raw_response",
]
RESUME_CONFIG_KEYS = (
    "embedded_manifest_csv",
    "photo_manifest_csv",
    "provider",
    "image_variant",
    "window_radius",
    "window_schema",
    "window_schema_seed",
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
    "prompt_template_id",
    "prompt_template_file",
    "response_contract_id",
    "photo_pre_model_dir",
    "effective_ml_model_run_id",
    "effective_extra_instructions",
)


class MlHintContext(NamedTuple):
    run_id: str
    mode: str
    model_dir: Path
    window_radius: int
    boundary_predictor: object
    left_segment_type_predictor: object
    right_segment_type_predictor: object
    boundary_feature_columns: list[str]
    left_segment_type_feature_columns: list[str]
    right_segment_type_feature_columns: list[str]
    descriptor_field_registry: dict[str, str]


class MlHintPrediction(NamedTuple):
    boundary_prediction: bool
    boundary_confidence: float
    left_segment_type_prediction: str
    left_segment_type_confidence: float
    right_segment_type_prediction: str
    right_segment_type_confidence: float


def _validate_ml_hint_feature_columns_contract(
    *,
    mode: str,
    window_radius: int,
    image_feature_columns: Sequence[str],
    boundary_feature_columns: Sequence[str],
    left_segment_type_feature_columns: Sequence[str],
    right_segment_type_feature_columns: Sequence[str],
) -> None:
    expected_image_feature_columns = training_image_feature_columns_for_mode(
        mode,
        window_radius=window_radius,
    )
    if list(image_feature_columns) != expected_image_feature_columns:
        raise ValueError(
            "feature_columns.json is inconsistent with training "
            f"window_radius={window_radius}: image_feature_columns mismatch"
        )
    descriptor_field_registry = _build_descriptor_field_registry_from_feature_columns(
        sorted(
            set(
                boundary_feature_columns
                + left_segment_type_feature_columns
                + right_segment_type_feature_columns
            )
        )
    )
    synthetic_candidate_row: dict[str, object] = {"window_radius": window_radius}
    for frame_index in range(1, window_radius_to_window_size(window_radius) + 1):
        synthetic_candidate_row[f"frame_{frame_index:02d}_timestamp"] = float(frame_index)
        synthetic_candidate_row[f"frame_{frame_index:02d}_photo_id"] = f"photo-{frame_index:02d}"
    expected_predictor_feature_columns = list(
        build_candidate_feature_row(
            synthetic_candidate_row,
            descriptors={},
            embeddings=None,
            descriptor_field_registry=descriptor_field_registry,
            heuristic_features=None,
            window_radius=window_radius,
        ).keys()
    ) + expected_image_feature_columns

    def validate_predictor_columns(predictor_name: str, columns: Sequence[str]) -> None:
        normalized_columns = [str(column_name) for column_name in columns]
        if normalized_columns != expected_predictor_feature_columns:
            raise ValueError(
                "feature_columns.json is inconsistent with training "
                f"window_radius={window_radius}: {predictor_name} mismatch"
            )

    validate_predictor_columns("boundary_feature_columns", boundary_feature_columns)
    validate_predictor_columns("left_segment_type_feature_columns", left_segment_type_feature_columns)
    validate_predictor_columns("right_segment_type_feature_columns", right_segment_type_feature_columns)

SYSTEM_PROMPT = (
    "You analyze consecutive stage performance photos. "
    "Decide whether the candidate gap between group_a and group_b is a real segment boundary. "
    "Return only valid JSON with keys decision, group_a_segment_type, group_b_segment_type, "
    "frame_notes, primary_evidence, and summary."
)


def build_system_prompt(response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE) -> str:
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


def map_ml_segment_label_to_prompt_label(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized == "":
        return "other"
    return ML_SEGMENT_TYPE_TO_PROMPT_LABEL.get(normalized, normalized)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
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
        "--window-radius",
        type=positive_window_radius_arg,
        default=DEFAULT_WINDOW_RADIUS,
        help=(
            "Number of photos to include on each side of the main candidate cut. "
            f"Internal window size is 2 * radius. Default: {DEFAULT_WINDOW_RADIUS}"
        ),
    )
    parser.add_argument(
        "--window-schema",
        choices=window_schema_lib.WINDOW_SCHEMA_VALUES,
        default=DEFAULT_WINDOW_SCHEMA,
        help=f"Photo selection schema inside each segment. Default: {DEFAULT_WINDOW_SCHEMA}",
    )
    parser.add_argument(
        "--window-schema-seed",
        type=int,
        default=DEFAULT_WINDOW_SCHEMA_SEED,
        help=f"Deterministic seed for schema-based selection. Default: {DEFAULT_WINDOW_SCHEMA_SEED}",
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
        "--prompt-template-id",
        default=DEFAULT_PROMPT_TEMPLATE_ID,
        help=f"Prompt template identifier reserved for grouped prompt variants. Default: {DEFAULT_PROMPT_TEMPLATE_ID}",
    )
    parser.add_argument(
        "--prompt-template-file",
        default="",
        help="Optional prompt template file override reserved for grouped prompt variants.",
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
    cli_provided: set[str] = set()
    for token in raw_args:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option_token = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option_token)
        if action is not None:
            cli_provided.add(str(action.dest))
    args = parser.parse_args(raw_args)
    setattr(args, "_cli_provided", frozenset(cli_provided))
    return args


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


def _validate_supported_vlm_config_keys(config: Mapping[str, str]) -> None:
    supported_keys = set(manual_vlm_models.VLM_MODEL_FIELDS)
    supported_keys.update(VLM_WORKFLOW_CONFIG_KEYS)
    supported_keys.update(VLM_IGNORED_LEGACY_CONFIG_KEYS)
    unsupported_keys = sorted(
        key
        for key in config
        if key.startswith("VLM_") and key not in supported_keys
    )
    if unsupported_keys:
        raise ValueError(
            "unsupported VLM config key(s) in .vocatio: "
            + ", ".join(unsupported_keys)
        )


def _resolve_vlm_preset_config(config: Mapping[str, str]) -> Mapping[str, Any]:
    loaded = manual_vlm_models.load_manual_vlm_models(VLM_MODELS_CONFIG_PATH)
    return manual_vlm_models.resolve_vlm_model_config(loaded.models, config)


def _arg_matches_default(args: argparse.Namespace, attr: str, default_value: Any) -> bool:
    return getattr(args, attr) == default_value


def _arg_is_inheritable(
    args: argparse.Namespace,
    cli_provided: frozenset[str],
    attr: str,
    default_value: Any,
) -> bool:
    return attr not in cli_provided and _arg_matches_default(args, attr, default_value)


def _needs_vlm_preset_resolution(args: argparse.Namespace, config: Mapping[str, str]) -> bool:
    cli_provided = frozenset(getattr(args, "_cli_provided", frozenset()))
    preset_name = str(config.get("VLM_NAME", "") or "").strip()
    if preset_name:
        return any(
            _arg_is_inheritable(args, cli_provided, attr_name, default_value)
            for _field_name, attr_name, default_value in VLM_MODEL_ARG_SPECS
        )
    return any(
        str(config.get(field_name, "") or "").strip()
        and _arg_is_inheritable(args, cli_provided, attr_name, default_value)
        for field_name, attr_name, default_value in VLM_MODEL_ARG_SPECS
    )


def apply_vocatio_defaults(args: argparse.Namespace, day_dir: Path) -> argparse.Namespace:
    config = load_vocatio_config(day_dir)
    _validate_supported_vlm_config_keys(config)
    cli_provided = frozenset(getattr(args, "_cli_provided", frozenset()))

    def apply_string(attr: str, default_value: str, config_key: str) -> None:
        if _arg_is_inheritable(args, cli_provided, attr, default_value):
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, configured)

    def apply_preset_string(attr: str, default_value: str, configured: Any) -> None:
        if _arg_is_inheritable(args, cli_provided, attr, default_value):
            setattr(args, attr, str(configured))

    def apply_preset_optional_int(attr: str, configured: Any) -> None:
        if configured is None:
            return
        if attr not in cli_provided and getattr(args, attr) is None:
            setattr(args, attr, int(configured))

    def apply_preset_float(attr: str, default_value: float, configured: Any) -> None:
        if _arg_is_inheritable(args, cli_provided, attr, default_value):
            setattr(args, attr, float(configured))

    def apply_int(attr: str, default_value: int, config_key: str, parser) -> None:
        if _arg_is_inheritable(args, cli_provided, attr, default_value):
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, parser(configured))

    def apply_window_schema(attr: str, default_value: str, config_key: str) -> None:
        if _arg_is_inheritable(args, cli_provided, attr, default_value):
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, window_schema_lib.parse_window_schema(configured))

    if _needs_vlm_preset_resolution(args, config):
        preset_config = _resolve_vlm_preset_config(config)
        apply_preset_string("provider", DEFAULT_PROVIDER, preset_config["VLM_PROVIDER"])
        apply_preset_string("model", DEFAULT_MODEL_NAME, preset_config["VLM_MODEL"])
        apply_preset_string("ollama_base_url", DEFAULT_OLLAMA_BASE_URL, preset_config["VLM_BASE_URL"])
        apply_preset_optional_int("ollama_num_ctx", preset_config.get("VLM_CONTEXT_TOKENS"))
        apply_preset_optional_int("ollama_num_predict", preset_config["VLM_MAX_OUTPUT_TOKENS"])
        if "VLM_KEEP_ALIVE" in preset_config:
            apply_preset_string("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE, preset_config["VLM_KEEP_ALIVE"])
        apply_preset_float("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, preset_config["VLM_TIMEOUT_SECONDS"])
        apply_preset_float("temperature", DEFAULT_TEMPERATURE, preset_config["VLM_TEMPERATURE"])
        if "VLM_REASONING_LEVEL" in preset_config:
            apply_preset_string("ollama_think", DEFAULT_OLLAMA_THINK, preset_config["VLM_REASONING_LEVEL"])
        apply_preset_string(
            "response_schema_mode",
            DEFAULT_RESPONSE_SCHEMA_MODE,
            preset_config["VLM_RESPONSE_SCHEMA_MODE"],
        )
        apply_preset_string(
            "json_validation_mode",
            DEFAULT_JSON_VALIDATION_MODE,
            preset_config["VLM_JSON_VALIDATION_MODE"],
        )
    apply_string("embedded_manifest_csv", PHOTO_EMBEDDED_MANIFEST_FILENAME, "VLM_EMBEDDED_MANIFEST_CSV")
    apply_string("photo_manifest_csv", PHOTO_MANIFEST_FILENAME, "VLM_PHOTO_MANIFEST_CSV")
    apply_string("image_variant", DEFAULT_IMAGE_VARIANT, "VLM_IMAGE_VARIANT")
    apply_int("window_radius", DEFAULT_WINDOW_RADIUS, "VLM_WINDOW_RADIUS", positive_window_radius_arg)
    apply_window_schema("window_schema", DEFAULT_WINDOW_SCHEMA, "VLM_WINDOW_SCHEMA")
    apply_int("window_schema_seed", DEFAULT_WINDOW_SCHEMA_SEED, "VLM_WINDOW_SCHEMA_SEED", int)
    apply_int(
        "boundary_gap_seconds",
        DEFAULT_BOUNDARY_GAP_SECONDS,
        "VLM_BOUNDARY_GAP_SECONDS",
        non_negative_int_arg,
    )
    apply_int("max_batches", DEFAULT_MAX_BATCHES, "VLM_MAX_BATCHES", max_batches_arg)
    apply_string("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID, "VLM_PROMPT_TEMPLATE_ID")
    apply_string("prompt_template_file", "", "VLM_PROMPT_TEMPLATE_FILE")
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


def segment_bounds_for_cut_index(
    rows: Sequence[Mapping[str, str]],
    *,
    cut_index: int,
    boundary_gap_seconds: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if cut_index < 0 or cut_index >= len(rows) - 1:
        raise ValueError("cut_index must reference a boundary between consecutive rows")
    left_start = 0
    for index in range(cut_index - 1, -1, -1):
        left_epoch_ms = int(str(rows[index]["start_epoch_ms"]))
        right_epoch_ms = int(str(rows[index + 1]["start_epoch_ms"]))
        if rounded_seconds(right_epoch_ms - left_epoch_ms) >= boundary_gap_seconds:
            left_start = index + 1
            break
    right_end = len(rows)
    for index in range(cut_index + 1, len(rows) - 1):
        left_epoch_ms = int(str(rows[index]["start_epoch_ms"]))
        right_epoch_ms = int(str(rows[index + 1]["start_epoch_ms"]))
        if rounded_seconds(right_epoch_ms - left_epoch_ms) >= boundary_gap_seconds:
            right_end = index + 1
            break
    return (left_start, cut_index + 1), (cut_index + 1, right_end)


def build_candidate_windows(
    rows: Sequence[Mapping[str, str]],
    window_radius: int,
    boundary_gap_seconds: int,
) -> List[Dict[str, int]]:
    window_size = window_radius_to_window_size(window_radius)
    total_rows = len(rows)
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    candidates_by_start: Dict[int, Dict[str, int]] = {}
    for cut_index in range(total_rows - 1):
        left_epoch_ms = int(str(rows[cut_index]["start_epoch_ms"]))
        right_epoch_ms = int(str(rows[cut_index + 1]["start_epoch_ms"]))
        time_gap_seconds = rounded_seconds(right_epoch_ms - left_epoch_ms)
        if time_gap_seconds <= boundary_gap_seconds:
            continue
        start_index, _ = build_centered_window_bounds(total_rows, cut_index, window_radius)
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
    window_radius: int,
    boundary_gap_seconds: int,
) -> List[int]:
    return [
        int(candidate["start_index"])
        for candidate in build_candidate_windows(
            rows,
            window_radius=window_radius,
            boundary_gap_seconds=boundary_gap_seconds,
        )
    ]


def build_window_rows_for_cut_index(
    *,
    joined_rows: Sequence[Mapping[str, str]],
    cut_index: int,
    window_radius: int,
    window_schema: str,
    window_schema_seed: int,
    boundary_gap_seconds: int,
) -> tuple[list[Mapping[str, str]], int]:
    left_bounds, right_bounds = segment_bounds_for_cut_index(
        joined_rows,
        cut_index=cut_index,
        boundary_gap_seconds=boundary_gap_seconds,
    )
    left_rows = window_schema_lib.select_segment_rows(
        joined_rows[left_bounds[0] : left_bounds[1]],
        radius=window_radius,
        schema=window_schema,
        gap_side="left",
        schema_seed=window_schema_seed,
    )
    right_rows = window_schema_lib.select_segment_rows(
        joined_rows[right_bounds[0] : right_bounds[1]],
        radius=window_radius,
        schema=window_schema,
        gap_side="right",
        schema_seed=window_schema_seed,
    )
    return [*left_rows, *right_rows], len(left_rows)


def build_manual_vlm_window_rows(
    joined_rows: Sequence[Mapping[str, str]],
    *,
    anchor_pair: Mapping[str, Any],
    window_radius: int,
) -> List[Mapping[str, str]]:
    left_row_index = int(anchor_pair.get("left_row_index", -1))
    right_row_index = int(anchor_pair.get("right_row_index", -1))
    if left_row_index < 0 or right_row_index < 0:
        raise ValueError("manual VLM anchor rows are unavailable")
    if left_row_index >= right_row_index:
        raise ValueError("manual VLM anchors must be ordered by joined rows")
    if right_row_index >= len(joined_rows):
        raise ValueError("manual VLM anchor rows are outside the joined manifest")
    context_count = window_radius - 1
    if left_row_index < context_count or right_row_index + context_count >= len(joined_rows):
        raise ValueError(
            "manual VLM window requires "
            f"{context_count} rows before the left anchor and {context_count} rows after the right anchor"
        )
    window_rows = (
        list(joined_rows[left_row_index - context_count : left_row_index])
        + [joined_rows[left_row_index], joined_rows[right_row_index]]
        + list(joined_rows[right_row_index + 1 : right_row_index + context_count + 1])
    )
    expected_window_size = window_radius_to_window_size(window_radius)
    if len(window_rows) != expected_window_size:
        raise ValueError(f"manual VLM window must contain exactly {expected_window_size} rows")
    return [dict(row) for row in window_rows]


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


def _build_descriptor_field_registry_from_feature_columns(
    feature_columns: Sequence[str],
) -> dict[str, str]:
    excluded_side_feature_names = {
        "left_internal_gap_mean",
        "right_internal_gap_mean",
        "left_consistency_score",
        "right_consistency_score",
    }
    registry: dict[str, str] = {}
    for column_name in feature_columns:
        if column_name in excluded_side_feature_names:
            continue
        if column_name.startswith("left_"):
            descriptor_name = column_name[len("left_") :]
        elif column_name.startswith("right_"):
            descriptor_name = column_name[len("right_") :]
        else:
            continue
        suffix = descriptor_name.rsplit("_", 1)
        if len(suffix) == 2 and len(suffix[1]) == 2 and suffix[1].isdigit():
            registry[suffix[0]] = "multivalue"
            continue
        registry.setdefault(descriptor_name, "scalar")
    return registry


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
    window_radius = int(training_metadata["window_radius"])
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1")
    image_feature_columns = [str(value) for value in feature_columns_payload.get("image_feature_columns", [])]
    boundary_feature_columns = [str(value) for value in feature_columns_payload.get("boundary_feature_columns", [])]
    left_segment_type_feature_columns = [
        str(value) for value in feature_columns_payload.get("left_segment_type_feature_columns", [])
    ]
    right_segment_type_feature_columns = [
        str(value) for value in feature_columns_payload.get("right_segment_type_feature_columns", [])
    ]
    _validate_ml_hint_feature_columns_contract(
        mode=mode,
        window_radius=window_radius,
        image_feature_columns=image_feature_columns,
        boundary_feature_columns=boundary_feature_columns,
        left_segment_type_feature_columns=left_segment_type_feature_columns,
        right_segment_type_feature_columns=right_segment_type_feature_columns,
    )
    descriptor_field_registry = _build_descriptor_field_registry_from_feature_columns(
        sorted(
            set(
                boundary_feature_columns
                + left_segment_type_feature_columns
                + right_segment_type_feature_columns
            )
        )
    )
    predictor_class = load_multimodal_predictor_class() if mode == "tabular_plus_thumbnail" else load_tabular_predictor_class()
    if not hasattr(predictor_class, "load"):
        raise ValueError(f"{predictor_class.__name__} does not support load()")
    left_segment_type_model_dir = ml_model_dir / LEFT_SEGMENT_TYPE_MODEL_DIRNAME
    right_segment_type_model_dir = ml_model_dir / RIGHT_SEGMENT_TYPE_MODEL_DIRNAME
    boundary_model_dir = ml_model_dir / BOUNDARY_MODEL_DIRNAME
    left_segment_type_predictor = predictor_class.load(str(left_segment_type_model_dir))
    right_segment_type_predictor = predictor_class.load(str(right_segment_type_model_dir))
    boundary_predictor = predictor_class.load(str(boundary_model_dir))
    return MlHintContext(
        run_id=ml_model_run_id,
        mode=mode,
        model_dir=ml_model_dir,
        window_radius=window_radius,
        boundary_predictor=boundary_predictor,
        left_segment_type_predictor=left_segment_type_predictor,
        right_segment_type_predictor=right_segment_type_predictor,
        boundary_feature_columns=boundary_feature_columns,
        left_segment_type_feature_columns=left_segment_type_feature_columns,
        right_segment_type_feature_columns=right_segment_type_feature_columns,
        descriptor_field_registry=descriptor_field_registry,
    )


def _build_ml_candidate_window_rows(
    joined_rows: Sequence[Mapping[str, str]],
    *,
    cut_index: int,
    window_radius: int,
) -> Optional[list[Mapping[str, str]]]:
    window_size = window_radius_to_window_size(window_radius)
    if len(joined_rows) < window_size:
        return None
    try:
        window_start, window_end = build_centered_window_bounds(
            len(joined_rows),
            cut_index,
            window_radius,
        )
    except ValueError:
        return None
    rows = list(joined_rows[window_start:window_end])
    if len(rows) != window_size:
        return None
    return rows


def _build_ml_candidate_row(
    rows: Sequence[Mapping[str, str]],
    *,
    day_id: str,
    window_radius: int,
) -> dict[str, object]:
    window_size = window_radius_to_window_size(window_radius)
    if len(rows) != window_size:
        raise ValueError(f"ML candidate rows must contain exactly {window_size} frames")
    normalized_day_id = day_id.strip()
    if normalized_day_id == "":
        raise ValueError("day_id must not be blank for ML prompt inference")
    candidate_row: dict[str, object] = {
        "day_id": normalized_day_id,
        "window_radius": window_radius,
        "center_left_photo_id": str(
            rows[window_radius - 1].get("photo_id", "") or rows[window_radius - 1].get("relative_path", "")
        ).strip(),
        "center_right_photo_id": str(
            rows[window_radius].get("photo_id", "") or rows[window_radius].get("relative_path", "")
        ).strip(),
        "boundary": False,
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
    window_radius: int,
) -> dict[str, dict[str, object]]:
    if photo_pre_model_dir is None or not photo_pre_model_dir.exists():
        return {}
    frame_count = window_radius_to_window_size(window_radius)
    relative_paths = [
        str(candidate_row.get(f"frame_{frame_index:02d}_relpath", "") or "").strip()
        for frame_index in range(1, frame_count + 1)
    ]
    annotations_by_relative_path = load_photo_pre_model_annotations_by_relative_path(
        photo_pre_model_dir,
        relative_paths,
    )
    descriptors: dict[str, dict[str, object]] = {}
    for frame_index in range(1, frame_count + 1):
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
    *,
    window_radius: int,
) -> dict[str, dict[str, str]]:
    heuristic_features: dict[str, dict[str, str]] = {}
    for pair_name, left_column, right_column in heuristic_pair_join_columns_for_window_radius(window_radius):
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
    candidate_window_radius = int(str(candidate_row.get("window_radius", "") or "").strip())
    if candidate_window_radius != ml_hint_context.window_radius:
        raise ValueError(
            "ml model window_radius mismatch: "
            f"runtime={candidate_window_radius}, artifact={ml_hint_context.window_radius}"
        )
    descriptors = _load_ml_candidate_descriptors(
        candidate_row,
        photo_pre_model_dir=photo_pre_model_dir,
        window_radius=ml_hint_context.window_radius,
    )
    heuristic_features = _build_ml_candidate_heuristic_features(
        candidate_row,
        boundary_rows_by_pair,
        window_radius=ml_hint_context.window_radius,
    )
    derived_features = build_candidate_feature_row(
        candidate_row,
        descriptors=descriptors,
        embeddings=None,
        descriptor_field_registry=ml_hint_context.descriptor_field_registry,
        heuristic_features=heuristic_features,
        window_radius=ml_hint_context.window_radius,
    )
    left_segment_type_row = _build_predictor_feature_row(
        feature_columns=ml_hint_context.left_segment_type_feature_columns,
        candidate_row=candidate_row,
        derived_features=derived_features,
    )
    right_segment_type_row = _build_predictor_feature_row(
        feature_columns=ml_hint_context.right_segment_type_feature_columns,
        candidate_row=candidate_row,
        derived_features=derived_features,
    )
    boundary_row = _build_predictor_feature_row(
        feature_columns=ml_hint_context.boundary_feature_columns,
        candidate_row=candidate_row,
        derived_features=derived_features,
    )
    left_segment_type_frame = _to_model_frame([left_segment_type_row])
    right_segment_type_frame = _to_model_frame([right_segment_type_row])
    boundary_frame = _to_model_frame([boundary_row])

    left_segment_type_prediction = str(
        _to_plain_list(ml_hint_context.left_segment_type_predictor.predict(left_segment_type_frame))[0]
    ).strip()
    left_segment_type_confidence = _extract_label_probability(
        ml_hint_context.left_segment_type_predictor.predict_proba(left_segment_type_frame),
        left_segment_type_prediction,
    )
    right_segment_type_prediction = str(
        _to_plain_list(ml_hint_context.right_segment_type_predictor.predict(right_segment_type_frame))[0]
    ).strip()
    right_segment_type_confidence = _extract_label_probability(
        ml_hint_context.right_segment_type_predictor.predict_proba(right_segment_type_frame),
        right_segment_type_prediction,
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
        left_segment_type_prediction=left_segment_type_prediction,
        left_segment_type_confidence=left_segment_type_confidence,
        right_segment_type_prediction=right_segment_type_prediction,
        right_segment_type_confidence=right_segment_type_confidence,
    )


def build_ml_hint_lines(
    prediction: Optional[MlHintPrediction],
) -> List[str]:
    if prediction is None:
        return ["ML hints are unavailable for this window."]
    boundary_label = "cut" if prediction.boundary_prediction else "no cut"
    left_segment_label = map_ml_segment_label_to_prompt_label(prediction.left_segment_type_prediction)
    right_segment_label = map_ml_segment_label_to_prompt_label(prediction.right_segment_type_prediction)
    return [
        f"ML hint for the main candidate gap in this window: likely {boundary_label} (confidence {prediction.boundary_confidence:.2f}).",
        f"ML hint for the left side of the candidate gap: likely {left_segment_label} (confidence {prediction.left_segment_type_confidence:.2f}).",
        f"ML hint for the right side of the candidate gap: likely {right_segment_label} (confidence {prediction.right_segment_type_confidence:.2f}).",
    ]


def build_ml_hint_lines_for_window_rows(
    *,
    day_id: str,
    window_rows: Sequence[Mapping[str, str]],
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
    ml_hint_context: Optional[MlHintContext],
    runtime_window_radius: int,
) -> List[str]:
    if ml_hint_context is None:
        return build_ml_hint_lines(None)
    expected_window_size = window_radius_to_window_size(runtime_window_radius)
    if len(window_rows) != expected_window_size:
        return build_ml_hint_lines(None)
    prediction = predict_ml_hint_for_candidate(
        ml_hint_context=ml_hint_context,
        candidate_row=_build_ml_candidate_row(
            list(window_rows),
            day_id=day_id,
            window_radius=runtime_window_radius,
        ),
        boundary_rows_by_pair=boundary_rows_by_pair,
        photo_pre_model_dir=photo_pre_model_dir,
    )
    return build_ml_hint_lines(prediction)


def build_ml_hint_lines_for_candidate(
    *,
    day_id: str,
    joined_rows: Sequence[Mapping[str, str]],
    cut_index: int,
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
    ml_hint_context: Optional[MlHintContext],
    runtime_window_radius: int,
) -> List[str]:
    if ml_hint_context is None:
        return build_ml_hint_lines(None)
    candidate_rows = _build_ml_candidate_window_rows(
        joined_rows,
        cut_index=cut_index,
        window_radius=runtime_window_radius,
    )
    if candidate_rows is None:
        return build_ml_hint_lines(None)
    return build_ml_hint_lines_for_window_rows(
        day_id=day_id,
        window_rows=candidate_rows,
        boundary_rows_by_pair=boundary_rows_by_pair,
        photo_pre_model_dir=photo_pre_model_dir,
        ml_hint_context=ml_hint_context,
        runtime_window_radius=runtime_window_radius,
    )


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


def build_group_frame_ids(window_size: int, group_a_count: Optional[int] = None) -> tuple[List[str], List[str]]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    frame_ids = [f"frame_{index:02d}" for index in range(1, window_size + 1)]
    resolved_group_a_count = max(1, window_size // 2) if group_a_count is None else int(group_a_count)
    if resolved_group_a_count < 1 or resolved_group_a_count > window_size:
        raise ValueError("group_a_count must be between 1 and window_size")
    return frame_ids[:resolved_group_a_count], frame_ids[resolved_group_a_count:]


def build_response_schema(*, group_a_ids: Sequence[str], group_b_ids: Sequence[str]) -> Dict[str, Any]:
    frame_note_variants = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "frame_id": {"type": "string", "enum": [frame_id]},
                "group": {"type": "string", "enum": [group_name]},
                "note": {"type": "string"},
            },
            "required": ["frame_id", "group", "note"],
        }
        for frame_id, group_name in (
            [(frame_id, "group_a") for frame_id in group_a_ids]
            + [(frame_id, "group_b") for frame_id in group_b_ids]
        )
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["same_segment", "different_segments"]},
            "group_a_segment_type": {
                "type": "string",
                "enum": list(SEGMENT_TYPES),
            },
            "group_b_segment_type": {
                "type": "string",
                "enum": list(SEGMENT_TYPES),
            },
            "frame_notes": {
                "type": "array",
                "items": {"oneOf": frame_note_variants},
                "minItems": len(group_a_ids) + len(group_b_ids),
                "maxItems": len(group_a_ids) + len(group_b_ids),
                "allOf": [
                    {
                        "contains": frame_note_variant,
                        "minContains": 1,
                        "maxContains": 1,
                    }
                    for frame_note_variant in frame_note_variants
                ],
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
            "decision",
            "group_a_segment_type",
            "group_b_segment_type",
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
    window_schema: str = DEFAULT_WINDOW_SCHEMA,
    group_a_count: Optional[int] = None,
) -> str:
    group_a_ids, group_b_ids = build_group_frame_ids(window_size, group_a_count=group_a_count)
    ordered_frame_ids = [*group_a_ids, *group_b_ids]
    frames_block = "\n".join(
        [f"{frame_id} = attached image {index}" for index, frame_id in enumerate(ordered_frame_ids, start=1)]
    )
    hints_block = "\n".join(ml_hint_lines) if ml_hint_lines else "ML hints are unavailable for this window."
    normalized_window_schema = window_schema_lib.parse_window_schema(window_schema)
    if normalized_window_schema == "consecutive":
        window_selection_block = (
            f"You will receive {window_size} consecutive stage-event photos.\n\n"
            "Important:\n"
            "The frames are consecutive photos from one chronological sequence.\n"
            "They are not random examples.\n"
        )
    else:
        window_selection_block = (
            f"You will receive {window_size} stage-event photos in chronological order.\n\n"
            "Important:\n"
            "The frames are shown in chronological order from left to right.\n"
            "The frames come from the left and right segments around one candidate gap.\n"
            "They do not have to be consecutive photos.\n"
        )
    if response_schema_mode == "on":
        decisions_block = '\n'.join(['- "same_segment"', '- "different_segments"'])
        response_header = "Allowed decision values:\n"
        response_rules = (
            '- Use "same_segment" if group_a and group_b most likely belong to the same segment.\n'
            '- Use "different_segments" only if group_a and group_b most likely belong to different segments.\n'
        )
        response_object = (
            '{\n'
            '  "decision": "<one of: same_segment, different_segments>",\n'
            f'  "group_a_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            f'  "group_b_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            '  "frame_notes": [\n'
            + ",\n".join(
                [
                    f'    {{"frame_id":"{frame_id}","group":"group_a","note":"<short note>"}}'
                    for frame_id in group_a_ids
                ]
                + [
                    f'    {{"frame_id":"{frame_id}","group":"group_b","note":"<short note>"}}'
                    for frame_id in group_b_ids
                ]
            )
            + '\n  ],\n'
            '  "primary_evidence": [\n'
            '    "<short evidence item>",\n'
            '    "<short evidence item>"\n'
            "  ],\n"
            '  "summary": "<one short sentence>"\n'
            '}'
        )
    else:
        decisions_block = '\n'.join(['- "same_segment"', '- "different_segments"'])
        response_header = "Allowed decisions:\n"
        response_rules = (
            '- Choose "same_segment" if group_a and group_b most likely belong to the same segment.\n'
            '- If there is no clear evidence for a boundary, choose "same_segment".\n'
            '- Choose "different_segments" only if the left group and right group most likely belong to different segments.\n'
        )
        response_object = (
            '{\n'
            '  "decision": "<one of: same_segment, different_segments>",\n'
            f'  "group_a_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            f'  "group_b_segment_type": "<one of: {"|".join(SEGMENT_TYPES)}>",\n'
            '  "frame_notes": [\n'
            + ",\n".join(
                [
                    f'    {{"frame_id":"{frame_id}","group":"group_a","note":"<short note>"}}'
                    for frame_id in group_a_ids
                ]
                + [
                    f'    {{"frame_id":"{frame_id}","group":"group_b","note":"<short note>"}}'
                    for frame_id in group_b_ids
                ]
            )
            + '\n  ],\n'
            '  "primary_evidence": [\n'
            '    "<short evidence item>",\n'
            '    "<short evidence item>"\n'
            "  ],\n"
            '  "summary": "<one short sentence>"\n'
            '}'
        )
    prompt = (
        f"{window_selection_block}"
        "Order:\n"
        f"{frames_block}\n\n"
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
        'Return "different_segments" only if at least one positive boundary condition below is clearly true.\n'
        'If none of the positive boundary conditions is clearly true, return "same_segment".\n\n'
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
        "- background change alone\n"
        "- lighting change alone\n"
        "- background change and lighting change together\n"
        "- a background change caused by camera angle, shooting direction, crop, or zoom on the same stage\n"
        "- a group shot followed by a solo shot of one dancer from the same group\n"
        "- a solo shot followed by a wider group shot from the same ongoing act\n"
        "- a change in visible performer count caused only by framing, crop, zoom, or partial visibility\n\n"
        "Continuity reminders:\n"
        "- Group performance -> single dancer from the same group can still be the same segment.\n"
        "- Single dancer -> wider group view can still be the same segment.\n"
        "- If costume identity and act identity still match, do not create a boundary only because fewer or more dancers are visible.\n"
        "- Ignore background differences if they can be explained by camera direction, framing, crop, zoom, or a different shooting angle on the same stage.\n\n"
        'If the change is only a new pose, new movement phrase, or another choreography moment within the same act, you must return "same_segment".\n\n'
        "Decision priority:\n"
        "1. images\n"
        "2. ML hints\n\n"
        "Images take priority over ML hints.\n\n"
        "ML hints:\n"
        f"{hints_block}\n\n"
        f"{response_header}"
        f"{decisions_block}\n\n"
        "Rules:\n"
        f"{response_rules}"
        "- Always assign group_a_segment_type and group_b_segment_type using only these values: "
        f"{'|'.join(SEGMENT_TYPES)}.\n"
        "- Background change and lighting change, whether alone or together, never justify a boundary.\n"
        "- Background change and lighting change, whether alone or together, never justify a segment-type change.\n"
        "- Include every frame_id exactly once in frame_notes.\n"
        "- Keep frame notes short and concrete.\n"
        '- If the answer is "same_segment", primary_evidence should describe continuity evidence.\n'
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
    request_context_tokens: Optional[int] = None
    request_keep_alive: Optional[str] = None
    request_reasoning_level: Optional[str] = None
    if provider == "ollama":
        request_context_tokens = context_tokens
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
        context_tokens=request_context_tokens,
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


def extract_segment_types(raw_response: str) -> tuple[str, str]:
    if not raw_response.strip():
        return "", ""
    try:
        payload = json.loads(extract_json_object_text(raw_response))
    except Exception:
        return "", ""
    group_a_segment_type = str(payload.get("group_a_segment_type", "") or "").strip()
    group_b_segment_type = str(payload.get("group_b_segment_type", "") or "").strip()
    if group_a_segment_type not in SEGMENT_TYPES:
        group_a_segment_type = ""
    if group_b_segment_type not in SEGMENT_TYPES:
        group_b_segment_type = ""
    return group_a_segment_type, group_b_segment_type


def choose_segment_type(candidates: Sequence[str]) -> str:
    normalized_candidates = [value for value in candidates if value]
    if not normalized_candidates:
        return ""
    counts: Dict[str, int] = {}
    for value in normalized_candidates:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ""
    return ranked[0][0]


def segment_type_to_code(segment_type: str) -> str:
    return GUI_TYPE_CODE_BY_SEGMENT_TYPE.get(segment_type.strip().lower(), "?")


def invalid_response(reason: str) -> Dict[str, str]:
    return {
        "decision": "invalid_response",
        "reason": reason,
        "response_status": "invalid_response",
    }


def parse_model_response(
    raw_response: str,
    *,
    group_a_ids: Sequence[str],
    group_b_ids: Sequence[str],
    response_contract_id: str,
    json_validation_mode: str = DEFAULT_JSON_VALIDATION_MODE,
) -> Dict[str, str]:
    if response_contract_id != DEFAULT_RESPONSE_CONTRACT_ID:
        raise ValueError(f"unsupported response_contract_id: {response_contract_id}")
    try:
        payload = json.loads(extract_json_object_text(raw_response))
    except Exception as error:
        return invalid_response(f"JSON parse error: {error}")
    decision = str(payload.get("decision", "") or "").strip()
    if decision not in {"same_segment", "different_segments"}:
        return invalid_response(f"Invalid decision value: {decision}")
    group_a_segment_type = str(payload.get("group_a_segment_type", "") or "").strip()
    group_b_segment_type = str(payload.get("group_b_segment_type", "") or "").strip()
    if json_validation_mode == "strict" and (
        group_a_segment_type not in SEGMENT_TYPES or group_b_segment_type not in SEGMENT_TYPES
    ):
        return invalid_response("Missing or invalid group segment type value")
    if group_a_segment_type not in SEGMENT_TYPES:
        group_a_segment_type = ""
    if group_b_segment_type not in SEGMENT_TYPES:
        group_b_segment_type = ""
    frame_notes = payload.get("frame_notes")
    if not isinstance(frame_notes, list):
        return invalid_response("Missing frame_notes array")
    expected_frame_groups = {frame_id: "group_a" for frame_id in group_a_ids}
    expected_frame_groups.update({frame_id: "group_b" for frame_id in group_b_ids})
    ordered_frame_ids = [*group_a_ids, *group_b_ids]
    seen_frame_ids: set[str] = set()
    normalized_frame_notes: List[str] = []
    for note_item in frame_notes:
        if not isinstance(note_item, Mapping):
            return invalid_response("Invalid frame_notes item")
        frame_id = str(note_item.get("frame_id", "") or "").strip()
        if not frame_id:
            return invalid_response("Missing frame_id value in frame_notes")
        if frame_id not in expected_frame_groups:
            return invalid_response(f"Invalid frame_id value: {frame_id}")
        if frame_id in seen_frame_ids:
            return invalid_response(f"duplicate frame_id: {frame_id}")
        group = str(note_item.get("group", "") or "").strip()
        expected_group = expected_frame_groups[frame_id]
        if group != expected_group:
            return invalid_response(f"Invalid group for {frame_id}: expected {expected_group}, got {group}")
        note = str(note_item.get("note", "") or "").strip()
        if not note:
            return invalid_response(f"Missing frame_notes value for {frame_id}")
        seen_frame_ids.add(frame_id)
        normalized_frame_notes.append(f"{frame_id}({group})={note}")
    for frame_id in ordered_frame_ids:
        if frame_id not in seen_frame_ids:
            return invalid_response(f"Missing frame_notes value for {frame_id}")
    primary_evidence = payload.get("primary_evidence")
    if not isinstance(primary_evidence, list):
        return invalid_response("Missing primary_evidence list")
    normalized_evidence = [str(item).strip() for item in primary_evidence if str(item).strip()]
    if not normalized_evidence:
        return invalid_response("Missing primary_evidence values")
    summary = str(payload.get("summary", "") or "").strip()
    if not summary:
        return invalid_response("Missing summary value")
    reason = (
        f"Group A segment type: {group_a_segment_type or '?'} | "
        f"Group B segment type: {group_b_segment_type or '?'} | "
        f"Frame notes: {'; '.join(normalized_frame_notes)} | "
        f"Primary evidence: {'; '.join(normalized_evidence)} | "
        f"Summary: {summary}"
    )
    compatibility_decision = "no_cut" if decision == "same_segment" else f"cut_after_{len(group_a_ids)}"
    return {
        "decision": compatibility_decision,
        "semantic_decision": decision,
        "semantic_group_a_segment_type": group_a_segment_type,
        "semantic_group_b_segment_type": group_b_segment_type,
        "response_contract_id": response_contract_id,
        "reason": reason,
        "response_status": "ok",
    }


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
    window_radius: int,
    raw_response: str,
    parsed_response: Mapping[str, str],
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    global_row_numbers: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    decision = str(parsed_response["decision"])
    cut_after_local_index = ""
    cut_after_global_row = ""
    cut_left_relative_path = ""
    cut_right_relative_path = ""
    resolved_global_row_numbers = list(global_row_numbers) if global_row_numbers is not None else list(
        range(start_row, start_row + len(rows))
    )
    if decision.startswith("cut_after_"):
        cut_after_local_index = decision.removeprefix("cut_after_")
        local_index = int(cut_after_local_index)
        cut_after_global_row = str(resolved_global_row_numbers[local_index - 1])
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
        "window_radius": str(window_radius),
        "relative_paths_json": json.dumps([str(row["relative_path"]) for row in rows], ensure_ascii=True),
        "filenames_json": json.dumps([str(row["filename"]) for row in rows], ensure_ascii=True),
        "image_paths_json": json.dumps([str(row["image_path"]) for row in rows], ensure_ascii=True),
        "delta_from_first_seconds_json": json.dumps(delta_from_first_seconds, ensure_ascii=True),
        "delta_from_previous_seconds_json": json.dumps(delta_from_previous_seconds, ensure_ascii=True),
        "decision": decision,
        "semantic_decision": str(parsed_response.get("semantic_decision", "")),
        "semantic_group_a_segment_type": str(parsed_response.get("semantic_group_a_segment_type", "")),
        "semantic_group_b_segment_type": str(parsed_response.get("semantic_group_b_segment_type", "")),
        "response_contract_id": str(parsed_response.get("response_contract_id", DEFAULT_RESPONSE_CONTRACT_ID)),
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


def build_example_ml_hint_lines() -> list[str]:
    return [
        "ML hint for the main candidate gap in this window: likely cut (confidence 0.82).",
        "ML hint for the left side of the candidate gap: likely dance (confidence 0.74).",
        "ML hint for the right side of the candidate gap: likely ceremony (confidence 0.73).",
    ]


def build_runtime_group_ids(group_a_count: int, group_b_count: int) -> tuple[list[str], list[str]]:
    return (
        [f"a_{index:02d}" for index in range(1, group_a_count + 1)],
        [f"b_{index:02d}" for index in range(1, group_b_count + 1)],
    )


def resolve_prompt_template_path(
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_file: str = "",
) -> Path:
    normalized_template_file = str(prompt_template_file or "").strip()
    if normalized_template_file:
        template_path = Path(normalized_template_file).expanduser()
        return template_path if template_path.is_absolute() else (REPO_ROOT / template_path).resolve()
    normalized_template_id = str(prompt_template_id or "").strip() or DEFAULT_PROMPT_TEMPLATE_ID
    templates_config = vlm_prompt_templates.load_prompt_templates_config(PROMPT_TEMPLATES_CONFIG_PATH)
    for template_entry in templates_config.templates:
        if template_entry.id == normalized_template_id:
            return (REPO_ROOT / template_entry.file).resolve()
    raise ValueError(f"Unknown prompt_template_id: {normalized_template_id}")


def render_runtime_user_prompt(
    *,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_file: str = "",
    group_a_ids: Sequence[str],
    group_b_ids: Sequence[str],
    ml_hint_lines: Sequence[str],
    extra_instructions: str = "",
) -> tuple[str, Path]:
    template_path = resolve_prompt_template_path(
        prompt_template_id=prompt_template_id,
        prompt_template_file=prompt_template_file,
    )
    prompt = vlm_prompt_templates.render_prompt_template(
        template_text=template_path.read_text(encoding="utf-8"),
        group_a_ids=list(group_a_ids),
        group_b_ids=list(group_b_ids),
        ml_hint_lines=list(ml_hint_lines),
    )
    normalized_extra_instructions = str(extra_instructions or "").strip()
    if normalized_extra_instructions:
        prompt = f"{prompt}\n\nAdditional instructions:\n{normalized_extra_instructions}"
    return prompt, template_path


def build_user_prompt_template(
    window_size: int,
    extra_instructions: str = "",
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    window_schema: str = DEFAULT_WINDOW_SCHEMA,
    group_a_count: Optional[int] = None,
) -> str:
    ml_hint_lines = build_example_ml_hint_lines()
    return build_user_prompt(
        window_size=window_size,
        ml_hint_lines=ml_hint_lines,
        extra_instructions=extra_instructions,
        response_schema_mode=response_schema_mode,
        window_schema=window_schema,
        group_a_count=group_a_count,
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
    rendered_user_prompt: str,
    response_schema: Optional[Mapping[str, Any]],
    response_contract_id: str,
    prompt_template_file: Optional[str] = None,
) -> Dict[str, Any]:
    runs_dir = workspace_dir / RUN_METADATA_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = runs_dir / f"{run_id}.json"
    prompt_template_id = str(args_payload.get("prompt_template_id", "") or "").strip()
    metadata = {
        "run_id": run_id,
        "generated_at": generated_at,
        "config_hash": config_hash,
        "embedded_manifest_csv": str(embedded_manifest_csv),
        "photo_manifest_csv": str(photo_manifest_csv),
        "output_csv": str(output_csv),
        "system_prompt": system_prompt,
        "prompt_template_id": prompt_template_id,
        "prompt_template_file": (
            str(prompt_template_file).strip()
            if prompt_template_file is not None
            else str(args_payload.get("prompt_template_file", "") or "").strip()
        ),
        "rendered_user_prompt": rendered_user_prompt,
        "response_schema": dict(response_schema) if response_schema is not None else None,
        "response_contract_id": response_contract_id,
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
    segment_types_by_pair: Dict[tuple[str, str], Dict[str, List[str]]] = {}
    for result_row in result_rows:
        if str(result_row.get("response_status", "")) != "ok":
            continue
        left_relative_path = str(result_row.get("cut_left_relative_path", "") or "")
        right_relative_path = str(result_row.get("cut_right_relative_path", "") or "")
        if not left_relative_path or not right_relative_path:
            continue
        pair = (left_relative_path, right_relative_path)
        cut_reasons_by_pair.setdefault(pair, []).append(str(result_row.get("reason", "") or "").strip())
        left_segment_type, right_segment_type = extract_segment_types(str(result_row.get("raw_response", "") or ""))
        pair_segment_types = segment_types_by_pair.setdefault(pair, {"left": [], "right": []})
        if left_segment_type:
            pair_segment_types["left"].append(left_segment_type)
        if right_segment_type:
            pair_segment_types["right"].append(right_segment_type)

    segments: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    current_left_boundary_pair: Optional[tuple[str, str]] = None
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
                    "left_boundary_pair": current_left_boundary_pair,
                    "right_boundary_pair": split_pair,
                    "cut_hits": len(reasons),
                    "cut_reasons": reasons,
                }
            )
            current_rows = [row]
            current_left_boundary_pair = split_pair
            continue
        current_rows.append(row)
    if current_rows:
        segments.append(
            {
                "rows": current_rows,
                "left_boundary_pair": current_left_boundary_pair,
                "right_boundary_pair": None,
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
        segment_type_candidates: List[str] = []
        left_boundary_pair = segment.get("left_boundary_pair")
        if isinstance(left_boundary_pair, tuple):
            pair_segment_types = segment_types_by_pair.get(left_boundary_pair, {})
            segment_type_candidates.extend(pair_segment_types.get("right", []))
        right_boundary_pair = segment.get("right_boundary_pair")
        if isinstance(right_boundary_pair, tuple):
            pair_segment_types = segment_types_by_pair.get(right_boundary_pair, {})
            segment_type_candidates.extend(pair_segment_types.get("left", []))
        segment_type = choose_segment_type(segment_type_candidates)
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
                "segment_type": segment_type,
                "type_code": segment_type_to_code(segment_type),
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
        if header != OUTPUT_HEADERS:
            raise ValueError(f"{output_csv.name} has unsupported header; expected current VLM probe columns")
        for row in reader:
            if not row:
                continue
            row_headers = detect_result_row_headers(output_csv.name, header, row)
            row_values = list(row) + [""] * max(0, len(row_headers) - len(row))
            rows.append(normalize_result_row({key: row_values[index] for index, key in enumerate(row_headers)}))
        return rows


def detect_result_row_headers(csv_name: str, file_header: Sequence[str], row: Sequence[str]) -> Sequence[str]:
    header_list = list(file_header)
    row_length = len(row)
    if header_list == OUTPUT_HEADERS and row_length <= len(OUTPUT_HEADERS):
        return OUTPUT_HEADERS
    raise ValueError(f"{csv_name} has unsupported row width {row_length}")


def normalize_result_row(row: Mapping[str, str]) -> Dict[str, str]:
    normalized = {header: "" for header in OUTPUT_HEADERS}
    for header in OUTPUT_HEADERS:
        if header in row:
            normalized[header] = str(row.get(header, ""))
    return normalized


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


def build_run_prompt_snapshot(
    *,
    day_id: str,
    joined_rows: Sequence[Mapping[str, str]],
    candidate: Mapping[str, Any],
    window_radius: int,
    window_schema: str,
    window_schema_seed: int,
    boundary_gap_seconds: int,
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
    ml_hint_context: Optional[MlHintContext],
    prompt_template_id: str,
    prompt_template_file: str,
    extra_instructions: str,
    response_schema_mode: str,
) -> tuple[str, Path, Optional[Dict[str, Any]]]:
    cut_index = int(candidate["cut_index"])
    window_rows, group_a_count = build_window_rows_for_cut_index(
        joined_rows=joined_rows,
        cut_index=cut_index,
        window_radius=window_radius,
        window_schema=window_schema,
        window_schema_seed=window_schema_seed,
        boundary_gap_seconds=boundary_gap_seconds,
    )
    if not window_rows:
        raise ValueError(f"candidate window at cut_index={cut_index} produced no rows")
    ml_hint_lines = build_ml_hint_lines_for_window_rows(
        day_id=day_id,
        window_rows=window_rows,
        boundary_rows_by_pair=boundary_rows_by_pair,
        photo_pre_model_dir=photo_pre_model_dir,
        ml_hint_context=ml_hint_context,
        runtime_window_radius=window_radius,
    )
    group_a_ids, group_b_ids = build_runtime_group_ids(
        group_a_count,
        max(0, len(window_rows) - group_a_count),
    )
    rendered_user_prompt, prompt_template_path = render_runtime_user_prompt(
        prompt_template_id=prompt_template_id,
        prompt_template_file=prompt_template_file,
        group_a_ids=group_a_ids,
        group_b_ids=group_b_ids,
        ml_hint_lines=ml_hint_lines,
        extra_instructions=extra_instructions,
    )
    response_schema = (
        build_response_schema(group_a_ids=group_a_ids, group_b_ids=group_b_ids)
        if response_schema_mode == "on"
        else None
    )
    return rendered_user_prompt, prompt_template_path, response_schema


def run_vlm_window_analysis(
    *,
    window_rows: Sequence[Mapping[str, str]],
    group_a_count: Optional[int] = None,
    ml_hint_lines: Sequence[str],
    window_schema: str = DEFAULT_WINDOW_SCHEMA,
    response_contract_id: str = DEFAULT_RESPONSE_CONTRACT_ID,
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
    keep_alive: str,
    temperature: float,
    context_tokens: Optional[int],
    max_output_tokens: Optional[int],
    reasoning_level: str,
    extra_instructions: str,
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    json_validation_mode: str = DEFAULT_JSON_VALIDATION_MODE,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_file: str = "",
    debug_dir: Optional[Path] = None,
    debug_run_id: str = "",
    debug_batch_index: int = 1,
) -> Dict[str, Any]:
    window_size = len(window_rows)
    resolved_group_a_count = group_a_count if group_a_count is not None else max(1, window_size // 2)
    group_a_ids, group_b_ids = build_runtime_group_ids(
        resolved_group_a_count,
        max(0, window_size - resolved_group_a_count),
    )
    response_schema = (
        build_response_schema(group_a_ids=group_a_ids, group_b_ids=group_b_ids)
        if response_schema_mode == "on"
        else None
    )
    prompt, _ = render_runtime_user_prompt(
        prompt_template_id=prompt_template_id,
        prompt_template_file=prompt_template_file,
        group_a_ids=group_a_ids,
        group_b_ids=group_b_ids,
        ml_hint_lines=ml_hint_lines,
        extra_instructions=extra_instructions,
    )
    request = build_vlm_request(
        provider=provider,
        base_url=base_url,
        response_schema_mode=response_schema_mode,
        prompt=prompt,
        image_paths=[Path(str(row["image_path"])) for row in window_rows],
        model=model,
        timeout_seconds=timeout_seconds,
        keep_alive=keep_alive,
        temperature=temperature,
        context_tokens=context_tokens,
        max_output_tokens=max_output_tokens,
        reasoning_level=reasoning_level,
        response_schema=response_schema,
    )
    request_payload = build_provider_request_payload(request) if debug_dir is not None else None
    try:
        response = run_vlm_request(request)
    except Exception as error:
        if debug_dir is not None:
            dump_debug_artifacts(
                debug_dir=debug_dir,
                run_id=debug_run_id,
                batch_index=debug_batch_index,
                prompt=prompt,
                request_payload=request_payload or {},
                response_payload=None,
                error_text=str(error),
            )
        raise
    if debug_dir is not None:
        dump_debug_artifacts(
            debug_dir=debug_dir,
            run_id=debug_run_id,
            batch_index=debug_batch_index,
            prompt=prompt,
            request_payload=request_payload or {},
            response_payload=response.raw_response,
            error_text=None,
        )
    raw_response = response.text
    parsed_response = parse_model_response(
        raw_response,
        group_a_ids=group_a_ids,
        group_b_ids=group_b_ids,
        response_contract_id=response_contract_id,
        json_validation_mode=json_validation_mode,
    )
    if parsed_response["response_status"] == "invalid_response" and should_retry_structural_response_error(
        parsed_response["reason"]
    ):
        repair_prompt = (
            prompt
            + "\n\nRepair instruction:\n"
            + "Your previous answer violated the JSON structure. Return the same analysis again, but include every frame_id exactly once."
        )
        repair_request = build_vlm_request(
            provider=provider,
            base_url=base_url,
            response_schema_mode=response_schema_mode,
            prompt=repair_prompt,
            image_paths=[Path(str(row["image_path"])) for row in window_rows],
            model=model,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            temperature=temperature,
            context_tokens=context_tokens,
            max_output_tokens=max_output_tokens,
            reasoning_level=reasoning_level,
            response_schema=response_schema,
        )
        request = repair_request
        prompt = repair_prompt
        request_payload = build_provider_request_payload(request) if debug_dir is not None else None
        response = run_vlm_request(repair_request)
        raw_response = response.text
        parsed_response = parse_model_response(
            raw_response,
            group_a_ids=group_a_ids,
            group_b_ids=group_b_ids,
            response_contract_id=response_contract_id,
            json_validation_mode=json_validation_mode,
        )
    if debug_dir is not None and str(parsed_response.get("response_status", "") or "") != "ok":
        dump_debug_artifacts(
            debug_dir=debug_dir,
            run_id=debug_run_id,
            batch_index=debug_batch_index,
            prompt=prompt,
            request_payload=request_payload or {},
            response_payload=response.raw_response,
            error_text=str(parsed_response.get("reason", "") or "invalid VLM response"),
        )
    return {
        "prompt": prompt,
        "request": request,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
    }


def should_retry_structural_response_error(reason: str) -> bool:
    return any(
        reason.startswith(prefix)
        for prefix in (
            "Missing frame_notes array",
            "Invalid frame_notes item",
            "Missing frame_id value in frame_notes",
            "Invalid frame_id value:",
            "duplicate frame_id:",
            "Invalid group for ",
            "Missing frame_notes value for ",
        )
    )


def build_manual_vlm_debug_run_id(anchor_pair: Mapping[str, Any]) -> str:
    payload = {
        "left_relative_path": str(anchor_pair.get("left_relative_path", "") or "").strip(),
        "right_relative_path": str(anchor_pair.get("right_relative_path", "") or "").strip(),
        "left_row_index": int(anchor_pair.get("left_row_index", -1)),
        "right_row_index": int(anchor_pair.get("right_row_index", -1)),
    }
    digest = hashlib.md5(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"manual_{payload['left_row_index']}_{payload['right_row_index']}_{digest}"


def run_manual_vlm_anchor_analysis(
    *,
    day_id: str,
    joined_rows: Sequence[Mapping[str, str]],
    anchor_pair: Mapping[str, Any],
    window_radius: int,
    boundary_rows_by_pair: Mapping[tuple[str, str], Mapping[str, str]],
    photo_pre_model_dir: Optional[Path],
    ml_hint_context: Optional[MlHintContext],
    provider: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
    keep_alive: str,
    temperature: float,
    context_tokens: Optional[int],
    max_output_tokens: Optional[int],
    reasoning_level: str,
    extra_instructions: str = "",
    response_schema_mode: str = DEFAULT_RESPONSE_SCHEMA_MODE,
    json_validation_mode: str = DEFAULT_JSON_VALIDATION_MODE,
) -> Dict[str, Any]:
    window_rows = build_manual_vlm_window_rows(
        joined_rows,
        anchor_pair=anchor_pair,
        window_radius=window_radius,
    )
    ml_hint_lines = build_ml_hint_lines_for_window_rows(
        day_id=day_id,
        window_rows=window_rows,
        boundary_rows_by_pair=boundary_rows_by_pair,
        photo_pre_model_dir=photo_pre_model_dir,
        ml_hint_context=ml_hint_context,
        runtime_window_radius=window_radius,
    )
    analysis = run_vlm_window_analysis(
        window_rows=window_rows,
        group_a_count=window_radius,
        ml_hint_lines=ml_hint_lines,
        window_schema=DEFAULT_WINDOW_SCHEMA,
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        keep_alive=keep_alive,
        temperature=temperature,
        context_tokens=context_tokens,
        max_output_tokens=max_output_tokens,
        reasoning_level=reasoning_level,
        extra_instructions=extra_instructions,
        response_schema_mode=response_schema_mode,
        json_validation_mode=json_validation_mode,
        debug_dir=Path("/tmp"),
        debug_run_id=build_manual_vlm_debug_run_id(anchor_pair),
        debug_batch_index=1,
    )
    analysis["window_rows"] = [dict(row) for row in window_rows]
    analysis["window_relative_paths"] = [str(row["relative_path"]) for row in window_rows]
    analysis["ml_hint_lines"] = list(ml_hint_lines)
    analysis["left_anchor"] = str(anchor_pair.get("left_relative_path", "") or "").strip()
    analysis["right_anchor"] = str(anchor_pair.get("right_relative_path", "") or "").strip()
    return analysis


def probe_vlm_photo_boundaries(
    *,
    day_id: str,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    output_csv: Path,
    provider: str,
    image_variant: str,
    window_radius: int,
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
    window_schema: str = DEFAULT_WINDOW_SCHEMA,
    window_schema_seed: int = DEFAULT_WINDOW_SCHEMA_SEED,
    response_contract_id: str = DEFAULT_RESPONSE_CONTRACT_ID,
) -> int:
    config_hash = build_config_hash(args_payload)
    joined_rows = read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )
    boundary_rows_by_pair = read_boundary_scores_by_pair(workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME)
    ml_hint_context = load_ml_hint_context(
        ml_model_run_id=ml_model_run_id,
        ml_model_dir=ml_model_dir,
    )
    all_candidates = build_candidate_windows(
        joined_rows,
        window_radius=window_radius,
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
        rendered_user_prompt, prompt_template_path, response_schema = build_run_prompt_snapshot(
            day_id=day_id,
            joined_rows=joined_rows,
            candidate=candidate_windows[0],
            window_radius=window_radius,
            window_schema=window_schema,
            window_schema_seed=window_schema_seed,
            boundary_gap_seconds=boundary_gap_seconds,
            boundary_rows_by_pair=boundary_rows_by_pair,
            photo_pre_model_dir=photo_pre_model_dir,
            ml_hint_context=ml_hint_context,
            prompt_template_id=str(args_payload.get("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID)),
            prompt_template_file=str(args_payload.get("prompt_template_file", "") or ""),
            extra_instructions=extra_instructions,
            response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
        )
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
            rendered_user_prompt=rendered_user_prompt,
            response_schema=response_schema,
            response_contract_id=response_contract_id,
            prompt_template_file=str(prompt_template_path),
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
        global_row_by_relative_path = {
            str(row.get("relative_path", "") or "").strip(): index
            for index, row in enumerate(joined_rows, start=1)
        }
        for batch_offset, candidate in enumerate(candidate_windows, start=1):
            batch_index = completed_batches + batch_offset
            cut_index = int(candidate["cut_index"])
            time_gap_seconds = int(candidate["time_gap_seconds"])
            window_rows, group_a_count = build_window_rows_for_cut_index(
                joined_rows=joined_rows,
                cut_index=cut_index,
                window_radius=window_radius,
                window_schema=window_schema,
                window_schema_seed=window_schema_seed,
                boundary_gap_seconds=boundary_gap_seconds,
            )
            if not window_rows:
                raise ValueError(f"candidate window at cut_index={cut_index} produced no rows")
            global_row_numbers = [
                global_row_by_relative_path[str(row.get("relative_path", "") or "").strip()]
                for row in window_rows
            ]
            ml_hint_lines = build_ml_hint_lines_for_window_rows(
                day_id=day_id,
                window_rows=window_rows,
                boundary_rows_by_pair=boundary_rows_by_pair,
                photo_pre_model_dir=photo_pre_model_dir,
                ml_hint_context=ml_hint_context,
                runtime_window_radius=window_radius,
            )
            window_size = len(window_rows)
            analysis = run_vlm_window_analysis(
                window_rows=window_rows,
                group_a_count=group_a_count,
                ml_hint_lines=ml_hint_lines,
                window_schema=window_schema,
                provider=provider,
                base_url=ollama_base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                keep_alive=ollama_keep_alive,
                temperature=temperature,
                context_tokens=ollama_num_ctx,
                max_output_tokens=ollama_num_predict,
                reasoning_level=ollama_think,
                extra_instructions=extra_instructions,
                response_schema_mode=str(args_payload.get("response_schema_mode", "off")),
                json_validation_mode=json_validation_mode,
                response_contract_id=response_contract_id,
                prompt_template_id=str(args_payload.get("prompt_template_id", DEFAULT_PROMPT_TEMPLATE_ID)),
                prompt_template_file=str(args_payload.get("prompt_template_file", "") or ""),
                debug_dir=dump_debug_dir,
                debug_run_id=run_id,
                debug_batch_index=batch_index,
            )
            raw_response = str(analysis["raw_response"])
            parsed_response = analysis["parsed_response"]
            result_row = build_result_row(
                generated_at=generated_at,
                run_id=run_id,
                config_hash=config_hash,
                image_variant=image_variant,
                batch_index=batch_index,
                start_row=min(global_row_numbers),
                end_row=max(global_row_numbers),
                rows=window_rows,
                window_radius=window_radius,
                raw_response=raw_response,
                parsed_response=parsed_response,
                model=model,
                temperature=temperature,
                global_row_numbers=global_row_numbers,
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
                        start_row=min(global_row_numbers),
                        end_row=max(global_row_numbers),
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
        "window_radius": args.window_radius,
        "window_schema": args.window_schema,
        "window_schema_seed": args.window_schema_seed,
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
        "prompt_template_id": args.prompt_template_id,
        "prompt_template_file": args.prompt_template_file,
        "response_contract_id": DEFAULT_RESPONSE_CONTRACT_ID,
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
        window_radius=args.window_radius,
        window_schema=args.window_schema,
        window_schema_seed=args.window_schema_seed,
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
        response_contract_id=str(args_payload["response_contract_id"]),
    )
    console.print(f"Wrote {row_count} VLM probe rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
