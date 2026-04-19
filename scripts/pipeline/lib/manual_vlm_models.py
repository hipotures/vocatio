from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


MODEL_FIELDS = (
    "VLM_NAME",
    "VLM_PROVIDER",
    "VLM_BASE_URL",
    "VLM_MODEL",
    "VLM_CONTEXT_TOKENS",
    "VLM_MAX_OUTPUT_TOKENS",
    "VLM_KEEP_ALIVE",
    "VLM_TIMEOUT_SECONDS",
    "VLM_TEMPERATURE",
    "VLM_REASONING_LEVEL",
    "VLM_RESPONSE_SCHEMA_MODE",
    "VLM_JSON_VALIDATION_MODE",
)

SUPPORTED_PROVIDERS = {"ollama", "llamacpp", "vllm"}


@dataclass(frozen=True)
class ManualVlmModelsConfig:
    models: list[dict[str, Any]]
    md5_hex: str


def compute_manual_vlm_models_md5(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.md5(payload).hexdigest()


def _validate_non_empty_string(value: Any, field_name: str, preset_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f'{field_name} must be a non-empty string in preset "{preset_name}"')
    return normalized


def _validate_positive_int(value: Any, field_name: str, preset_name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{field_name} must be a positive integer in preset "{preset_name}"'
        ) from None
    if normalized <= 0:
        raise ValueError(f'{field_name} must be > 0 in preset "{preset_name}"')
    return normalized


def _validate_positive_float(value: Any, field_name: str, preset_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{field_name} must be a positive number in preset "{preset_name}"'
        ) from None
    if normalized <= 0:
        raise ValueError(f'{field_name} must be > 0 in preset "{preset_name}"')
    return normalized


def _validate_model_entry(model: Any, index: int) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError(f"models[{index}] must be a mapping")
    normalized = dict(model)
    for field in MODEL_FIELDS:
        if field not in normalized:
            raise ValueError(f"missing {field} in preset at index {index}")
    name = str(normalized.get("VLM_NAME", "") or "").strip()
    if not name:
        raise ValueError(f"models[{index}] has empty VLM_NAME")
    provider = str(normalized["VLM_PROVIDER"] or "").strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f'unsupported VLM_PROVIDER "{provider}" in preset "{name}"')

    normalized["VLM_NAME"] = name
    normalized["VLM_PROVIDER"] = provider
    normalized["VLM_BASE_URL"] = _validate_non_empty_string(
        normalized["VLM_BASE_URL"],
        "VLM_BASE_URL",
        name,
    )
    normalized["VLM_MODEL"] = _validate_non_empty_string(
        normalized["VLM_MODEL"],
        "VLM_MODEL",
        name,
    )
    normalized["VLM_CONTEXT_TOKENS"] = _validate_positive_int(
        normalized["VLM_CONTEXT_TOKENS"],
        "VLM_CONTEXT_TOKENS",
        name,
    )
    normalized["VLM_MAX_OUTPUT_TOKENS"] = _validate_positive_int(
        normalized["VLM_MAX_OUTPUT_TOKENS"],
        "VLM_MAX_OUTPUT_TOKENS",
        name,
    )
    normalized["VLM_TIMEOUT_SECONDS"] = _validate_positive_float(
        normalized["VLM_TIMEOUT_SECONDS"],
        "VLM_TIMEOUT_SECONDS",
        name,
    )
    return normalized


def load_manual_vlm_models(path: Path) -> ManualVlmModelsConfig:
    if not path.is_file():
        raise ValueError(f"manual VLM model config does not exist: {path}")
    payload = path.read_bytes()
    raw = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manual VLM model config must be a mapping")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manual VLM model config must define a non-empty models list")

    normalized_models: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, model in enumerate(models):
        normalized = _validate_model_entry(model, index)
        name = str(normalized["VLM_NAME"])
        if name in seen_names:
            raise ValueError(f'duplicate VLM_NAME "{name}"')
        seen_names.add(name)
        normalized_models.append(normalized)

    return ManualVlmModelsConfig(
        models=normalized_models,
        md5_hex=hashlib.md5(payload).hexdigest(),
    )
