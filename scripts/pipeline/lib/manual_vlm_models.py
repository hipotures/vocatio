from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scripts.pipeline.lib.vlm_transport import VlmRequest, VlmTransportError, validate_vlm_request


VLM_MODEL_FIELDS = (
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

PREMODEL_MODEL_FIELDS = (
    "PREMODEL_NAME",
    "PREMODEL_PROVIDER",
    "PREMODEL_BASE_URL",
    "PREMODEL_MODEL",
    "PREMODEL_MAX_OUTPUT_TOKENS",
    "PREMODEL_TEMPERATURE",
    "PREMODEL_TIMEOUT_SECONDS",
)

MODEL_FIELDS = VLM_MODEL_FIELDS

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


def _normalize_optional_scalar(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return normalized


def _validate_positive_int(value: Any, field_name: str, preset_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f'{field_name} must be a positive integer in preset "{preset_name}"'
        )
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str):
        try:
            normalized = int(value.strip())
        except ValueError:
            raise ValueError(
                f'{field_name} must be a positive integer in preset "{preset_name}"'
            ) from None
    else:
        raise ValueError(
            f'{field_name} must be a positive integer in preset "{preset_name}"'
        )
    if normalized <= 0:
        raise ValueError(f'{field_name} must be > 0 in preset "{preset_name}"')
    return normalized


def _validate_finite_float(value: Any, field_name: str, preset_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{field_name} must be a finite number in preset "{preset_name}"'
        ) from None
    if not math.isfinite(normalized):
        raise ValueError(
            f'{field_name} must be a finite number in preset "{preset_name}"'
        )
    return normalized


def _validate_positive_float(value: Any, field_name: str, preset_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{field_name} must be a positive number in preset "{preset_name}"'
        ) from None
    if not math.isfinite(normalized):
        raise ValueError(
            f'{field_name} must be a positive number in preset "{preset_name}"'
        )
    if normalized <= 0:
        raise ValueError(f'{field_name} must be > 0 in preset "{preset_name}"')
    return normalized


def _validate_provider(value: Any, field_name: str, preset_name: str) -> str:
    provider = str(value or "").strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f'unsupported {field_name} "{provider}" in preset "{preset_name}"')
    return provider


def _validate_model_entry(model: Any, index: int) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError(f"models[{index}] must be a mapping")
    has_vlm_name = "VLM_NAME" in model
    has_premodel_name = "PREMODEL_NAME" in model
    if has_vlm_name and has_premodel_name:
        raise ValueError(
            f"models[{index}] must define exactly one of VLM_NAME or PREMODEL_NAME"
        )
    if has_premodel_name or any(str(key).startswith("PREMODEL_") for key in model):
        return _validate_premodel_model_entry(dict(model), index)
    return _validate_vlm_model_entry(dict(model), index)


def _validate_vlm_model_entry(normalized: dict[str, Any], index: int) -> dict[str, Any]:
    description = _normalize_optional_scalar(normalized.get("VLM_DESCRIPTION"))
    if description is None:
        normalized.pop("VLM_DESCRIPTION", None)
    else:
        normalized["VLM_DESCRIPTION"] = description
    for field in VLM_MODEL_FIELDS:
        if field not in normalized:
            raise ValueError(f"missing {field} in preset at index {index}")
    name = str(normalized.get("VLM_NAME", "") or "").strip()
    if not name:
        raise ValueError(f"models[{index}] has empty VLM_NAME")

    normalized["VLM_NAME"] = name
    normalized["VLM_PROVIDER"] = _validate_provider(
        normalized["VLM_PROVIDER"],
        "VLM_PROVIDER",
        name,
    )
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
    normalized["VLM_TEMPERATURE"] = _validate_finite_float(
        normalized["VLM_TEMPERATURE"],
        "VLM_TEMPERATURE",
        name,
    )
    keep_alive = _normalize_optional_scalar(normalized["VLM_KEEP_ALIVE"])
    normalized["VLM_KEEP_ALIVE"] = keep_alive or "0"
    normalized["VLM_REASONING_LEVEL"] = _validate_non_empty_string(
        normalized["VLM_REASONING_LEVEL"],
        "VLM_REASONING_LEVEL",
        name,
    )
    normalized["VLM_RESPONSE_SCHEMA_MODE"] = _validate_non_empty_string(
        normalized["VLM_RESPONSE_SCHEMA_MODE"],
        "VLM_RESPONSE_SCHEMA_MODE",
        name,
    )
    normalized["VLM_JSON_VALIDATION_MODE"] = _validate_non_empty_string(
        normalized["VLM_JSON_VALIDATION_MODE"],
        "VLM_JSON_VALIDATION_MODE",
        name,
    )
    return normalized


def _validate_premodel_model_entry(normalized: dict[str, Any], index: int) -> dict[str, Any]:
    for field in PREMODEL_MODEL_FIELDS:
        if field not in normalized:
            raise ValueError(f"missing {field} in preset at index {index}")
    name = str(normalized.get("PREMODEL_NAME", "") or "").strip()
    if not name:
        raise ValueError(f"models[{index}] has empty PREMODEL_NAME")

    normalized["PREMODEL_NAME"] = name
    normalized["PREMODEL_PROVIDER"] = _validate_provider(
        normalized["PREMODEL_PROVIDER"],
        "PREMODEL_PROVIDER",
        name,
    )
    normalized["PREMODEL_BASE_URL"] = _validate_non_empty_string(
        normalized["PREMODEL_BASE_URL"],
        "PREMODEL_BASE_URL",
        name,
    )
    normalized["PREMODEL_MODEL"] = _validate_non_empty_string(
        normalized["PREMODEL_MODEL"],
        "PREMODEL_MODEL",
        name,
    )
    normalized["PREMODEL_MAX_OUTPUT_TOKENS"] = _validate_positive_int(
        normalized["PREMODEL_MAX_OUTPUT_TOKENS"],
        "PREMODEL_MAX_OUTPUT_TOKENS",
        name,
    )
    normalized["PREMODEL_TEMPERATURE"] = _validate_finite_float(
        normalized["PREMODEL_TEMPERATURE"],
        "PREMODEL_TEMPERATURE",
        name,
    )
    normalized["PREMODEL_TIMEOUT_SECONDS"] = _validate_positive_float(
        normalized["PREMODEL_TIMEOUT_SECONDS"],
        "PREMODEL_TIMEOUT_SECONDS",
        name,
    )
    return normalized


def _find_preset_by_exact_name(
    models: Sequence[Mapping[str, Any]],
    name_field: str,
    preset_name: str,
) -> dict[str, Any]:
    for model in models:
        model_name = str(model.get(name_field, "") or "").strip()
        if model_name == preset_name:
            return dict(model)
    raise ValueError(f'unknown {name_field} "{preset_name}"')


def _normalize_resolved_vlm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _validate_vlm_model_entry(dict(config), 0)
    validate_resolved_vlm_provider_config(normalized)
    return normalized


def _normalize_resolved_premodel_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_premodel_model_entry(dict(config), 0)


def _configured_override(
    vocatio_config: Mapping[str, Any],
    field_name: str,
) -> Any | None:
    if field_name not in vocatio_config:
        return None
    value = vocatio_config[field_name]
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped
    return value


def validate_resolved_vlm_provider_config(config: Mapping[str, Any]) -> None:
    request = VlmRequest(
        provider=str(config["VLM_PROVIDER"]),
        base_url=str(config["VLM_BASE_URL"]),
        model=str(config["VLM_MODEL"]),
        messages=[{"role": "user", "content": "config validation"}],
        image_paths=[],
        timeout_seconds=float(config["VLM_TIMEOUT_SECONDS"]),
        temperature=float(config["VLM_TEMPERATURE"]),
        context_tokens=int(config["VLM_CONTEXT_TOKENS"]),
        max_output_tokens=int(config["VLM_MAX_OUTPUT_TOKENS"]),
        reasoning_level=_normalize_optional_scalar(config.get("VLM_REASONING_LEVEL")),
        keep_alive=_normalize_optional_scalar(config.get("VLM_KEEP_ALIVE")),
    )
    try:
        validate_vlm_request(request)
    except VlmTransportError as exc:
        raise ValueError(str(exc)) from exc


def resolve_vlm_model_config(
    models: Sequence[Mapping[str, Any]],
    vocatio_config: Mapping[str, Any],
) -> dict[str, Any]:
    preset_name = str(vocatio_config.get("VLM_NAME", "") or "").strip()
    if not preset_name:
        raise ValueError("missing VLM_NAME in .vocatio")
    resolved = _find_preset_by_exact_name(models, "VLM_NAME", preset_name)
    for field_name in VLM_MODEL_FIELDS:
        if field_name == "VLM_NAME":
            continue
        override = _configured_override(vocatio_config, field_name)
        if override is not None:
            resolved[field_name] = override
    return _normalize_resolved_vlm_config(resolved)


def resolve_premodel_model_config(
    models: Sequence[Mapping[str, Any]],
    vocatio_config: Mapping[str, Any],
) -> dict[str, Any]:
    preset_name = str(vocatio_config.get("PREMODEL_NAME", "") or "").strip()
    if not preset_name:
        raise ValueError("missing PREMODEL_NAME in .vocatio")
    resolved = _find_preset_by_exact_name(models, "PREMODEL_NAME", preset_name)
    for field_name in PREMODEL_MODEL_FIELDS:
        if field_name == "PREMODEL_NAME":
            continue
        override = _configured_override(vocatio_config, field_name)
        if override is not None:
            resolved[field_name] = override
    return _normalize_resolved_premodel_config(resolved)


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
    seen_names: dict[str, set[str]] = {
        "VLM_NAME": set(),
        "PREMODEL_NAME": set(),
    }
    for index, model in enumerate(models):
        normalized = _validate_model_entry(model, index)
        name_field = "VLM_NAME" if "VLM_NAME" in normalized else "PREMODEL_NAME"
        name = str(normalized[name_field])
        if name in seen_names[name_field]:
            raise ValueError(f'duplicate {name_field} "{name}"')
        seen_names[name_field].add(name)
        normalized_models.append(normalized)

    return ManualVlmModelsConfig(
        models=normalized_models,
        md5_hex=hashlib.md5(payload).hexdigest(),
    )
