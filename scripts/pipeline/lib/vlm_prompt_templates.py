from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptTemplateEntry:
    id: str
    label: str
    file: str
    description: str


@dataclass(frozen=True)
class PromptTemplatesConfig:
    templates: list[PromptTemplateEntry]
    md5_hex: str


def _validate_required_string_field(item: dict[str, Any], index: int, field_name: str) -> str:
    if field_name not in item:
        raise ValueError(f"templates[{index}] is missing {field_name}")
    value = item[field_name]
    if not isinstance(value, str):
        raise ValueError(f"templates[{index}] {field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"templates[{index}] is missing {field_name}")
    return normalized


def load_prompt_templates_config(path: Path) -> PromptTemplatesConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("prompt template registry root must be a mapping")
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list):
        raise ValueError("templates must be a list")

    templates: list[PromptTemplateEntry] = []
    for index, item in enumerate(raw_templates):
        if not isinstance(item, dict):
            raise ValueError(f"templates[{index}] must be a mapping")

        entry_id = _validate_required_string_field(item, index, "id")
        label = _validate_required_string_field(item, index, "label")
        file_value = _validate_required_string_field(item, index, "file")
        description = _validate_required_string_field(item, index, "description")

        templates.append(
            PromptTemplateEntry(
                id=entry_id,
                label=label,
                file=file_value,
                description=description,
            )
        )

    return PromptTemplatesConfig(
        templates=templates,
        md5_hex=hashlib.md5(path.read_bytes()).hexdigest(),
    )
