from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

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


def load_prompt_templates_config(path: Path) -> PromptTemplatesConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list):
        raise ValueError("templates must be a list")

    templates: list[PromptTemplateEntry] = []
    for index, item in enumerate(raw_templates):
        if not isinstance(item, dict):
            raise ValueError(f"templates[{index}] must be a mapping")

        entry_id = str(item.get("id", "") or "").strip()
        label = str(item.get("label", "") or "").strip()
        file_value = str(item.get("file", "") or "").strip()
        description = str(item.get("description", "") or "").strip()

        if not entry_id:
            raise ValueError(f"templates[{index}] is missing id")
        if not label:
            raise ValueError(f'templates[{index}] "{entry_id}" is missing label')
        if not file_value:
            raise ValueError(f'templates[{index}] "{entry_id}" is missing file')
        if not description:
            raise ValueError(f'templates[{index}] "{entry_id}" is missing description')

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
