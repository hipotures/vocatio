from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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


REQUIRED_TEMPLATE_PLACEHOLDERS = (
    "{{GROUP_MAPPING}}",
    "{{ML_HINTS_BLOCK}}",
    "{{FRAME_NOTES_JSON_EXAMPLE}}",
)


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


def build_group_mapping_lines(group_a_ids: list[str], group_b_ids: list[str]) -> list[str]:
    ordered_ids = [*group_a_ids, *group_b_ids]
    return [
        f"{frame_id} = attached image {index}"
        for index, frame_id in enumerate(ordered_ids, start=1)
    ]


def build_frame_notes_json_example(group_a_ids: list[str], group_b_ids: list[str]) -> str:
    records = [
        {"frame_id": frame_id, "group": "group_a", "note": "<short note>"}
        for frame_id in group_a_ids
    ] + [
        {"frame_id": frame_id, "group": "group_b", "note": "<short note>"}
        for frame_id in group_b_ids
    ]
    return json.dumps(records, ensure_ascii=True, indent=2)


def render_prompt_template(
    *,
    template_text: str,
    group_a_ids: list[str],
    group_b_ids: list[str],
    ml_hint_lines: list[str],
) -> str:
    replacements = {
        "{{WINDOW_SIZE}}": str(len(group_a_ids) + len(group_b_ids)),
        "{{WINDOW_RADIUS}}": str(len(group_a_ids)),
        "{{GROUP_A_COUNT}}": str(len(group_a_ids)),
        "{{GROUP_B_COUNT}}": str(len(group_b_ids)),
        "{{GROUP_A_IDS}}": ", ".join(group_a_ids),
        "{{GROUP_B_IDS}}": ", ".join(group_b_ids),
        "{{GROUP_MAPPING}}": "\n".join(build_group_mapping_lines(group_a_ids, group_b_ids)),
        "{{ML_HINTS_BLOCK}}": "\n".join(ml_hint_lines)
        if ml_hint_lines
        else "ML hints are unavailable for this pair of groups.",
        "{{FRAME_NOTES_JSON_EXAMPLE}}": build_frame_notes_json_example(group_a_ids, group_b_ids),
        "{{SEGMENT_TYPES_INLINE}}": "dance|ceremony|audience|rehearsal|other",
    }
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


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
