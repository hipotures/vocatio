from __future__ import annotations

import json
import re
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence


SCHEMA_VERSION = "photo_pre_model_v1"
DEFAULT_OUTPUT_DIRNAME = "photo_pre_model_annotations"

PEOPLE_COUNT_VALUES = ["no_visible_people", "solo", "duet_trio", "quartet", "small_group", "large_group"]
PERFORMER_VIEW_VALUES = ["solo", "duo", "group", "unclear"]
UPPER_GARMENT_VALUES = ["leotard", "top", "shirt", "jacket", "dress_upper", "unitard_upper", "mixed", "unclear"]
LOWER_GARMENT_VALUES = ["tutu", "skirt", "dress", "pants", "shorts", "unitard", "mixed", "unclear"]
SLEEVES_VALUES = ["none", "short", "long", "mixed", "unclear"]
LEG_COVERAGE_VALUES = ["bare", "short", "long", "mixed", "unclear"]
HEADWEAR_VALUES = ["none", "hat", "headband", "hair_accessory", "mixed", "unclear"]
FOOTWEAR_VALUES = ["barefoot", "ballet_shoes", "dance_shoes", "sneakers", "mixed", "unclear"]
DANCE_STYLE_VALUES = ["ballet", "contemporary", "jazz", "ballroom", "latin", "hiphop", "folk", "tap", "other", "unclear"]
COLOR_VALUES = [
    "black",
    "white",
    "gray",
    "silver",
    "gold",
    "red",
    "pink",
    "purple",
    "lavender",
    "blue",
    "turquoise",
    "green",
    "yellow",
    "orange",
    "brown",
    "beige",
    "multicolor",
    "unclear",
]
REQUIRED_FIELDS = {
    "people_count",
    "performer_view",
    "upper_garment",
    "lower_garment",
    "sleeves",
    "leg_coverage",
    "dominant_colors",
    "headwear",
    "footwear",
    "props",
    "dance_style_hint",
}
MULTIVALUE_FIELDS = {
    "dominant_colors",
    "props",
}


def normalize_annotation_data_key_part(key: object) -> str:
    return str(key).strip().replace(".", "_").replace(" ", "_")


def flatten_annotation_data(
    data: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for raw_key in sorted(data.keys(), key=str):
        key_part = normalize_annotation_data_key_part(raw_key)
        if not key_part:
            continue
        field_name = f"{prefix}_{key_part}" if prefix else key_part
        value = data[raw_key]
        if isinstance(value, Mapping):
            flattened.update(flatten_annotation_data(value, prefix=field_name))
            continue
        flattened[field_name] = value
    return flattened


def build_annotation_output_path(output_dir: Path, relative_path: str) -> Path:
    candidate = PurePosixPath(relative_path.strip())
    if not candidate.parts or candidate.is_absolute():
        raise ValueError(f"Invalid relative_path: {relative_path}")
    normalized_parts = [part for part in candidate.parts if part not in {"", "."}]
    if not normalized_parts or any(part == ".." for part in normalized_parts):
        raise ValueError(f"Invalid relative_path: {relative_path}")
    relative_file = Path(*normalized_parts)
    return output_dir / relative_file.parent / f"{relative_file.name}.json"


def build_annotation_record(
    *,
    relative_path: str,
    model: str,
    data: Mapping[str, Any],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "relative_path": relative_path,
        "generated_at": generated_at
        or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model": model,
        "data": normalize_annotation_data(data),
    }


def parse_annotation_content(content: str) -> Dict[str, Any]:
    text = content.strip()
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
    object_text = text[start : end + 1]
    try:
        parsed = json.loads(object_text)
    except json.JSONDecodeError:
        parsed = json.loads(repair_annotation_json_text(object_text))
    if not isinstance(parsed, dict):
        raise ValueError("Schema response is not a JSON object")
    return normalize_annotation_data(parsed)


def validate_annotation_data(result: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(result.keys()))
    if missing:
        raise ValueError(f"Schema response missing required fields: {', '.join(missing)}")
    canonicalize_people_count(result.get("people_count"))


def canonicalize_people_count(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Unsupported people_count value: {value!r}")
    if isinstance(value, int):
        return _canonicalize_people_count_from_number(value)
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("people_count is empty")
    if normalized in PEOPLE_COUNT_VALUES:
        return normalized
    legacy_aliases = {
        "0": "no_visible_people",
        "1": "solo",
        "2": "duet_trio",
        "3": "duet_trio",
        "4": "quartet",
        "4plus": "small_group",
        "soloist": "solo",
        "single": "solo",
        "pair": "duet_trio",
        "duet": "duet_trio",
        "two": "duet_trio",
        "trio": "duet_trio",
        "triplet": "duet_trio",
        "three": "duet_trio",
        "quartet": "quartet",
        "four": "quartet",
        "none": "no_visible_people",
        "no_people": "no_visible_people",
        "no visible people": "no_visible_people",
        "no_visible_people": "no_visible_people",
    }
    if normalized in legacy_aliases:
        return legacy_aliases[normalized]
    if normalized.endswith("+") and normalized[:-1].isdigit():
        return _canonicalize_people_count_from_number(int(normalized[:-1]) + 1)
    if normalized.isdigit():
        return _canonicalize_people_count_from_number(int(normalized))
    raise ValueError(f"Unsupported people_count value: {value!r}")


def normalize_annotation_data(result: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    normalized["people_count"] = canonicalize_people_count(result.get("people_count"))
    return normalized


def repair_annotation_json_text(object_text: str) -> str:
    pattern = re.compile(r'("people_count"\s*:\s*)([^"\[\{][^,\}\n]*)')

    def replace(match: re.Match[str]) -> str:
        raw_value = match.group(2).strip()
        if not raw_value:
            return match.group(0)
        return f'{match.group(1)}{json.dumps(raw_value)}'

    return pattern.sub(replace, object_text, count=1)


def _canonicalize_people_count_from_number(value: int) -> str:
    if value < 0:
        raise ValueError(f"Unsupported people_count value: {value!r}")
    if value == 0:
        return "no_visible_people"
    if value == 1:
        return "solo"
    if value in {2, 3}:
        return "duet_trio"
    if value == 4:
        return "quartet"
    if value <= 10:
        return "small_group"
    return "large_group"


def build_prompt_only_json_prompt() -> str:
    return (
        "Analyze this single stage-event photo and return only valid JSON. "
        "Describe only visible people and costume-related attributes. "
        "Do not infer background, venue, organization, or event names. "
        "Use exactly these keys: "
        "people_count, performer_view, upper_garment, lower_garment, sleeves, leg_coverage, "
        "dominant_colors, headwear, footwear, props, dance_style_hint. "
        f"people_count must be a JSON string and one of: {', '.join(PEOPLE_COUNT_VALUES)}. "
        "Use people_count=no_visible_people when no person is visible in the frame. "
        f"performer_view must be one of: {', '.join(PERFORMER_VIEW_VALUES)}. "
        f"upper_garment must be one of: {', '.join(UPPER_GARMENT_VALUES)}. "
        f"lower_garment must be one of: {', '.join(LOWER_GARMENT_VALUES)}. "
        f"sleeves must be one of: {', '.join(SLEEVES_VALUES)}. "
        f"leg_coverage must be one of: {', '.join(LEG_COVERAGE_VALUES)}. "
        f"dominant_colors must be an array of 1 to 3 values chosen from: {', '.join(COLOR_VALUES)}. "
        f"headwear must be one of: {', '.join(HEADWEAR_VALUES)}. "
        f"footwear must be one of: {', '.join(FOOTWEAR_VALUES)}. "
        "props must be an array of 1 to 3 short visible object names, or ['none'] if no props are visible. "
        f"dance_style_hint must be one of: {', '.join(DANCE_STYLE_VALUES)}."
    )


def load_photo_pre_model_data_by_relative_path(
    output_dir: Path,
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for relative_path in relative_paths:
        output_path = build_annotation_output_path(output_dir, relative_path)
        if not output_path.exists():
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        data = payload.get("data")
        if isinstance(data, Mapping):
            loaded[relative_path] = normalize_annotation_data(data)
    return loaded


def load_photo_pre_model_annotations_by_relative_path(
    output_dir: Path,
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    return load_photo_pre_model_data_by_relative_path(output_dir, relative_paths)


def build_photo_pre_model_descriptor_field_registry() -> Dict[str, str]:
    return {
        field_name: "multivalue" if field_name in MULTIVALUE_FIELDS else "scalar"
        for field_name in sorted(REQUIRED_FIELDS)
    }


def build_dataset_photo_pre_model_descriptor_field_registry(
    annotation_data_by_relative_path: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    registry = build_photo_pre_model_descriptor_field_registry()
    extra_field_kinds: Dict[str, str] = {}
    for annotation_data in annotation_data_by_relative_path.values():
        flattened = flatten_annotation_data(annotation_data)
        for field_name, value in flattened.items():
            if field_name in registry:
                continue
            if _value_is_multivalue(value):
                extra_field_kinds[field_name] = "multivalue"
                continue
            extra_field_kinds.setdefault(field_name, "scalar")
    for field_name in sorted(extra_field_kinds):
        registry[field_name] = extra_field_kinds[field_name]
    return registry


def _value_is_multivalue(value: object) -> bool:
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return not isinstance(value, MappingABC)
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return False
    for delimiter in (",", ";", "|", "/"):
        parts = [part.strip() for part in normalized.split(delimiter)]
        if len([part for part in parts if part]) >= 2:
            return True
    return False
