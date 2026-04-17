from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence


SCHEMA_VERSION = "photo_pre_model_v1"
DEFAULT_OUTPUT_DIRNAME = "photo_pre_model_annotations"

PEOPLE_COUNT_VALUES = ["1", "2", "3", "4plus", "unclear"]
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
PHOTO_PRE_MODEL_DESCRIPTOR_FIELD_REGISTRY = {
    "people_count": "scalar",
    "performer_view": "scalar",
    "upper_garment": "scalar",
    "lower_garment": "scalar",
    "sleeves": "scalar",
    "leg_coverage": "scalar",
    "dominant_colors": "multivalue",
    "headwear": "scalar",
    "footwear": "scalar",
    "props": "multivalue",
    "dance_style_hint": "scalar",
}
PHOTO_PRE_MODEL_MULTIVALUE_FIELDS = frozenset(
    field_name
    for field_name, field_kind in PHOTO_PRE_MODEL_DESCRIPTOR_FIELD_REGISTRY.items()
    if field_kind == "multivalue"
)


def get_photo_pre_model_descriptor_field_registry() -> Dict[str, str]:
    return dict(PHOTO_PRE_MODEL_DESCRIPTOR_FIELD_REGISTRY)


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
        "data": dict(data),
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
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Schema response is not a JSON object")
    return parsed


def validate_annotation_data(result: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(result.keys()))
    if missing:
        raise ValueError(f"Schema response missing required fields: {', '.join(missing)}")


def build_prompt_only_json_prompt() -> str:
    return (
        "Analyze this single stage-event photo and return only valid JSON. "
        "Describe only visible people and costume-related attributes. "
        "Do not infer background, venue, organization, or event names. "
        "Use exactly these keys: "
        "people_count, performer_view, upper_garment, lower_garment, sleeves, leg_coverage, "
        "dominant_colors, headwear, footwear, props, dance_style_hint. "
        f"people_count must be one of: {', '.join(PEOPLE_COUNT_VALUES)}. "
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
            loaded[relative_path] = dict(data)
    return loaded


def load_photo_pre_model_annotations_by_relative_path(
    output_dir: Path,
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    return load_photo_pre_model_data_by_relative_path(output_dir, relative_paths)
