from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import review_index_loader


VALID_SEGMENT_TYPES = {"performance", "ceremony", "warmup"}


def default_review_state(day: str = "") -> dict[str, Any]:
    return {
        "version": 2,
        "day": day,
        "updated_at": "",
        "performances": {},
        "splits": {},
        "merges": [],
    }


def load_review_state_json(path: Path | str, *, day: str = "") -> dict[str, Any]:
    review_state_path = Path(path)
    if not review_state_path.exists():
        return default_review_state(day)
    try:
        payload = json.loads(review_state_path.read_text(encoding="utf-8"))
    except Exception:
        return default_review_state(day)
    if not isinstance(payload, dict):
        return default_review_state(day)
    payload.setdefault("version", 2)
    payload.setdefault("day", day)
    payload.setdefault("updated_at", "")
    payload.setdefault("performances", {})
    payload.setdefault("splits", {})
    payload.setdefault("merges", [])
    if not isinstance(payload["performances"], dict):
        payload["performances"] = {}
    if not isinstance(payload["splits"], dict):
        payload["splits"] = {}
    if not isinstance(payload["merges"], list):
        payload["merges"] = []
    return payload


def load_review_index_payload_json(path: Path | str, *, day_dir: Path | str | None = None) -> dict[str, Any]:
    return review_index_loader.load_review_index(path, day_dir=day_dir)


def split_specs_for_original(review_state: dict[str, Any], original_set_id: str) -> list[dict[str, Any]]:
    splits = review_state.get("splits", {})
    if not isinstance(splits, dict):
        return []
    specs = splits.get(original_set_id, [])
    if not isinstance(specs, list):
        return []
    return [spec for spec in specs if isinstance(spec, dict)]


def merge_specs(review_state: dict[str, Any]) -> list[dict[str, Any]]:
    merges = review_state.get("merges", [])
    if not isinstance(merges, list):
        return []
    return [spec for spec in merges if isinstance(spec, dict)]


def _display_name_to_segment_type(display_name: str, *, original_performance_number: str, set_id: str) -> str:
    normalized = display_name.strip().lower()
    original_normalized = original_performance_number.strip().lower()
    if normalized == "ceremony":
        return "ceremony"
    if normalized == "warmup":
        return "warmup"
    if normalized == original_normalized:
        return "performance"
    raise ValueError(
        f"Unknown explicit split name for display set {set_id}: {display_name}. "
        "Use the canonical labels ceremony or warmup, or keep the original performance name."
    )


def _stable_segment_id(display_set_id: str) -> str:
    return str(display_set_id).strip()


def migrate_split_state_keys(
    review_index_payload: dict[str, Any],
    review_state: dict[str, Any],
) -> dict[str, Any]:
    splits = review_state.get("splits", {})
    if not isinstance(splits, dict):
        migrated_state = dict(review_state)
        migrated_state["splits"] = {}
        return migrated_state

    performances = [
        performance
        for performance in review_index_payload.get("performances", [])
        if isinstance(performance, dict)
    ]
    valid_base_ids = {
        str(performance.get("set_id", "") or "").strip()
        for performance in performances
        if str(performance.get("set_id", "") or "").strip()
    }

    def base_set_candidates_for_number(performance_number: str) -> list[dict[str, Any]]:
        candidates = [
            performance
            for performance in performances
            if str(performance.get("performance_number", "") or "").strip() == performance_number
        ]
        return sorted(
            candidates,
            key=lambda item: (
                str(item.get("performance_start_local", "") or ""),
                str(item.get("set_id", "") or ""),
            ),
        )

    migrated_splits: dict[str, list[dict[str, Any]]] = {}
    for key, value in splits.items():
        mapped_key = str(key).strip()
        if mapped_key not in valid_base_ids:
            candidates = base_set_candidates_for_number(mapped_key)
            if candidates:
                mapped_key = str(candidates[0].get("set_id", "") or mapped_key).strip()
        target = migrated_splits.setdefault(mapped_key, [])
        if isinstance(value, list):
            target.extend(spec for spec in value if isinstance(spec, dict))

    migrated_state = dict(review_state)
    migrated_state["splits"] = migrated_splits
    return migrated_state


def rebuild_final_display_sets(
    review_index_payload: dict[str, Any],
    review_state: dict[str, Any],
) -> list[dict[str, Any]]:
    review_state = migrate_split_state_keys(review_index_payload, review_state)
    display_sets: list[dict[str, Any]] = []
    for original in review_index_payload.get("performances", []):
        if not isinstance(original, dict):
            continue
        base_set_id = str(original.get("set_id") or original.get("performance_number") or "").strip()
        original_number = str(original.get("performance_number", "") or "").strip()
        photos = [dict(photo) for photo in original.get("photos", []) if isinstance(photo, dict)]
        photo_index = {
            str(photo.get("filename", "")).strip(): index
            for index, photo in enumerate(photos)
            if str(photo.get("filename", "")).strip()
        }

        valid_specs: list[dict[str, Any]] = []
        for spec in split_specs_for_original(review_state, base_set_id):
            start_filename = str(spec.get("start_filename", "") or "").strip()
            if start_filename not in photo_index:
                continue
            valid_specs.append(
                {
                    "start_filename": start_filename,
                    "start_index": photo_index[start_filename],
                    "new_name": str(spec.get("new_name", "") or "").strip(),
                }
            )
        valid_specs.sort(key=lambda spec: int(spec["start_index"]))

        segment_starts = [0] + [int(spec["start_index"]) for spec in valid_specs]
        segment_names = [original_number] + [spec["new_name"] or original_number for spec in valid_specs]
        segment_ids = [base_set_id] + [f"{base_set_id}::{spec['start_filename']}" for spec in valid_specs]

        for index, start_index in enumerate(segment_starts):
            end_index = segment_starts[index + 1] if index + 1 < len(segment_starts) else len(photos)
            segment_photos = [dict(photo) for photo in photos[start_index:end_index]]
            if not segment_photos:
                continue
            display_sets.append(
                {
                    "set_id": segment_ids[index],
                    "display_name": segment_names[index],
                    "original_performance_number": original_number,
                    "photos": segment_photos,
                }
            )

    merged_sets = [dict(display_set) for display_set in display_sets]
    for display_set in merged_sets:
        display_set["photos"] = [dict(photo) for photo in display_set.get("photos", [])]

    for spec in merge_specs(review_state):
        target_set_id = str(spec.get("target_set_id", "") or "").strip()
        source_set_id = str(spec.get("source_set_id", "") or "").strip()
        if not target_set_id or not source_set_id or target_set_id == source_set_id:
            continue
        index_by_set_id = {
            str(display_set.get("set_id", "")).strip(): index
            for index, display_set in enumerate(merged_sets)
        }
        if target_set_id not in index_by_set_id or source_set_id not in index_by_set_id:
            continue
        target_index = index_by_set_id[target_set_id]
        source_index = index_by_set_id[source_set_id]
        target_set = merged_sets[target_index]
        source_set = merged_sets[source_index]
        combined_photos = list(target_set.get("photos", [])) + list(source_set.get("photos", []))
        combined_photos.sort(
            key=lambda photo: (
                str(photo.get("adjusted_start_local", "") or ""),
                str(photo.get("filename", "") or ""),
            )
        )
        target_set["photos"] = combined_photos
        merged_sets.pop(source_index)

    return merged_sets


def flatten_final_display_sets(display_sets: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_photo_ids: set[str] = set()
    for display_set in display_sets:
        display_set_id = _stable_segment_id(str(display_set.get("set_id", "") or ""))
        display_name = str(display_set.get("display_name", "") or "")
        original_performance_number = str(display_set.get("original_performance_number", "") or "")
        segment_type = _display_name_to_segment_type(
            display_name,
            original_performance_number=original_performance_number,
            set_id=display_set_id,
        )
        if segment_type not in VALID_SEGMENT_TYPES:
            raise ValueError(f"Unsupported segment type for display set {display_set_id}: {segment_type}")
        for photo in display_set.get("photos", []):
            photo_id = str(photo.get("photo_id", "") or "").strip()
            if not photo_id:
                raise ValueError(f"photo_id is required for display set {display_set_id}")
            if photo_id in seen_photo_ids:
                raise ValueError(f"duplicate photo_id in final display sets: {photo_id}")
            seen_photo_ids.add(photo_id)
            rows.append(
                {
                    "photo_id": photo_id,
                    "segment_id": display_set_id,
                    "segment_type": segment_type,
                }
            )
    return rows
