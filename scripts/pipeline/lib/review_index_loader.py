import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1

REQUIRED_TOP_LEVEL_FIELDS = (
    "day",
    "workspace_dir",
    "performance_count",
    "photo_count",
    "performances",
)

REQUIRED_IMAGE_ONLY_PHOTO_FIELDS = (
    "relative_path",
    "source_path",
    "proxy_path",
)


def validate_required_fields(name: str, required: Iterable[str], payload: Mapping[str, Any]) -> None:
    missing = sorted(field for field in required if field not in payload)
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(missing)}")


def load_review_index(index_path: Path | str) -> dict[str, Any]:
    index_file_path = Path(index_path).expanduser()
    if not index_file_path.is_absolute():
        cwd_path = Path.cwd()
        pwd_value = os.environ.get("PWD", "")
        base_dir = cwd_path
        if pwd_value:
            pwd_path = Path(pwd_value).expanduser()
            if pwd_path.is_absolute() and pwd_path.resolve() == cwd_path.resolve():
                base_dir = pwd_path
        index_file_path = base_dir / index_file_path
    payload = json.loads(index_file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Review index payload must be a JSON object: {index_file_path}")
    validate_required_fields("review index payload", REQUIRED_TOP_LEVEL_FIELDS, payload)

    source_mode = str(payload.get("source_mode", "") or "").strip()
    day_dir, workspace_dir = resolve_day_and_workspace_dir(payload, index_file_path, source_mode)
    performances = payload["performances"]
    if not isinstance(performances, list):
        raise ValueError("review index payload field 'performances' must be a list")

    if source_mode and source_mode != SOURCE_MODE_IMAGE_ONLY_V1:
        raise ValueError(f"Unsupported review index source_mode: {source_mode}")
    normalized_performances = [
        normalize_performance(
            performance,
            index=index,
            day_dir=day_dir,
            workspace_dir=workspace_dir,
            source_mode=source_mode,
        )
        for index, performance in enumerate(performances)
    ]

    performance_count = normalize_count("performance_count", payload["performance_count"])
    photo_count = normalize_count("photo_count", payload["photo_count"])
    actual_photo_count = sum(len(performance["photos"]) for performance in normalized_performances)
    if performance_count != len(normalized_performances):
        raise ValueError(
            "review index payload performance_count does not match performances length: "
            f"{performance_count} != {len(normalized_performances)}"
        )
    if photo_count != actual_photo_count:
        raise ValueError(
            "review index payload photo_count does not match normalized photo total: "
            f"{photo_count} != {actual_photo_count}"
        )

    day = str(payload["day"]).strip()
    if not day:
        raise ValueError("review index payload field 'day' must not be empty")

    normalized_payload = dict(payload)
    normalized_payload["day"] = day
    normalized_payload["workspace_dir"] = str(workspace_dir)
    normalized_payload["performance_count"] = performance_count
    normalized_payload["photo_count"] = photo_count
    normalized_payload["performances"] = normalized_performances
    return normalized_payload


def resolve_day_and_workspace_dir(payload: Mapping[str, Any], index_path: Path, source_mode: str) -> tuple[Path, Path]:
    day = str(payload.get("day", "") or "").strip()
    if not day:
        raise ValueError("review index payload field 'day' must not be empty")
    workspace_value = str(payload.get("workspace_dir", "") or "").strip()
    if not workspace_value:
        raise ValueError("review index payload field 'workspace_dir' must not be empty")
    declared_workspace_dir = Path(workspace_value)
    if not declared_workspace_dir.is_absolute():
        relative_workspace = normalize_relative_workspace_dir(workspace_value)
        declared_workspace_dir = index_path.parent / relative_workspace
    if source_mode == SOURCE_MODE_IMAGE_ONLY_V1:
        declared_day_dir = declared_workspace_dir.parent
        if declared_day_dir.name != day:
            raise ValueError(
                "review index payload day/workspace_dir mismatch: "
                f"day={day} workspace_dir={declared_workspace_dir}"
            )
        return declared_day_dir.resolve(), declared_workspace_dir.resolve()

    if index_path.parent.name == "_workspace" and index_path.parent.parent.name == day:
        return index_path.parent.parent.resolve(), declared_workspace_dir.resolve()
    if declared_workspace_dir.parent.name == day:
        return declared_workspace_dir.parent.resolve(), declared_workspace_dir.resolve()
    return index_path.parent.resolve(), declared_workspace_dir.resolve()


def normalize_relative_workspace_dir(value: str) -> Path:
    if value == ".":
        return Path(".")
    candidate = PurePosixPath(value)
    if not candidate.parts:
        raise ValueError("review index payload field 'workspace_dir' must not be empty")
    if candidate.is_absolute():
        raise ValueError(f"review index payload relative workspace_dir must stay under the index directory: {value}")
    normalized_parts: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"review index payload relative workspace_dir must stay under the index directory: {value}")
        normalized_parts.append(part)
    if not normalized_parts:
        return Path(".")
    return Path(*normalized_parts)


def normalize_count(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"review index payload field '{name}' must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"review index payload field '{name}' must be an integer")


def normalize_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def normalize_performance(
    performance: Any,
    *,
    index: int,
    day_dir: Path,
    workspace_dir: Path,
    source_mode: str,
) -> dict[str, Any]:
    if not isinstance(performance, dict):
        raise ValueError(f"performance at index {index} must be an object")
    photos = performance.get("photos")
    if not isinstance(photos, list):
        raise ValueError(f"performance at index {index} field 'photos' must be a list")

    performance_number = str(performance.get("performance_number", "") or "").strip()
    if not performance_number:
        raise ValueError(f"performance at index {index} missing required field: performance_number")

    normalized_photos = [
        normalize_photo(
            photo,
            performance_number=performance_number,
            index=photo_index,
            day_dir=day_dir,
            workspace_dir=workspace_dir,
            source_mode=source_mode,
        )
        for photo_index, photo in enumerate(photos)
    ]
    review_count = sum(1 for photo in normalized_photos if photo["assignment_status"] == "review")
    first_proxy_path = next((photo["proxy_path"] for photo in normalized_photos if photo["proxy_exists"]), "")
    last_proxy_path = ""
    for photo in normalized_photos:
        if photo["proxy_exists"]:
            last_proxy_path = photo["proxy_path"]
    first_source_path = normalized_photos[0]["source_path"] if normalized_photos else ""
    last_source_path = normalized_photos[-1]["source_path"] if normalized_photos else ""
    first_photo_local = normalized_photos[0]["adjusted_start_local"] if normalized_photos else ""
    last_photo_local = normalized_photos[-1]["adjusted_start_local"] if normalized_photos else ""
    set_id = str(performance.get("set_id", "") or performance_number).strip()

    normalized_performance = dict(performance)
    normalized_performance["set_id"] = set_id
    normalized_performance["base_set_id"] = str(performance.get("base_set_id", "") or set_id).strip()
    normalized_performance["display_name"] = str(performance.get("display_name", "") or performance_number).strip()
    normalized_performance["original_performance_number"] = (
        str(performance.get("original_performance_number", "") or performance_number).strip()
    )
    normalized_performance["performance_number"] = performance_number
    normalized_performance["occurrence_index"] = str(performance.get("occurrence_index", "") or "").strip()
    normalized_performance["duplicate_status"] = str(performance.get("duplicate_status", "") or "normal").strip()
    normalized_performance["timeline_status"] = str(performance.get("timeline_status", "") or "").strip()
    normalized_performance["performance_start_local"] = str(performance.get("performance_start_local", "") or "").strip()
    normalized_performance["performance_end_local"] = str(performance.get("performance_end_local", "") or "").strip()
    normalized_performance["photo_count"] = len(normalized_photos)
    normalized_performance["review_count"] = review_count
    normalized_performance["first_photo_local"] = first_photo_local
    normalized_performance["last_photo_local"] = last_photo_local
    normalized_performance["first_proxy_path"] = first_proxy_path
    normalized_performance["last_proxy_path"] = last_proxy_path
    normalized_performance["first_source_path"] = first_source_path
    normalized_performance["last_source_path"] = last_source_path
    normalized_performance["photos"] = normalized_photos
    return normalized_performance


def normalize_photo(
    photo: Any,
    *,
    performance_number: str,
    index: int,
    day_dir: Path,
    workspace_dir: Path,
    source_mode: str,
) -> dict[str, Any]:
    if not isinstance(photo, dict):
        raise ValueError(f"photo at performance {performance_number} index {index} must be an object")
    if source_mode == SOURCE_MODE_IMAGE_ONLY_V1:
        validate_required_fields(
            f"image-only photo at performance {performance_number} index {index}",
            REQUIRED_IMAGE_ONLY_PHOTO_FIELDS,
            photo,
        )
        relative_path = normalize_relative_path(str(photo["relative_path"]).strip(), "relative_path", day_dir)
        source_relative = normalize_relative_path(str(photo["source_path"]).strip(), "source_path", day_dir)
        if source_relative != relative_path:
            raise ValueError(
                "image-only photo source_path must match relative_path: "
                f"{source_relative.as_posix()} != {relative_path.as_posix()}"
            )
        proxy_relative = normalize_relative_path(str(photo["proxy_path"]).strip(), "proxy_path", workspace_dir)
        source_path = resolve_runtime_path(day_dir, source_relative, "source_path")
        proxy_path = resolve_runtime_path(workspace_dir, proxy_relative, "proxy_path")
        photo_id = relative_path.as_posix()
        photo_start_local = str(photo.get("photo_start_local", "") or "").strip()
        normalized_photo = dict(photo)
        normalized_photo["photo_id"] = photo_id
        normalized_photo["filename"] = str(photo.get("filename", "") or relative_path.name).strip()
        normalized_photo["relative_path"] = photo_id
        normalized_photo["source_path"] = str(source_path)
        normalized_photo["proxy_path"] = str(proxy_path)
        normalized_photo["proxy_exists"] = (
            normalize_bool("proxy_exists", photo.get("proxy_exists")) if "proxy_exists" in photo else proxy_path.exists()
        )
        normalized_photo["photo_start_local"] = photo_start_local
        normalized_photo["adjusted_start_local"] = photo_start_local
        normalized_photo["assignment_status"] = str(photo.get("assignment_status", "") or "").strip()
        normalized_photo["assignment_reason"] = str(photo.get("assignment_reason", "") or "").strip()
        normalized_photo["seconds_to_nearest_boundary"] = str(photo.get("seconds_to_nearest_boundary", "") or "").strip()
        normalized_photo["stream_id"] = str(photo.get("stream_id", "") or "").strip()
        normalized_photo["device"] = str(photo.get("device", "") or "").strip()
        return normalized_photo

    source_path = str(photo.get("source_path", "") or "").strip()
    proxy_path = str(photo.get("proxy_path", "") or "").strip()
    filename = str(photo.get("filename", "") or Path(source_path).name).strip()
    adjusted_start_local = str(photo.get("adjusted_start_local", "") or photo.get("photo_start_local", "") or "").strip()
    normalized_photo = dict(photo)
    normalized_photo["photo_id"] = str(photo.get("photo_id", "") or source_path or filename).strip()
    normalized_photo["filename"] = filename
    normalized_photo["source_path"] = source_path
    normalized_photo["proxy_path"] = proxy_path
    normalized_photo["proxy_exists"] = (
        normalize_bool("proxy_exists", photo.get("proxy_exists")) if "proxy_exists" in photo else False
    )
    normalized_photo["photo_start_local"] = str(photo.get("photo_start_local", "") or "").strip()
    normalized_photo["adjusted_start_local"] = adjusted_start_local
    normalized_photo["assignment_status"] = str(photo.get("assignment_status", "") or "").strip()
    normalized_photo["assignment_reason"] = str(photo.get("assignment_reason", "") or "").strip()
    normalized_photo["seconds_to_nearest_boundary"] = str(photo.get("seconds_to_nearest_boundary", "") or "").strip()
    normalized_photo["stream_id"] = str(photo.get("stream_id", "") or "").strip()
    normalized_photo["device"] = str(photo.get("device", "") or "").strip()
    return normalized_photo


def normalize_relative_path(value: str, field_name: str, root_dir: Path) -> Path:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative under {root_dir}: {value}")
    resolved = (root_dir / path).resolve()
    try:
        resolved.relative_to(root_dir)
    except ValueError as error:
        raise ValueError(f"{field_name} must stay under {root_dir}: {value}") from error
    return Path(resolved.relative_to(root_dir))


def resolve_runtime_path(root_dir: Path, relative_path: Path, field_name: str) -> Path:
    resolved = (root_dir / relative_path).resolve()
    try:
        resolved.relative_to(root_dir)
    except ValueError as error:
        raise ValueError(f"{field_name} must stay under {root_dir}: {relative_path.as_posix()}") from error
    return resolved
