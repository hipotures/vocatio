from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .image_pipeline_contracts import (
    MEDIA_MANIFEST_PHOTO_REQUIRED_COLUMNS,
    MEDIA_MANIFEST_REQUIRED_COLUMNS,
    MEDIA_MANIFEST_VIDEO_REQUIRED_COLUMNS,
    validate_required_columns,
)


def validate_required_values(
    name: str,
    required: Iterable[str],
    row: Dict[str, str],
) -> None:
    missing = sorted(column for column in required if str(row.get(column) or "").strip() == "")
    if missing:
        raise ValueError(f"{name} missing required values: {', '.join(missing)}")


def read_media_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, MEDIA_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows: List[Dict[str, str]] = []
        photo_headers_validated = False
        video_headers_validated = False
        for row_number, row in enumerate(reader, start=2):
            row_dict = dict(row)
            media_type = str(row_dict.get("media_type") or "").strip()
            required_columns = MEDIA_MANIFEST_REQUIRED_COLUMNS
            if media_type == "photo":
                required_columns = MEDIA_MANIFEST_PHOTO_REQUIRED_COLUMNS
                if not photo_headers_validated:
                    validate_required_columns(path.name, required_columns, reader.fieldnames)
                    photo_headers_validated = True
            elif media_type == "video":
                required_columns = MEDIA_MANIFEST_VIDEO_REQUIRED_COLUMNS
                if not video_headers_validated:
                    validate_required_columns(path.name, required_columns, reader.fieldnames)
                    video_headers_validated = True
            else:
                raise ValueError(
                    f"{path.name} row {row_number} has invalid media_type: {media_type or '<empty>'}"
                )
            validate_required_values(f"{path.name} row {row_number}", required_columns, row_dict)
            rows.append(row_dict)
        if not rows:
            raise ValueError(f"{path.name} is empty")
        return rows


def select_photo_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "photo"]


def select_video_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "video"]


def group_rows_by_stream_id(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stream_id") or "")].append(dict(row))
    return dict(grouped)
