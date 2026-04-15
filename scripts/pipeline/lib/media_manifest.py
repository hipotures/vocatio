from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from lib.image_pipeline_contracts import MEDIA_MANIFEST_REQUIRED_COLUMNS, validate_required_columns


def read_media_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, MEDIA_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        return [dict(row) for row in reader]


def select_photo_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "photo"]


def select_video_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "video"]


def group_rows_by_stream_id(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stream_id") or "")].append(dict(row))
    return dict(grouped)
