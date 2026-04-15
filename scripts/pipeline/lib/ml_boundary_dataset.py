from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import math


def _normalize_timestamp(value: object) -> float:
    if value is None:
        raise ValueError("timestamp is required and must not be blank")

    if isinstance(value, bool):
        raise ValueError("timestamp must be numeric or ISO-8601")

    if isinstance(value, (int, float)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("timestamp must be finite")
        return normalized

    text = str(value)
    if not text.strip():
        raise ValueError("timestamp is required and must not be blank")
    try:
        normalized = float(text)
        if not math.isfinite(normalized):
            raise ValueError("timestamp must be finite")
        return normalized
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"timestamp must be numeric or ISO-8601: {value}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.timestamp()


def _normalize_order_idx(value: object) -> int:
    if value in (None, ""):
        raise ValueError("order_idx is required and must not be blank")
    if isinstance(value, bool):
        raise ValueError(f"order_idx must be an integer: {value}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"order_idx must be an integer: {value}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order_idx must be an integer: {value}") from exc


def _normalize_photo_id(value: object) -> str:
    if value is None or not str(value).strip():
        raise ValueError("photo_id is required and must not be blank")
    return str(value)


def sort_photo_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            _normalize_timestamp(row.get("timestamp")),
            _normalize_order_idx(row.get("order_idx")),
            _normalize_photo_id(row.get("photo_id")),
        ),
    )


def canonical_candidate_id(
    *,
    day_id: str,
    center_left_photo_id: str,
    center_right_photo_id: str,
    candidate_rule_version: str,
) -> str:
    raw = json.dumps(
        [day_id, center_left_photo_id, center_right_photo_id, candidate_rule_version],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:16]
