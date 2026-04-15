from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1


def _normalize_timestamp(value: object) -> float:
    if value is None:
        raise ValueError("timestamp is required and must not be blank")

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    if not text.strip():
        raise ValueError("timestamp is required and must not be blank")
    try:
        return float(text)
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
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"order_idx must be an integer: {value}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order_idx must be an integer: {value}") from exc


def sort_photo_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            _normalize_timestamp(row["timestamp"]),
            _normalize_order_idx(row.get("order_idx")),
            row["photo_id"],
        ),
    )


def canonical_candidate_id(
    *,
    day_id: str,
    center_left_photo_id: str,
    center_right_photo_id: str,
    candidate_rule_version: str,
) -> str:
    raw = f"{day_id}|{center_left_photo_id}|{center_right_photo_id}|{candidate_rule_version}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]
