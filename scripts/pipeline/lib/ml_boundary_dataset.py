from __future__ import annotations

from hashlib import sha1


def _normalize_order_idx(value: object) -> int:
    if value in (None, ""):
        raise ValueError("order_idx is required and must not be blank")
    return int(value)


def sort_photo_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            row["timestamp"],
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
