from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

VALID_SEGMENT_TYPES = {"performance", "ceremony", "warmup"}


@dataclass(frozen=True)
class FinalPhotoTruth:
    photo_id: str
    segment_id: str
    segment_type: str


def _require_non_blank_field(row: dict[str, str], field_name: str) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    return str(value)


def build_final_photo_truth(rows: Iterable[dict[str, str]]) -> dict[str, FinalPhotoTruth]:
    truth_by_photo_id: dict[str, FinalPhotoTruth] = {}
    for row in rows:
        photo_id = _require_non_blank_field(row, "photo_id")
        segment_id = _require_non_blank_field(row, "segment_id")
        segment_type = _require_non_blank_field(row, "segment_type")
        if segment_type not in VALID_SEGMENT_TYPES:
            raise ValueError(
                "segment_type must be one of: performance, ceremony, warmup"
            )
        if photo_id in truth_by_photo_id:
            raise ValueError(f"duplicate photo_id in final photo truth rows: {photo_id}")
        truth_by_photo_id[photo_id] = FinalPhotoTruth(
            photo_id=photo_id,
            segment_id=segment_id,
            segment_type=segment_type,
        )
    return truth_by_photo_id
