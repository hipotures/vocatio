from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FinalPhotoTruth:
    photo_id: str
    segment_id: str
    segment_type: str


def build_final_photo_truth(rows: Iterable[dict[str, str]]) -> dict[str, FinalPhotoTruth]:
    truth_by_photo_id: dict[str, FinalPhotoTruth] = {}
    for row in rows:
        photo_id = row["photo_id"]
        if photo_id in truth_by_photo_id:
            raise ValueError(f"duplicate photo_id in final photo truth rows: {photo_id}")
        truth_by_photo_id[photo_id] = FinalPhotoTruth(
            photo_id=row["photo_id"],
            segment_id=row["segment_id"],
            segment_type=row["segment_type"],
        )
    return truth_by_photo_id
