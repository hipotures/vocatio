from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FinalPhotoTruth:
    photo_id: str
    segment_id: str
    segment_type: str


def build_final_photo_truth(rows: Iterable[dict[str, str]]) -> dict[str, FinalPhotoTruth]:
    return {
        row["photo_id"]: FinalPhotoTruth(
            photo_id=row["photo_id"],
            segment_id=row["segment_id"],
            segment_type=row["segment_type"],
        )
        for row in rows
    }
