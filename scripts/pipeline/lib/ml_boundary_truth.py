from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

VALID_SEGMENT_TYPES = {"performance", "ceremony", "warmup"}


@dataclass(frozen=True)
class FinalPhotoTruth:
    photo_id: str
    segment_id: str
    segment_type: str


@dataclass(frozen=True)
class BoundaryTruthRow:
    left_segment_type: str
    right_segment_type: str
    boundary: bool


def _require_non_blank_field(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    return str(value).strip()


def _parse_boundary_flag(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError("boundary must be one of: 1, 0, true, false, yes, no")


def _validate_segment_type(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized not in VALID_SEGMENT_TYPES:
        raise ValueError(
            f"{field_name} must be one of: performance, ceremony, warmup"
        )
    return normalized


def load_truth_row(row: Mapping[str, object]) -> BoundaryTruthRow:
    left_segment_type = _validate_segment_type(
        _require_non_blank_field(row, "left_segment_type"),
        field_name="left_segment_type",
    )
    right_segment_type = _validate_segment_type(
        _require_non_blank_field(row, "right_segment_type"),
        field_name="right_segment_type",
    )
    boundary = _parse_boundary_flag(_require_non_blank_field(row, "boundary"))
    return BoundaryTruthRow(
        left_segment_type=left_segment_type,
        right_segment_type=right_segment_type,
        boundary=boundary,
    )


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
