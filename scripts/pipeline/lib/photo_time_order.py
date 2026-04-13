from __future__ import annotations

import re
from datetime import datetime
from typing import Mapping, NamedTuple, Optional, Sequence


PHOTO_CAPTURE_TIME_FIELDS: Sequence[tuple[str, str]] = (
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("DateTimeOriginal", "datetime_original"),
    ("SubSecCreateDate", "subsec_create_date"),
    ("CreateDate", "create_date"),
    ("FileModifyDate", "file_modify_date"),
    ("FileCreateDate", "file_create_date"),
)

_EXIF_TIMESTAMP_RE = re.compile(
    r"^(?P<base>\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))?(?:Z|[+-]\d{2}:?\d{2})?$"
)


class CaptureTimeParts(NamedTuple):
    capture_time_local: str
    capture_subsec: str
    timestamp_source: str
    start_local: str
    start_epoch_ms: str


def parse_exif_datetime(value: object) -> Optional[tuple[datetime, str]]:
    if value is None:
        return None
    text = str(value).strip().replace("T", " ")
    match = _EXIF_TIMESTAMP_RE.match(text)
    if match is None:
        return None
    dt = datetime.strptime(match.group("base"), "%Y:%m:%d %H:%M:%S")
    fraction = match.group("fraction") or ""
    if fraction:
        dt = dt.replace(microsecond=int((fraction + "000000")[:6]))
    return dt, fraction


def format_capture_subsec(fraction: str) -> str:
    return fraction if fraction else "0"


def format_capture_time_local(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat(timespec="seconds")


def format_start_local(value: datetime) -> str:
    if value.microsecond == 0:
        return value.isoformat(timespec="seconds")
    if value.microsecond % 1000 == 0:
        return value.isoformat(timespec="milliseconds")
    return value.isoformat(timespec="microseconds")


def format_start_epoch_ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


def pick_capture_time_parts(item: Mapping[str, object]) -> CaptureTimeParts:
    for field_name, source_name in PHOTO_CAPTURE_TIME_FIELDS:
        parsed = parse_exif_datetime(item.get(field_name))
        if parsed is None:
            continue
        dt, fraction = parsed
        return CaptureTimeParts(
            capture_time_local=format_capture_time_local(dt),
            capture_subsec=format_capture_subsec(fraction),
            timestamp_source=source_name,
            start_local=format_start_local(dt),
            start_epoch_ms=format_start_epoch_ms(dt),
        )
    raise ValueError("Could not determine capture time from photo metadata")
