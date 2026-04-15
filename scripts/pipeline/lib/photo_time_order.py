from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Mapping, NamedTuple, Optional, Sequence


PHOTO_CAPTURE_TIME_FIELDS: Sequence[tuple[str, str]] = (
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("DateTimeOriginal", "datetime_original"),
    ("SubSecCreateDate", "subsec_create_date"),
    ("CreateDate", "create_date"),
)

_EXIF_TIMESTAMP_RE = re.compile(
    r"^(?P<base>\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))?(?P<offset>Z|[+-]\d{2}:?\d{2})?$"
)


class CaptureTimeParts(NamedTuple):
    capture_time_local: str
    capture_subsec: str
    timestamp_source: str
    start_local: str
    start_epoch_ms: str
    sort_dt: datetime


class ParsedExifDatetime(NamedTuple):
    local_dt: datetime
    fraction: str
    aware_dt: Optional[datetime]


def parse_timezone_offset(value: str) -> timezone:
    if value == "Z":
        return timezone.utc
    normalized = value.replace(":", "")
    sign = 1 if normalized[0] == "+" else -1
    hours = int(normalized[1:3])
    minutes = int(normalized[3:5])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def parse_exif_datetime(value: object) -> Optional[ParsedExifDatetime]:
    if value is None:
        return None
    text = str(value).strip().replace("T", " ")
    match = _EXIF_TIMESTAMP_RE.match(text)
    if match is None:
        return None
    local_dt = datetime.strptime(match.group("base"), "%Y:%m:%d %H:%M:%S")
    fraction = match.group("fraction") or ""
    if fraction:
        local_dt = local_dt.replace(microsecond=int((fraction + "000000")[:6]))
    offset = match.group("offset") or ""
    aware_dt = local_dt.replace(tzinfo=parse_timezone_offset(offset)) if offset else None
    return ParsedExifDatetime(local_dt=local_dt, fraction=fraction, aware_dt=aware_dt)


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


def format_start_epoch_ms(value: datetime, aware_value: Optional[datetime]) -> str:
    timestamp_value = aware_value if aware_value is not None else value.replace(tzinfo=timezone.utc)
    return str(int(timestamp_value.timestamp() * 1000))


def normalize_sort_datetime(value: datetime, aware_value: Optional[datetime]) -> datetime:
    if aware_value is None:
        return value
    return aware_value.astimezone(timezone.utc).replace(tzinfo=None)


def pick_capture_time_parts(item: Mapping[str, object]) -> CaptureTimeParts:
    for field_name, source_name in PHOTO_CAPTURE_TIME_FIELDS:
        parsed = parse_exif_datetime(item.get(field_name))
        if parsed is None:
            continue
        dt = parsed.local_dt
        fraction = parsed.fraction
        return CaptureTimeParts(
            capture_time_local=format_capture_time_local(dt),
            capture_subsec=format_capture_subsec(fraction),
            timestamp_source=source_name,
            start_local=format_start_local(dt),
            start_epoch_ms=format_start_epoch_ms(dt, parsed.aware_dt),
            sort_dt=normalize_sort_datetime(dt, parsed.aware_dt),
        )
    raise ValueError("Could not determine capture time from trusted EXIF metadata")
