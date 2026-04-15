from typing import Iterable


SOURCE_MODE_IMAGE_ONLY_V1 = "image_only_v1"

PHOTO_MANIFEST_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_root",
    "source_dir",
    "source_rel_dir",
    "path",
    "relative_path",
    "photo_id",
    "filename",
    "extension",
    "capture_time_local",
    "capture_subsec",
    "photo_order_index",
    "start_local",
    "start_epoch_ms",
    "timestamp_source",
    "model",
    "make",
    "sequence",
    "actual_size_bytes",
    "create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]

PHOTO_MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "path",
        "relative_path",
        "photo_order_index",
    }
)

MEDIA_MANIFEST_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_root",
    "source_dir",
    "source_rel_dir",
    "path",
    "relative_path",
    "media_id",
    "photo_id",
    "filename",
    "extension",
    "capture_time_local",
    "capture_subsec",
    "photo_order_index",
    "start_local",
    "end_local",
    "start_epoch_ms",
    "end_epoch_ms",
    "duration_seconds",
    "timestamp_source",
    "model",
    "make",
    "sequence",
    "width",
    "height",
    "fps",
    "embedded_size_bytes",
    "actual_size_bytes",
    "metadata_status",
    "metadata_error",
    "create_date_raw",
    "track_create_date_raw",
    "media_create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]

MEDIA_MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "media_type",
        "stream_id",
        "path",
        "relative_path",
        "media_id",
        "start_local",
        "start_epoch_ms",
    }
)

MEDIA_MANIFEST_PHOTO_REQUIRED_COLUMNS = frozenset(
    set(MEDIA_MANIFEST_REQUIRED_COLUMNS)
    | {"photo_id", "capture_time_local", "capture_subsec", "photo_order_index"}
)

MEDIA_MANIFEST_VIDEO_REQUIRED_COLUMNS = frozenset(
    set(MEDIA_MANIFEST_REQUIRED_COLUMNS)
    | {"end_local", "end_epoch_ms", "duration_seconds", "width", "height", "fps"}
)


def validate_required_columns(
    name: str,
    required: Iterable[str],
    actual: Iterable[str] | None,
) -> None:
    required_names = set(required)
    actual_names = set(actual or ())
    missing = sorted(required_names - actual_names)
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")
