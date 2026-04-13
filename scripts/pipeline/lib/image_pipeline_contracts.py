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


def validate_required_columns(name: str, required: set[str], actual: set[str]) -> None:
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")
