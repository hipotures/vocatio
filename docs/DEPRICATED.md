# Deprecated Pipeline Entry Points

The following scripts have been superseded by the canonical unified media exporter:

- `scripts/pipeline/export_event_media_csv.py`
  - Replaced by `scripts/pipeline/export_media.py`
  - The old script exported stream-specific media CSV files for the audio-assisted path.
  - Use `export_media.py` to build the canonical `media_manifest.csv` instead.

- `scripts/pipeline/export_recursive_photo_csv.py`
  - Replaced by `scripts/pipeline/export_media.py`
  - The old script exported a photo-only manifest for the image-only path.
  - Use `export_media.py` to build the canonical `media_manifest.csv` instead.
