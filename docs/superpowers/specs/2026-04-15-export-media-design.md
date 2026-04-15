# Export Media Unification Design

## Goal

Replace the current split entrypoint model:

- `scripts/pipeline/export_event_media_csv.py`
- `scripts/pipeline/export_recursive_photo_csv.py`

with one universal exporter:

- `scripts/pipeline/export_media.py`

The new exporter must write exactly one canonical input artifact:

- `DAY/_workspace/media_manifest.csv`

This manifest must contain the full field coverage needed by both existing pipelines:

- audio-assisted pipeline
- image-only pipeline

The design goal is to minimize downstream changes by preserving existing field names wherever possible.

## Why This Change

Today the repository has two overlapping metadata exporters:

- `export_event_media_csv.py`
  - writes per-stream CSV files for both photos and video
  - is used by the audio-assisted pipeline
- `export_recursive_photo_csv.py`
  - writes a single logical photo manifest
  - is used by the image-only pipeline

They duplicate EXIF work, use different output shapes, and force later steps to depend on exporter-specific contracts.

The new design creates one canonical manifest and shifts downstream adaptation to lightweight readers and filters instead of duplicate export stages.

## Current Field Coverage

### Current photo rows from `export_event_media_csv.py`

- `day`
- `stream_id`
- `device`
- `media_type`
- `source_dir`
- `path`
- `filename`
- `extension`
- `start_local`
- `start_epoch_ms`
- `timestamp_source`
- `model`
- `make`
- `sequence`
- `embedded_size_bytes`
- `actual_size_bytes`
- `metadata_status`
- `metadata_error`
- `create_date_raw`
- `datetime_original_raw`
- `subsec_datetime_original_raw`
- `subsec_create_date_raw`
- `file_modify_date_raw`
- `file_create_date_raw`

### Current video rows from `export_event_media_csv.py`

- `day`
- `stream_id`
- `device`
- `media_type`
- `source_dir`
- `path`
- `filename`
- `extension`
- `start_local`
- `end_local`
- `start_epoch_ms`
- `end_epoch_ms`
- `duration_seconds`
- `timestamp_source`
- `model`
- `make`
- `width`
- `height`
- `fps`
- `embedded_size_bytes`
- `actual_size_bytes`
- `create_date_raw`
- `track_create_date_raw`
- `media_create_date_raw`
- `datetime_original_raw`
- `subsec_datetime_original_raw`
- `file_modify_date_raw`
- `file_create_date_raw`

### Current photo rows from `export_recursive_photo_csv.py`

- `day`
- `stream_id`
- `device`
- `media_type`
- `source_root`
- `source_dir`
- `source_rel_dir`
- `path`
- `relative_path`
- `photo_id`
- `filename`
- `extension`
- `capture_time_local`
- `capture_subsec`
- `photo_order_index`
- `start_local`
- `start_epoch_ms`
- `timestamp_source`
- `model`
- `make`
- `sequence`
- `actual_size_bytes`
- `create_date_raw`
- `datetime_original_raw`
- `subsec_datetime_original_raw`
- `subsec_create_date_raw`
- `file_modify_date_raw`
- `file_create_date_raw`

## Design Decision

The new exporter will produce one CSV schema that is a superset of both existing photo contracts plus the current video contract.

This is an additive compatibility design:

- keep current field names when they already exist and are used downstream
- add missing fields instead of renaming old ones
- allow empty values for fields that do not apply to a given `media_type`

This avoids a broad downstream rename migration and lets later steps consume filtered subsets of the same canonical file.

## Canonical Output

### File

- `DAY/_workspace/media_manifest.csv`

### Canonical headers

- `day`
- `stream_id`
- `device`
- `media_type`
- `source_root`
- `source_dir`
- `source_rel_dir`
- `path`
- `relative_path`
- `media_id`
- `photo_id`
- `filename`
- `extension`
- `capture_time_local`
- `capture_subsec`
- `photo_order_index`
- `start_local`
- `end_local`
- `start_epoch_ms`
- `end_epoch_ms`
- `duration_seconds`
- `timestamp_source`
- `model`
- `make`
- `sequence`
- `width`
- `height`
- `fps`
- `embedded_size_bytes`
- `actual_size_bytes`
- `create_date_raw`
- `track_create_date_raw`
- `media_create_date_raw`
- `datetime_original_raw`
- `subsec_datetime_original_raw`
- `subsec_create_date_raw`
- `file_modify_date_raw`
- `file_create_date_raw`

## Field Semantics

### Shared identity and location fields

- `path`
  - absolute source path, matching current behavior
- `relative_path`
  - path relative to `DAY`
  - required for both photos and video
  - becomes the canonical durable ID source
- `media_id`
  - canonical media identifier for all media
  - always equal to `relative_path`
- `photo_id`
  - compatibility field for existing image-only code
  - equal to `relative_path` for photos
  - empty for videos

### Photo-only fields

- `capture_time_local`
  - canonical chosen photo capture time
- `capture_subsec`
  - canonical photo subsecond string
- `photo_order_index`
  - deterministic global order across all photos in the logical photo stream

### Shared temporal fields

- `start_local`
  - for photos: the chosen capture start time
  - for video: clip start time
- `start_epoch_ms`
  - epoch representation of `start_local`

### Video-only temporal fields

- `end_local`
- `end_epoch_ms`
- `duration_seconds`

### Video-only format fields

- `width`
- `height`
- `fps`

### Size and metadata fields

- `embedded_size_bytes`
  - filled where current filename-derived parsing provides it
  - otherwise empty
- `actual_size_bytes`
  - actual file size
- `metadata_status`
  - per-file extraction status
  - expected values:
    - `ok`
    - `partial`
    - `error`
- `metadata_error`
  - short machine-readable or human-readable extraction error summary
  - empty when `metadata_status=ok`
- raw metadata fields stay unchanged:
  - `create_date_raw`
  - `track_create_date_raw`
  - `media_create_date_raw`
  - `datetime_original_raw`
  - `subsec_datetime_original_raw`
  - `subsec_create_date_raw`
  - `file_modify_date_raw`
  - `file_create_date_raw`

## Exporter Behavior

### CLI surface

`export_media.py` should expose one unified CLI instead of separate photo-specific and video-specific entrypoints.

Required argument:

- `day_dir`

Recommended options:

- `--workspace-dir`
- `--output`
  - default: `media_manifest.csv`
- `--targets`
  - optional explicit stream filtering such as `p-a7r5 v-gh7`
- `--list-targets`
- `--media-types {all,photo,video}`
  - default: `all`
  - this is only an input filter for debugging and staged migration
  - it must not change the schema of the output file
- `--jobs`
  - default: `4`
  - one global concurrency limit for metadata extraction batches across both photos and video

Rejected CLI additions for the first implementation:

- separate `--photo-jobs` and `--video-jobs`
- multiple output schema modes
- a single explicit `--stream-id` mode

### Discovery

`export_media.py` scans the day directory for:

- photo streams under `p-*`
- video streams under `v-*`

It should preserve existing stream naming behavior and derive:

- `stream_id`
- `device`
- `media_type`

### Photos

For photos, the exporter must preserve the current image-only ordering guarantees:

1. chosen capture datetime
2. capture subsecond
3. `relative_path`

It must produce:

- deterministic `photo_order_index`
- `relative_path`
- `source_rel_dir`
- `capture_time_local`
- `capture_subsec`

If EXIF data is missing or partially unreadable, the exporter must still emit a photo row:

- use the best available fallback timestamp logic
- keep the row in the manifest
- set `metadata_status` to `partial` or `error`
- populate `metadata_error`

### Videos

For videos, the exporter must preserve the current normalized per-stream video fields used by the audio-assisted pipeline:

- start/end times
- duration
- width/height/fps
- raw QuickTime date fields

If video metadata is missing or partially unreadable, the exporter must still emit a video row:

- keep the row in the manifest
- fill any available metadata fields
- leave unavailable fields empty
- set `metadata_status` to `partial` or `error`
- populate `metadata_error`

### Atomic write behavior

The manifest write must remain atomic:

- write to a sibling temp file
- validate output rows
- replace final output with `os.replace`

### Failure policy

The exporter must be fail-open at the per-file level and must not abort the whole day because one file has broken metadata.

Rules:

- do not silently skip a supported media file
- do not stop the entire export because one file has unreadable EXIF or container metadata
- always emit one manifest row for every supported file that was discovered
- represent metadata extraction problems through:
  - `metadata_status`
  - `metadata_error`

Only unsupported file extensions should be excluded during discovery.

### Progress UX

The exporter must use `rich.progress` and show both elapsed time and ETA during long-running metadata extraction.

The progress layout should follow the repository standard:

- `SpinnerColumn()`
- `TextColumn("[progress.description]{task.description}")`
- `BarColumn(bar_width=40)`
- `MofNCompleteColumn()`
- `TaskProgressColumn()`
- `TimeElapsedColumn()`

Additionally, because this exporter is expected to process large day directories, it should also include:

- `TimeRemainingColumn()`

Task descriptions should remain fixed-width so the progress bar does not shift horizontally.

## Reader Layer

The migration should not make every consumer parse `media_manifest.csv` directly.

Add a shared helper module:

- `scripts/pipeline/lib/media_manifest.py`

Initial functions:

- `read_media_manifest(path)`
- `select_photo_rows(rows)`
- `select_video_rows(rows)`
- `group_rows_by_stream_id(rows)`

This isolates the new contract and keeps downstream updates small and explicit.

## Downstream Migration Impact

### Image-only pipeline

Expected impact is low.

The image-only pipeline already depends on a small photo-only subset:

- `relative_path`
- `path`
- `photo_order_index`
- `start_local`
- `start_epoch_ms`

These fields are preserved with the same names, so most image-only steps should only need:

- a new loader entrypoint
- filtering `media_type=photo`

No broad schema rename is expected for image-only consumers.

### Audio-assisted pipeline

Expected impact is moderate at the input layer and low beyond that.

The main structural change is that the audio-assisted pipeline currently expects:

- per-stream CSV files from `export_event_media_csv.py`
- then `merge_event_media_csv.py`

Under the new model:

- `export_media.py` directly produces the already-normalized canonical manifest
- `merge_event_media_csv.py` becomes redundant or a thin compatibility wrapper

Later audio-assisted stages mostly already use fields that remain unchanged:

- `stream_id`
- `media_type`
- `start_local`
- `start_epoch_ms`
- `duration_seconds`
- `width`
- `height`
- `fps`

So the main migration effort should stay near the front of the audio-assisted pipeline.

## Migration Strategy

### Phase 1

Implement:

- `export_media.py`
- `lib/media_manifest.py`
- tests for canonical manifest generation

Do not remove existing exporters yet.

### Phase 2

Adapt image-only readers to consume:

- `media_manifest.csv` filtered to `media_type=photo`

This is the safest first migration because the schema overlap is already strong.

### Phase 3

Adapt the audio-assisted pipeline entry path:

- replace or wrap `merge_event_media_csv.py`
- move early audio/video consumers to `media_manifest.csv`

### Phase 4

Once both pipelines are confirmed stable:

- deprecate
  - `export_event_media_csv.py`
  - `export_recursive_photo_csv.py`

## Testing

### Exporter tests

Add tests for:

- mixed photo and video discovery in one day
- deterministic `relative_path` for both media types
- stable `photo_order_index` for photos
- expected empty fields for media-type-specific columns
- atomic output replacement
- compatibility of `timestamp_source`

### Contract tests

Add tests that verify:

- all required image-only fields exist for photo rows
- all required audio-assisted fields exist for video rows
- the canonical schema is stable and ordered

### Migration tests

Add tests that prove:

- image-only consumers can derive their required photo subset from `media_manifest.csv`
- audio-assisted consumers can derive their required video subset from `media_manifest.csv`

## Rejected Alternatives

### Brand-new media-neutral field naming

Rejected for now because it would force widespread downstream renaming with little immediate benefit.

### Single scanner that still writes multiple CSV outputs

Rejected because it does not solve the contract split. The goal is one canonical manifest, not one implementation with multiple divergent outputs.

## Recommendation

Proceed with:

- one canonical `media_manifest.csv`
- additive superset schema
- shared reader helpers
- image-only migration first
- audio-assisted migration second

This gives the smallest-risk path toward one export contract without forcing a large immediate rewrite of both pipelines.
