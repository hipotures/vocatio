# Image-Only Stage 1 Design

## Purpose

Design the first implementation stage of an image-only segmentation pipeline for Vocatio that starts from a day directory and ends at a GUI-ready review index:

```text
DAY -> DAY/_workspace/performance_proxy_index.image.json
```

This stage must coexist with the current audio-first workflow in `scripts/pipeline/` without changing its behavior.

## Scope

In scope for this stage:

- recursive photo discovery under a day directory
- embedded thumbnail and preview extraction into `_workspace`
- quality annotations for photos
- DINOv2 preview embeddings
- boundary feature generation for adjacent photo pairs
- heuristic bootstrap boundary scoring
- predicted segment generation
- GUI-ready image-only review index generation
- GUI support for loading both the existing audio-first index and the new image-only index

Out of scope for this stage:

- reviewed index materialization
- supervised boundary label export
- reusable boundary model training
- downstream asset export compatibility outside the GUI
- any change in current audio-first pipeline behavior

## Design Decisions

Chosen architecture: dual-mode index with a GUI normalization layer.

Rationale:

- the image-only pipeline should not be forced into the exact semantics of the current audio-first JSON
- the GUI should remain the compatibility boundary and load both payload variants through one loader
- each pipeline step should remain independently runnable and testable through explicit on-disk contracts

## Hard Constraints

- All generated artifacts live under `DAY/_workspace` unless an explicit external model path is used in a future stage.
- Recursive discovery must ignore `_workspace` completely.
- The pipeline must never auto-delete, deduplicate, prune, or suppress photos.
- Removing `_workspace` must allow a full deterministic rebuild from source photos.
- Canonical photo order must match the best recoverable shot order from source metadata. This order is a hard correctness requirement because all boundary records are derived from adjacent photos.
- DINOv2 is required in stage 1. Missing model or runtime prerequisites are hard failures for the embedding step.
- The current audio-first pipeline remains behaviorally unchanged.

## Pipeline Boundary

Stage 1 consists of these scripts:

1. `export_recursive_photo_csv.py`
2. `extract_embedded_photo_jpg.py`
3. `build_photo_quality_annotations.py`
4. `embed_photo_previews_dinov2.py`
5. `build_photo_boundary_features.py`
6. `bootstrap_photo_boundaries.py`
7. `build_photo_segments.py`
8. `build_photo_review_index.py`

The GUI is not part of the segmentation pipeline itself. It consumes the final review index through a shared loader.

## Data Contracts

The design uses three contract layers.

### Manifest Layer

Artifacts:

- `photo_manifest.csv`
- `photo_embedded_manifest.csv`
- `photo_quality.csv`
- `features/dinov2_index.csv`

Contract rules:

- `relative_path` is the primary photo key across all manifest-like artifacts
- `relative_path` is also the durable application-level photo identifier for stage 1
- `relative_path` is always relative to `DAY` and written in POSIX form
- row ordering is deterministic and follows `photo_manifest.csv`
- canonical sort order is a hard contract:
  - primary: normalized capture timestamp in chronological order
  - secondary: normalized capture subseconds where available
  - tertiary: `relative_path`
- `photo_manifest.csv` should materialize this canonical ordering as `photo_order_index` so downstream steps can preserve adjacency without re-sorting from scratch
- if full EXIF timestamp precision is unavailable, the script must apply a documented fallback policy and still emit a deterministic total ordering
- downstream artifacts must preserve manifest order rather than recomputing their own independent ordering
- scripts validate required input columns before doing work
- `stream_id` in the image-only pipeline is a logical day-level photo stream identifier, typically one value for the whole photo branch unless explicitly overridden, and must not be inferred from hour subdirectories or other recursive container folders

### Boundary Layer

Artifacts:

- `photo_boundary_features.csv`
- `photo_boundary_scores.csv`

Contract rules:

- each row describes the potential cut between adjacent photos `(i, i+1)`
- the record key is `left_relative_path` + `right_relative_path`
- these artifacts contain no GUI-specific assumptions
- neighboring-photo similarity is a feature only and never a deletion rule

### Segment And Review Layer

Artifacts:

- `photo_segments.csv`
- `performance_proxy_index.image.json`

Contract rules:

- `photo_segments.csv` is the canonical tabular output of segmentation
- `performance_proxy_index.image.json` is a GUI adapter built from segments and manifests
- synthetic identifiers are stable and deterministic:
  - `performance_number`: `SEG0001`, `SEG0002`, ...
  - `set_id`: `imgset-000001`, `imgset-000002`, ...
- `photo_segments.csv` must expose a minimum operational schema:
  - `set_id`
  - `performance_number`
  - `segment_index`
  - `start_relative_path`
  - `end_relative_path`
  - `start_local`
  - `end_local`
  - `photo_count`
  - `segment_confidence`

## Script Responsibilities

Each script owns exactly one step.

### `export_recursive_photo_csv.py`

- recursively discovers supported photo files
- ignores `_workspace`
- extracts EXIF metadata in deterministic order
- writes `photo_manifest.csv`

### `extract_embedded_photo_jpg.py`

- extracts or generates `thumb` and `preview` JPEG derivatives
- mirrors the source directory tree under `_workspace/embedded_jpg`
- writes `photo_embedded_manifest.csv`

### `build_photo_quality_annotations.py`

- computes technical quality metrics and flags
- writes `photo_quality.csv`
- never excludes or hides photos

### `embed_photo_previews_dinov2.py`

- computes one DINOv2 embedding per preview JPEG
- preserves manifest row order
- writes `features/dinov2_embeddings.npy` and `features/dinov2_index.csv`

### `build_photo_boundary_features.py`

- computes features for each adjacent photo pair
- uses time, embedding distance, rolling statistics, and quality deltas
- writes `photo_boundary_features.csv`

### `bootstrap_photo_boundaries.py`

- computes first-pass boundary scores using a deterministic heuristic
- writes `photo_boundary_scores.csv`
- stage 1 supports only the heuristic bootstrap mode as the required path

### `build_photo_segments.py`

- converts boundary scores into chronological segments
- applies minimum segment-size safeguards
- writes `photo_segments.csv`

### `build_photo_review_index.py`

- builds the image-only review payload for the GUI
- writes `performance_proxy_index.image.json`
- depends only on image-only artifacts and not on announcements or audio outputs

## CLI Rules For New Scripts

Every new script in this stage must:

- accept positional `day_dir`
- accept `--workspace-dir`
- accept `--overwrite` where rewriting artifacts is meaningful
- fail early if prerequisites are missing
- validate required columns in upstream CSV inputs
- use explicit input/output paths derived from `workspace_dir`
- avoid hidden state outside `_workspace`

Where a script processes multiple files, it should use the repository's standard `rich.progress` layout.

## GUI Compatibility Strategy

The GUI should support both index variants through a dedicated shared loader, for example:

```text
scripts/pipeline/lib/review_index_loader.py
```

The loader is responsible for:

- reading the JSON payload
- recognizing the payload variant
- validating the minimum supported schema
- normalizing the payload to one shared GUI input model that remains close to the current raw-performance shape

The loader should not materialize final `display_sets`. Existing GUI logic should continue to build `display_sets` from normalized input payload data so that split/merge behavior changes stay narrowly scoped.

## Payload Strategy

The image-only payload may add explicit variant metadata, for example:

- `source_mode: "image_only_v1"`

The audio-first payload may remain unchanged or later gain a corresponding mode field.

Shared top-level fields expected by the loader:

- `day`
- `workspace_dir`
- `performance_count`
- `photo_count`
- `performances`

Optional variant-specific fields:

- `proxy_root`
- `assignments_csv`
- `timeline_csv`
- `announcements_csv`
- `source_mode`
- `source_artifacts`
- `build_metadata`

Path policy for the image-only payload on disk:

- `source_path` should be stored relative to `DAY`
- `proxy_path` should be stored relative to `workspace_dir`
- the loader resolves these to absolute paths for GUI runtime use
- normalized GUI input may contain resolved absolute paths even if the serialized payload stores relative ones

## Minimum Normalized GUI Input

After loading, the GUI should be able to work with a normalized raw-input structure that provides these performance-level fields:

- `set_id`
- `base_set_id`
- `display_name`
- `original_performance_number`
- `duplicate_status`
- `timeline_status`
- `performance_start_local`
- `performance_end_local`
- `photo_count`
- `review_count`
- `first_photo_local`
- `last_photo_local`
- `first_proxy_path`
- `last_proxy_path`
- `first_source_path`
- `last_source_path`
- `photos`

And these photo-level fields:

- `photo_id`
- `filename`
- `source_path`
- `proxy_path`
- `proxy_exists`
- `photo_start_local`
- `adjusted_start_local`
- `assignment_status`
- `assignment_reason`
- `seconds_to_nearest_boundary`
- `stream_id`
- `device`

For stage 1 image-only payloads:

- `photo_id == relative_path`
- `adjusted_start_local == photo_start_local`
- `proxy_path` points to the extracted preview JPEG
- `assignment_status` marks review priority around uncertain or low-confidence boundaries
- split and review references must use the durable photo identifier rather than basename-only filenames

## Boundary And Segmentation Strategy

Stage 1 segmentation is intentionally deterministic and bootstrap-only.

Boundary features should include at least:

- `time_gap_seconds`
- `dino_cosine_distance`
- `rolling_dino_distance_mean`
- `rolling_dino_distance_std`
- `distance_zscore`
- left/right quality flags
- brightness and contrast deltas

Bootstrap scoring should:

- use embedding-distance change as the main signal
- strengthen likely cuts with time-gap evidence
- smooth isolated spikes to reduce noisy cuts
- enforce minimum segment-size constraints in photos and seconds

`build_photo_segments.py` converts scored boundaries into stable synthetic sets for review. It does not attempt to infer real performance numbers.

## Error Handling

Scripts should fail early and clearly for:

- missing required input files
- missing required CSV columns
- empty manifests
- missing preview images for the embedding step
- missing DINOv2 runtime or model prerequisites
- row-count mismatches between manifest-derived and feature-derived artifacts

Stage 1 should not use silent fallbacks or partial-success behavior for required steps.

## Testing Strategy

Testing is organized at three levels.

### Level 1: Single-Script Tests

Use a small fixture and validate one artifact at a time.

Examples:

- discovery ignores `_workspace`
- preview extraction mirrors source directory structure
- embedding index matches manifest row count and order
- boundary features contain `N-1` rows for `N` photos

### Level 2: Short-Chain Integration Tests

Run a reduced chain:

- manifest
- preview extraction
- embedding
- boundary features
- bootstrap boundaries
- segments

This catches integration breakage without requiring GUI interaction.

### Level 2.5: Loader Contract Tests

Validate the compatibility boundary directly:

- audio-first payload in -> normalized GUI input out
- image-only payload in -> normalized GUI input out
- missing required fields -> clear validation error
- relative path payload fields -> correctly resolved runtime paths

### Level 3: GUI Smoke Test

Run a very small day fixture through:

- `performance_proxy_index.image.json`
- GUI load via `--index performance_proxy_index.image.json`
- split and merge actions
- review-state save

## Acceptance Criteria

Stage 1 is complete when:

- each pipeline step is independently runnable on explicit inputs
- deleting `_workspace` allows a full rebuild
- no photo is automatically dropped, deduplicated, or hidden
- `performance_proxy_index.image.json` is produced from image-only artifacts
- `review_performance_proxy_gui.py` can open both the existing audio-first index and the new image-only index
- the existing audio-first workflow remains unchanged in behavior

## Follow-On Work

Later stages can build on this design by adding:

- shared review-state materialization helpers
- reviewed index output
- supervised boundary-label export
- reusable boundary model training and prediction
