# Vocatio

`vocatio` is a CLI event-media workflow with two parallel per-day pipelines:

- an audio-assisted pipeline that uses synced video and transcription to build performance boundaries
- an image-only pipeline that works from photos alone when usable audio is unavailable or unreliable

Both pipelines write artifacts into `DAY/_workspace` and feed the same review GUI.

## Repository Layout

- `scripts/pipeline/` - operational pipeline scripts
- `conf/` - export profiles for reviewed set delivery
- `docs/PROJECT_INTENT.md` - product direction

## Pipeline Modes

| Mode | Use When | Main Inputs | Main Review Index |
| --- | --- | --- | --- |
| Audio-assisted | you have usable video audio and want the strongest boundary detection | `v-*` streams, optional `p-*` streams | `performance_proxy_index.json` |
| Image-only | you only have photos, or audio/video timing is unusable | `p-*` streams | `performance_proxy_index.image.json` or `performance_proxy_index.image.vlm.json` |

## Requirements

Common requirements:

- Python 3.10+
- `ffmpeg` and `ffprobe`
- `exiftool`
- `PyYAML`, `rich`, `PySide6`
- ImageMagick (`magick`) for JPG proxy and embedded JPG extraction paths

Audio-assisted extras:

- `whisperx`

Image-only extras:

- `numpy`
- local DINOv2 runtime for preview embeddings

Optional image-only VLM extras:

- Ollama with a vision-capable model for `probe_vlm_photo_boundaries.py`
- a local OpenAI-compatible `llama.cpp` server if you want per-photo pre-model annotations

Install Python dependencies in your preferred environment, then run scripts directly with `python3`.

## Data Model (Per Day)

The pipeline runs per `day_dir`, for example:

```bash
/data/20260323
```

If you process multiple days, run the same sequence for each directory.

By default, generated artifacts are written to:

```bash
DAY/_workspace
```

Workspace resolution precedence is:

1. `--workspace-dir`
2. `DAY/.vocatio` with `WORKSPACE_DIR=...`
3. `DAY/_workspace`

Example `DAY/.vocatio`:

```bash
WORKSPACE_DIR=/fast/local/vocatio/20250324
```

## Unified Media Export

Use `export_media.py` to build the canonical single-manifest export for a day:

```bash
python3 scripts/pipeline/export_media.py DAY
```

By default this writes:

```bash
DAY/_workspace/media_manifest.csv
```

The exporter uses a resumable checkpoint file during interrupted or in-progress work:

```bash
DAY/_workspace/media_manifest.csv.partial
```

Behavior:

- resume is the default
- if `.partial` exists, the exporter resumes from it
- if `.partial` does not exist but `media_manifest.csv` exists, the exporter reconstructs `.partial` from the final manifest and resumes
- `--restart` ignores both files and rebuilds from scratch
- after each completed metadata batch, the exporter updates both `.partial` and `media_manifest.csv` using atomic replace
- after a clean successful finish, `.partial` is removed

Operational notes:

- `Ctrl+C` is safe
- the first interrupt stops scheduling new work, lets active batch(es) finish, writes the latest snapshot, and exits
- press `Ctrl+C` again only if you want to abort immediately
- `--jobs` remains safe with resume enabled because state is only promoted from the main thread after completed batches

Useful options:

- `--list-targets` to inspect detected `p-*` and `v-*` streams without writing output
- `--media-types photo` and `--media-types video` for staged debugging and migration checks
- `--restart` to discard existing export state and rebuild from zero

## Expected Day Directory Structure

Example under `/data/DAY/`:

- `/data/20260323/p-a7r5/`
- `/data/20260323/v-a7r5/`
- `/data/20260323/v-gh7/`
- `/data/20260323/v-pocket3/`
- `/data/20260323/_workspace/`

Prefix meaning:

- `v-...` = video stream
- `p-...` = photo stream
- `_workspace/` = generated pipeline artifacts (CSV, JSON, transcripts, proxies, review state)

The pipeline discovers stream directories using these prefixes.

## Audio-Assisted Pipeline

This path uses video sync, transcription, and announcement extraction to build performance boundaries, then assigns photos to those intervals.

### Main Workspace Outputs

- `merged_video.csv`
- `sync_map.csv`
- `sync_diagnostics.csv`
- `merged_video_synced.csv`
- `transcripts_manifest.csv`
- `announcement_candidates.csv` or `announcement_candidates_semantic.csv`
- `performance_timeline.csv`
- `photo_assignments.csv`
- `photo_review.csv`
- `photo_unassigned.csv`
- `photo_assignment_summary.csv`
- `photo_proxy_manifest.csv`
- `performance_proxy_index.json`
- `review_state.json`
- `selected_photos_*.json` optional GUI photo selection exports

Directories:

- `transcripts/`
- `proxy_jpg/`

### Recommended Execution Order

Replace `DAY` with your day directory path.

#### 1. Export the canonical unified media manifest

```bash
python3 scripts/pipeline/export_media.py DAY
```

This writes `DAY/_workspace/media_manifest.csv` and replaces the old stream-specific media export step.

#### 2. Estimate sync map between video streams

```bash
python3 scripts/pipeline/estimate_video_sync_map.py DAY
```

#### 3. Apply sync corrections

```bash
python3 scripts/pipeline/apply_video_sync_map.py DAY
```

#### 4. Transcribe synced videos

```bash
python3 scripts/pipeline/transcribe_video_batch.py DAY --all-streams
```

#### 5A. Extract announcement candidates (rule-based)

```bash
python3 scripts/pipeline/extract_announcement_candidates.py DAY --all-streams
```

#### 5B. Extract announcement candidates (semantic, optional)

```bash
python3 scripts/pipeline/extract_announcement_candidates_semantic.py DAY --all-streams
```

If you use semantic output, pass it explicitly in the next step:

```bash
python3 scripts/pipeline/build_performance_timeline.py DAY --candidates-csv DAY/_workspace/announcement_candidates_semantic.csv
```

#### 6. Build performance timeline

```bash
python3 scripts/pipeline/build_performance_timeline.py DAY
```

#### 7. Assign photos to timeline intervals

```bash
python3 scripts/pipeline/assign_photos_to_timeline.py DAY
```

#### 8. Generate proxy JPG files

```bash
python3 scripts/pipeline/generate_photo_proxy_jpg.py DAY --all-streams
```

#### 9. Build per-performance proxy index

```bash
python3 scripts/pipeline/build_performance_proxy_index.py DAY
```

#### 10. Review assignments in GUI

```bash
python3 scripts/pipeline/review_performance_proxy_gui.py DAY
```

This step creates or updates `review_state.json`.

The GUI also supports exporting an ad-hoc photo selection:

- select one or more photo rows inside a set
- press `Ctrl+E`
- enter a JSON filename such as `selected_photos_a.json`
- the file is written to `DAY/_workspace` unless you enter an absolute path

Example GUI views:

![Single preview mode](assets/gui-review-single-view.png)

![First/Last comparison mode](assets/gui-review-first-last-compare.png)

#### 11. Export one reviewed set

```bash
python3 scripts/pipeline/copy_reviewed_set_assets.py DAY out 158 --config conf/copy_reviewed_set_assets.default.yaml
```

Where:

- `out` is the target root directory
- `158` is the final set name (number or text label)

You can also export only explicitly selected photos from a GUI selection JSON:

```bash
python3 scripts/pipeline/copy_reviewed_set_assets.py DAY out selected_photos_a.json --streams photo
```

In this mode:

- the third argument is a selection JSON saved from the GUI
- the export goes to `out/selected_photos_a/`
- only the listed photos are exported
- `--index-json` still works as usual when you need a non-default index

## Image-Only Pipeline

This path builds photo ordering, quality signals, DINOv2 embeddings, and photo boundary candidates directly from images. It has two review flows:

- a deterministic heuristic flow
- an optional VLM-assisted flow

### Core Image-Only Outputs

- `media_manifest.csv`
- `photo_manifest.csv`
- `photo_embedded_manifest.csv`
- `photo_quality.csv`
- `photo_boundary_features.csv`
- `photo_boundary_scores.csv`
- `photo_segments.csv`
- `performance_proxy_index.image.json`

Directories:

- `embedded_jpg/thumb/`
- `embedded_jpg/preview/`
- `features/`

### Image-Only Deterministic Flow

#### 1. Export the canonical unified media manifest

```bash
python3 scripts/pipeline/export_media.py DAY
```

This writes `DAY/_workspace/media_manifest.csv` and is the canonical exporter for image-only work as well.

#### 2. Extract embedded JPG variants

```bash
python3 scripts/pipeline/extract_embedded_photo_jpg.py DAY
```

This reads photo rows from `DAY/_workspace/media_manifest.csv` and creates `photo_embedded_manifest.csv` with `thumb_path` and `preview_path`.

#### 3. Build photo quality annotations

```bash
python3 scripts/pipeline/build_photo_quality_annotations.py DAY
```

#### 4. Embed preview JPGs with DINOv2

```bash
python3 scripts/pipeline/embed_photo_previews_dinov2.py DAY
```

This creates:

- `features/dinov2_embeddings.npy`
- `features/dinov2_index.csv`

#### 5. Build pairwise photo boundary features

```bash
python3 scripts/pipeline/build_photo_boundary_features.py DAY
```

#### 6. Bootstrap image-only boundary scores

```bash
python3 scripts/pipeline/bootstrap_photo_boundaries.py DAY
```

#### 7. Build heuristic photo segments

```bash
python3 scripts/pipeline/build_photo_segments.py DAY
```

#### 8. Build GUI review index

```bash
python3 scripts/pipeline/build_photo_review_index.py DAY
```

#### 9. Review heuristic image-only sets

```bash
python3 scripts/pipeline/review_performance_proxy_gui.py DAY --index performance_proxy_index.image.json --state review_state.image.json
```

Use a dedicated `--state` file for image-only review so you do not mix it with the audio-assisted review state.

### Image-Only VLM-Assisted Flow

The VLM flow starts from the same image-only artifacts as the deterministic flow. It optionally adds a lightweight per-photo pre-model pass, then probes only candidate time gaps with a local VLM.

#### Optional: build per-photo pre-model annotations

```bash
python3 scripts/pipeline/build_photo_pre_model_annotations.py DAY --limit 1000 --workers 4
```

Default behavior:

- input index: `photo_embedded_manifest.csv`
- image column: `preview_path`
- output directory: `DAY/_workspace/photo_pre_model_annotations`
- resume behavior: existing per-photo JSON files are skipped unless you pass `--overwrite`
- default `--limit` is `20`, so use a larger explicit value for real runs and rerun the same command until it finishes the remaining photos

#### Probe candidate image boundaries with a local VLM

```bash
python3 scripts/pipeline/probe_vlm_photo_boundaries.py DAY \
  --photo-manifest-csv DAY/_workspace/photo_manifest.csv \
  --image-variant thumb \
  --window-size 7 \
  --overlap 2 \
  --boundary-gap-seconds 10 \
  --max-batches 100 \
  --response-schema-mode on \
  --json-validation-mode strict \
  --photo-pre-model-dir photo_pre_model_annotations \
  --new-run
```

Important behavior:

- only gaps larger than `--boundary-gap-seconds` are probed
- default `--max-batches` is `10`, so use a larger explicit value for real runs and rerun the same command or continue with `--run-id`
- `--new-run` starts a fresh VLM run
- `--run-id ...` resumes a specific existing VLM run
- `--photo-pre-model-dir ...` is optional; if per-photo annotations exist, they are appended to the prompt frame-by-frame

This creates:

- `vlm_boundary_test.csv`
- `vlm_runs/`

Optional debugging:

```bash
python3 scripts/pipeline/probe_vlm_photo_boundaries.py DAY \
  --photo-manifest-csv DAY/_workspace/photo_manifest.csv \
  --image-variant thumb \
  --window-size 7 \
  --overlap 2 \
  --boundary-gap-seconds 10 \
  --max-batches 100 \
  --dump-debug-dir /tmp \
  --new-run
```

#### Build a GUI index for a specific VLM run

```bash
python3 scripts/pipeline/build_vlm_photo_boundary_gui_index.py DAY --run-id YOUR_RUN_ID
```

This creates:

- `performance_proxy_index.image.vlm.json`

#### Review VLM-assisted image-only sets

```bash
python3 scripts/pipeline/review_performance_proxy_gui.py DAY \
  --index performance_proxy_index.image.vlm.json \
  --state review_state.image.vlm.json
```

Use a dedicated VLM review-state file so you do not mix VLM review decisions with the heuristic image-only or audio-assisted states.

## Export Profiles

Available profiles:

- `conf/copy_reviewed_set_assets.default.yaml`
  - photo: converted JPG (max edge 3200, quality 90)
  - video: converted MP4 (H.264/AAC), default end padding 10s
- `conf/copy_reviewed_set_assets.raw.yaml`
  - photo/video copied as raw sources

Use with:

```bash
python3 scripts/pipeline/copy_reviewed_set_assets.py DAY out SET_NAME --config conf/copy_reviewed_set_assets.raw.yaml
```

## Semantic Tooling (Optional)

Additional semantic and benchmark scripts:

- `build_semantic_announcement_demo.py`
- `demo_semantic_announcement_classifier.py`
- `benchmark_semantic_announcement_models.py`

Use these for model experiments and benchmark runs, not for the minimal production flow.

## Useful Checks

Inspect per-script options:

```bash
python3 scripts/pipeline/export_media.py --help
python3 scripts/pipeline/build_photo_pre_model_annotations.py --help
python3 scripts/pipeline/probe_vlm_photo_boundaries.py --help
python3 scripts/pipeline/review_performance_proxy_gui.py --help
python3 scripts/pipeline/copy_reviewed_set_assets.py --help
```
