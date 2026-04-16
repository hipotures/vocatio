# Pipeline Scripts

Safe copied snapshot of the current event workflow scripts migrated from `scriptoza`.

This directory currently contains the operational pipeline pieces for:

- day media export and merge
- audio/video sync estimation and application
- transcription
- announcement extraction
- semantic announcement experiments and benchmarking
- performance timeline building
- photo assignment
- proxy generation
- review GUI support
- reviewed set export

Important:

- these files are copied here for migration work
- the original scripts in `scriptoza` are still the active source during the transition
- changes made here are intentionally isolated from the current production workflow until the migration is completed

Included files:

- `export_event_media_csv.py`
- `merge_event_media_csv.py`
- `estimate_video_sync_map.py`
- `apply_video_sync_map.py`
- `transcribe_video_batch.py`
- `transcribe_video_batch_api.py`
- `extract_announcement_candidates.py`
- `extract_announcement_candidates_semantic.py`
- `build_performance_timeline.py`
- `assign_photos_to_timeline.py`
- `generate_photo_proxy_jpg.py`
- `build_performance_proxy_index.py`
- `review_performance_proxy_gui.py`
- `copy_reviewed_set_assets.py`
- `test_segment_speech_music.py`
- `demo_semantic_announcement_classifier.py`
- `test_diarize_clip.py`
- `build_semantic_announcement_demo.py`
- `benchmark_semantic_announcement_models.py`
- `generate_mv_commands_from_timeline.py`

Configuration files:

- `../../conf/copy_reviewed_set_assets.default.yaml`
- `../../conf/copy_reviewed_set_assets.raw.yaml`

Video export behavior in `copy_reviewed_set_assets.py`:

- uses the reviewed set interval as the trim source
- prefers `performance_start_local` and `performance_end_local`
- falls back to first/last photo timestamps when needed
- reads video trim settings from the YAML profile:
  - `start_trim_seconds`
  - `end_padding_seconds`
- writes `video_markers.csv` next to exported set videos

Photo selection export behavior:

- `review_performance_proxy_gui.py` supports `Ctrl+E` to save the currently selected photo rows to a JSON file
- relative JSON names are written to `DAY/_workspace`
- `copy_reviewed_set_assets.py` accepts that JSON path as the third positional argument
- in JSON selection mode it exports only the listed photos and skips video selection

Image-only stage 1 pipeline:

- `export_media.py` -> `_workspace/media_manifest.csv`
- `extract_embedded_photo_jpg.py` -> `_workspace/photo_embedded_manifest.csv`
- `extract_embedded_photo_jpg.py` reads photo rows from `_workspace/media_manifest.csv`
- `build_photo_quality_annotations.py` -> `_workspace/photo_quality.csv` from preview JPGs in `photo_embedded_manifest.csv`
- `embed_photo_previews_dinov2.py` -> `_workspace/features/dinov2_embeddings.npy`, `_workspace/features/dinov2_index.csv`
- `build_photo_boundary_features.py` -> `_workspace/photo_boundary_features.csv`
- `bootstrap_photo_boundaries.py` -> `_workspace/photo_boundary_scores.csv`
- `build_photo_segments.py` -> `_workspace/photo_segments.csv`
- `build_photo_review_index.py` -> `_workspace/performance_proxy_index.json`

ML boundary verifier scaffold:

- `build_ml_boundary_candidate_dataset.py` -> `_workspace/ml_boundary_candidates.csv`, `_workspace/ml_boundary_attrition.json`, `_workspace/ml_boundary_dataset_report.json`
- `validate_ml_boundary_dataset.py` -> validates candidate CSV + attrition JSON and can write `_workspace/ml_boundary_validation_report.json`
- `train_ml_boundary_verifier.py` -> writes scaffold training artifacts:
  - `_workspace/ml_boundary_models/RUN/training_plan.json`
  - `_workspace/ml_boundary_models/RUN/training_metadata.json`
- `evaluate_ml_boundary_verifier.py` -> writes scaffold evaluation metrics:
  - `_workspace/ml_boundary_eval/RUN/metrics.json`

Important for `uv` users:

- the ML verifier train/eval scaffold should run under the isolated `autogluon` dependency group
- use:
  - `uv run --no-default-groups --group autogluon ...`
- this avoids the intentional `torch` / `torchvision` conflict with the default `gpu` group

Manual smoke checklist:

```bash
python3 scripts/pipeline/export_media.py /data/20260323
python3 scripts/pipeline/extract_embedded_photo_jpg.py /data/20260323
python3 scripts/pipeline/build_photo_quality_annotations.py /data/20260323
python3 scripts/pipeline/embed_photo_previews_dinov2.py /data/20260323
python3 scripts/pipeline/build_photo_boundary_features.py /data/20260323
python3 scripts/pipeline/bootstrap_photo_boundaries.py /data/20260323
python3 scripts/pipeline/build_photo_segments.py /data/20260323
python3 scripts/pipeline/build_photo_review_index.py /data/20260323
python3 scripts/pipeline/review_performance_proxy_gui.py /data/20260323 --index performance_proxy_index.json
```

ML boundary verifier scaffold checklist:

```bash
python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py /data/20260323
python3 scripts/pipeline/validate_ml_boundary_dataset.py /data/20260323/_workspace/ml_boundary_candidates.csv --attrition-json /data/20260323/_workspace/ml_boundary_attrition.json --report-json /data/20260323/_workspace/ml_boundary_validation_report.json
uv run --no-default-groups --group autogluon python3 scripts/pipeline/train_ml_boundary_verifier.py /data/20260323/_workspace/ml_boundary_candidates.csv --mode tabular_only --output-dir /data/20260323/_workspace/ml_boundary_models/run-001
uv run --no-default-groups --group autogluon python3 scripts/pipeline/evaluate_ml_boundary_verifier.py /data/20260323/_workspace/ml_boundary_candidates.csv --model-dir /data/20260323/_workspace/ml_boundary_models/run-001 --output-dir /data/20260323/_workspace/ml_boundary_eval/run-001
```
