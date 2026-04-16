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

ML boundary verifier:

- `export_ml_boundary_reviewed_truth.py` -> `_workspace/ml_boundary_reviewed_truth.csv` from `performance_proxy_index.json` + `review_state.json`
- `build_ml_boundary_candidate_dataset.py` -> `_workspace/ml_boundary_candidates.csv`, `_workspace/ml_boundary_attrition.json`, `_workspace/ml_boundary_dataset_report.json`
- `validate_ml_boundary_dataset.py` -> validates candidate CSV + attrition JSON and can write `_workspace/ml_boundary_validation_report.json`
- `run_ml_boundary_pipeline.py` -> end-to-end orchestrator (per-day export/build/validate, merged-corpus split, train, evaluate)
- `train_ml_boundary_verifier.py` -> writes training artifacts:
  - `.../ml_boundary_models/RUN/training_plan.json`
  - `.../ml_boundary_models/RUN/training_metadata.json`
  - `.../ml_boundary_models/RUN/feature_columns.json`
  - `.../ml_boundary_models/RUN/training_summary.json`
- `evaluate_ml_boundary_verifier.py` -> writes evaluation metrics from model inference on test split:
  - `.../ml_boundary_eval/RUN/metrics.json`

ML boundary corpus split surface:

- all input days are merged into one candidate corpus first
- train/validation/test split is applied to merged candidate rows
- input days are data sources, not split units
- one-day runs are allowed when the merged corpus has at least 3 candidate rows
- `--split-strategy` accepts `global_random` or `global_stratified`
- `global_stratified` is the requested default, but the pipeline may fall back to `global_random` when stratified allocation or held-out coverage cannot be satisfied
- exact defaults:
  - `split_strategy = global_stratified`
  - `train_fraction = 0.70`
  - `validation_fraction = 0.15`
  - `test_fraction = 0.15`
  - `split_seed = 42`
- fraction and seed controls:
  - `--train-fraction`
  - `--validation-fraction`
  - `--test-fraction`
  - `--split-seed`
  - `--required-heldout-classes`
- `.vocatio` keys on the first day directory:
  - `ML_SPLIT_STRATEGY`
  - `ML_SPLIT_TRAIN_FRACTION`
  - `ML_SPLIT_VALIDATION_FRACTION`
  - `ML_SPLIT_TEST_FRACTION`
  - `ML_SPLIT_SEED`
- CLI flags override `.vocatio`; `.vocatio` overrides built-in defaults

```bash
ML_SPLIT_STRATEGY=global_stratified
ML_SPLIT_TRAIN_FRACTION=0.70
ML_SPLIT_VALIDATION_FRACTION=0.15
ML_SPLIT_TEST_FRACTION=0.15
ML_SPLIT_SEED=42
```

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

ML boundary verifier checklist:

```bash
python3 scripts/pipeline/run_ml_boundary_pipeline.py /data/20260323 --mode tabular_only --split-strategy global_stratified --train-fraction 0.70 --validation-fraction 0.15 --test-fraction 0.15 --split-seed 42 --model-run-id run-001
python3 scripts/pipeline/run_ml_boundary_pipeline.py /data/20260323 /data/20260324 /data/20260325 --mode tabular_only --split-strategy global_stratified --model-run-id run-001
python3 scripts/pipeline/run_ml_boundary_pipeline.py /data/20260323 /data/20260324 /data/20260325 --mode tabular_only --prepare-only --model-run-id run-001
```

After a pipeline run, inspect `FIRST_DAY_WORKSPACE/ml_boundary_corpus/ml_boundary_pipeline_summary.json` and verify:

- `requested_split_strategy`
- `effective_split_strategy`
- `required_heldout_classes` to confirm which held-out coverage classes were enforced for the run
