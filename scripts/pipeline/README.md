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
