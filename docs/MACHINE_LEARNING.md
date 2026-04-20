# Machine Learning in Vocatio

## Scope

`vocatio` currently uses several ML/NLP components, but only one subsystem is trained inside this repository:

- the **ML boundary verifier** for image-based segment boundaries

Everything else is either:

- a feature generator feeding that verifier
- a heuristic baseline
- a VLM/LLM-assisted inference path
- an ASR or benchmark utility

This document describes the current code-level contract: models, features, metrics, artifacts, and how the pieces fit together.

## At a Glance

| Area | Purpose | Main scripts | Model family | Status |
| --- | --- | --- | --- | --- |
| ML boundary verifier | Predict boundary + left/right segment type around a candidate gap | `run_ml_boundary_pipeline.py`, `train_ml_boundary_verifier.py`, `evaluate_ml_boundary_verifier.py` | AutoGluon `TabularPredictor` or `MultiModalPredictor` | Trainable in repo |
| Image-only embeddings | Embed photo previews for boundary signals | `embed_photo_previews_dinov2.py` | DINOv2 | Inference only |
| Image-only bootstrap scorer | Score adjacent photo gaps before ML verification | `build_photo_boundary_features.py`, `bootstrap_photo_boundaries.py` | Heuristic, not learned | Baseline |
| Photo pre-model annotations | Extract structured costume/person descriptors per photo | `build_photo_pre_model_annotations.py` | VLM via `llamacpp`, `ollama`, or `vllm` | Inference only |
| VLM boundary probe | Ask a VLM to detect one cut inside a centered photo window | `probe_vlm_photo_boundaries.py` | Local VLMs such as Qwen2.5-VL / InternVL3 | Inference only |
| Audio transcription | Transcribe synced video clips | `transcribe_video_batch.py`, `transcribe_video_batch_api.py` | WhisperX | Inference only |
| Semantic announcement extraction | Detect competition announcements from transcript windows | `demo_semantic_announcement_classifier.py`, `benchmark_semantic_announcement_models.py` | LLM/OpenAI-compatible backend | Experimental benchmark |
| Photo captioning experiments | Caption stage photos or extract schema fields | `benchmark_caption_llamacpp_qwen4b.py`, `benchmark_schema_llamacpp_qwen4b.py`, `test_caption_*.py` | BLIP, Florence-2, GIT, Ollama Qwen/Gemma, llama.cpp Qwen | Experimental benchmark |

## 1. ML Boundary Verifier

### What it predicts

The verifier is the core trained ML system in the repo. It always trains three predictors on the same candidate windows:

- `left_segment_type`: multiclass
- `right_segment_type`: multiclass
- `boundary`: binary

Current segment labels come from `scripts/pipeline/lib/ml_boundary_truth.py`:

- `performance`
- `ceremony`
- `warmup`

### Training data flow

The end-to-end flow is:

1. `export_ml_boundary_reviewed_truth.py`
   - exports reviewed per-photo truth from GUI state
2. `build_ml_boundary_candidate_dataset.py`
   - generates centered candidate windows around large photo gaps
3. `validate_ml_boundary_dataset.py`
   - checks schema and split assumptions
4. `run_ml_boundary_pipeline.py`
   - merges day datasets into one corpus, splits it, trains, and evaluates
5. `train_ml_boundary_verifier.py`
   - trains predictor artifacts
6. `evaluate_ml_boundary_verifier.py`
   - runs inference on the test split and writes metrics

### Candidate generation

Candidate windows are built around adjacent photos whose center gap exceeds a threshold:

- default `gap_threshold_seconds = 20.0`
- default `window_radius = 2`
- internal window size = `2 * window_radius`

A candidate row includes:

- center photo ids
- left/right ground-truth segment ids and segment types
- the binary `boundary` label
- per-frame photo ids, relative paths, timestamps, thumb paths, and preview paths

Rows without a full centered window or required artifacts are excluded and counted in attrition reports.

### Feature set

The model features are built in `scripts/pipeline/lib/ml_boundary_features.py`.

Feature families:

1. **Gap geometry**
   - `gap_12`, `gap_23`, ...
   - `center_gap_seconds`
   - `left_internal_gap_mean`
   - `right_internal_gap_mean`
   - `local_gap_median`
   - `gap_ratio`
   - `gap_is_local_outlier`
   - `max_gap_in_window`
   - `gap_variance`

2. **Heuristic boundary carry-over**
   - imported from `photo_boundary_scores.csv`
   - columns per adjacent pair:
   - `heuristic_dino_dist_*`
   - `heuristic_boundary_score_*`
   - `heuristic_distance_zscore_*`
   - `heuristic_smoothed_distance_zscore_*`
   - `heuristic_time_gap_boost_*`
   - `heuristic_boundary_label_*`

3. **Per-side descriptor aggregates**
   - built from `photo_pre_model_annotations/*.json`
   - examples:
   - `left_people_count`, `right_people_count`
   - `left_performer_view`, `right_performer_view`
   - `left_upper_garment`, `right_upper_garment`
   - `left_lower_garment`, `right_lower_garment`
   - `left_sleeves`, `right_sleeves`
   - `left_leg_coverage`, `right_leg_coverage`
   - `left_headwear`, `right_headwear`
   - `left_footwear`, `right_footwear`
   - `left_dance_style_hint`, `right_dance_style_hint`
   - multivalue expansions such as `left_dominant_colors_01..05`, `right_props_01..05`

4. **Optional image-path features**
   - only in `tabular_plus_thumbnail`
   - `frame_01_thumb_path`, `frame_02_thumb_path`, ...
   - these are passed to AutoGluon as `image_path` columns

Important:

- the verifier currently uses `embeddings=None` in training-time feature derivation
- DINOv2 distances are consumed through heuristic CSV features, not as raw embedding vectors

### Training modes

Supported modes from `scripts/pipeline/lib/ml_boundary_training_data.py`:

- `tabular_only`
  - uses AutoGluon `TabularPredictor`
- `tabular_plus_thumbnail`
  - uses AutoGluon `MultiModalPredictor`
  - combines tabular features with thumbnail image paths

Training defaults from `scripts/pipeline/lib/ml_boundary_training_options.py`:

- preset: `medium_quality`
- time limit: none unless `--train-minutes` is provided

Important:

- the repo does **not** hard-code one concrete estimator such as LightGBM or CatBoost
- AutoGluon chooses the winning internal model family at runtime
- the selected winner is recorded in `training_summary.json` and surfaced as `best_model` in `training_report.json`

### Split strategy

Corpus splitting happens after merging all candidate rows from all days.

Defaults:

- `split_strategy = global_stratified`
- `train_fraction = 0.70`
- `validation_fraction = 0.15`
- `test_fraction = 0.15`
- `split_seed = 42`

`global_stratified` may fall back to `global_random` if held-out class coverage cannot be satisfied.

### Metrics

Metric specs are defined in `scripts/pipeline/lib/ml_boundary_metrics.py`.

Training-time optimization:

- `left_segment_type`: `f1_macro`
- `right_segment_type`: `f1_macro`
- `boundary`: `f1`

Evaluation-time metrics from `evaluate_ml_boundary_verifier.py`:

- `left_segment_type_macro_f1`
- `left_segment_type_accuracy`
- `left_segment_type_correct_count`
- `left_segment_type_incorrect_count`
- `left_segment_type_confusion_matrix`
- `right_segment_type_macro_f1`
- `right_segment_type_accuracy`
- `right_segment_type_correct_count`
- `right_segment_type_incorrect_count`
- `right_segment_type_confusion_matrix`
- `boundary_f1`
- `boundary_true_positive_count`
- `boundary_false_positive_count`
- `boundary_false_negative_count`
- `boundary_true_negative_count`
- `boundary_correct_count`
- `boundary_incorrect_count`
- `review_cost_metrics`

`review_cost_metrics` is operational rather than academic. It estimates manual correction cost:

- `merge_run_count`
- `split_run_count`
- `estimated_correction_actions`

Boundary thresholding is currently fixed:

- policy: `fixed`
- threshold: `0.5`

### Artifacts

Training artifacts:

- `ml_boundary_models/<run_id>/training_plan.json`
- `ml_boundary_models/<run_id>/training_metadata.json`
- `ml_boundary_models/<run_id>/feature_columns.json`
- `ml_boundary_models/<run_id>/training_summary.json`
- `ml_boundary_models/<run_id>/training_report.json`
- `ml_boundary_models/<run_id>/left_segment_type_model/`
- `ml_boundary_models/<run_id>/right_segment_type_model/`
- `ml_boundary_models/<run_id>/boundary_model/`

Evaluation artifacts:

- `ml_boundary_eval/<run_id>/metrics.json`

Corpus-level summary:

- `ml_boundary_corpus/ml_boundary_pipeline_summary.json`

## 2. Image-Only Stage-1 Signals

This layer builds the raw visual signals used by heuristics and later consumed by the verifier.

### DINOv2 preview embeddings

`embed_photo_previews_dinov2.py` embeds preview JPEGs and writes:

- `features/dinov2_embeddings.npy`
- `features/dinov2_index.csv`

Supported DINOv2 model names:

- `dinov2_vits14` -> 384 dims
- `dinov2_vitb14` -> 768 dims
- `dinov2_vitl14` -> 1024 dims
- `dinov2_vitg14` -> 1536 dims

Default:

- `model_name = dinov2_vitb14`

### Adjacent-pair boundary features

`build_photo_boundary_features.py` converts adjacent preview embeddings and photo quality data into deterministic pair features:

- `time_gap_seconds`
- `dino_cosine_distance`
- `rolling_dino_distance_mean`
- `rolling_dino_distance_std`
- `distance_zscore`
- `left_flag_blurry`
- `right_flag_blurry`
- `left_flag_dark`
- `right_flag_dark`
- `brightness_delta`
- `contrast_delta`

These are not yet the trained verifier features. They are stage-1 pairwise signals.

### Bootstrap heuristic scorer

`bootstrap_photo_boundaries.py` turns adjacent-pair features into a heuristic boundary score.

Defaults:

- `zscore_threshold = 1.5`
- `soft_gap_seconds = 20.0`
- `hard_gap_seconds = 90.0`

Outputs in `photo_boundary_scores.csv`:

- `smoothed_distance_zscore`
- `time_gap_boost`
- `boundary_score`
- `boundary_label`
- `boundary_reason`
- `model_source = bootstrap_heuristic`

This is a deterministic baseline, not a learned model.

### Pre-model photo descriptors

`build_photo_pre_model_annotations.py` runs a VLM per photo and stores structured descriptor JSON files under:

- `photo_pre_model_annotations/`

Canonical descriptor fields from `lib/photo_pre_model_annotations.py`:

- `people_count`
- `performer_view`
- `upper_garment`
- `lower_garment`
- `sleeves`
- `leg_coverage`
- `dominant_colors`
- `headwear`
- `footwear`
- `props`
- `dance_style_hint`

These descriptors are later aggregated across left and right sides of a candidate window and become verifier features.

Default inference path:

- provider: `llamacpp`
- model: `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`

Supported providers:

- `llamacpp`
- `ollama`
- `vllm`

Operational metrics recorded by this script are throughput-oriented:

- `prompt_n`
- `prompt_ms`
- `predicted_n`
- `predicted_ms`
- `predicted_per_second`

## 3. VLM Boundary Probe

`probe_vlm_photo_boundaries.py` is a separate inference path that asks a VLM to decide whether a centered window contains one cut.

### Inputs

The VLM receives:

- a centered sequence of consecutive photos
- optional ML hints from the trained verifier
- optional heuristic pair data from `photo_boundary_scores.csv`
- optional per-photo descriptors from `photo_pre_model_annotations/`

The prompt explicitly instructs the model to choose at most one boundary and to return structured JSON.

### Outputs

Each batch writes a row to `vlm_boundary_results.csv` with fields such as:

- `decision`
- `cut_after_local_index`
- `cut_after_global_row`
- `cut_left_relative_path`
- `cut_right_relative_path`
- `reason`
- `response_status`
- `raw_response`

### Model configuration

Defaults in code:

- provider: `ollama`
- model: `qwen3.5:9b`

Manual presets in `conf/manual_vlm_models.yaml` currently define:

- `Qwen2.5-VL 7B temp 0`
- `Qwen2.5-VL GGUF think off`
- `InternVL3 8B temp 0.2`

Supported providers:

- `ollama`
- `llamacpp`
- `vllm`

### ML hint contract

If a trained verifier model is available, the probe can inject:

- boundary likelihood
- left-side segment type likelihood
- right-side segment type likelihood

These hints are built by reusing the verifier feature contract and running predictor inference at probe time.

### Metrics

There is no built-in offline accuracy report for the VLM boundary probe in the repository.

Current outputs support:

- structured response validation
- manual review in the GUI
- comparison between VLM runs

## 4. Audio and Transcript ML

### WhisperX transcription

Transcription is handled by:

- `transcribe_video_batch.py`
- `transcribe_video_batch_api.py`

Default ASR model:

- `large`

Key runtime options:

- device, batch size, chunk size, compute type
- language default `pl`
- optional alignment
- optional Hugging Face token for alignment/diarization models

The Python API variant also adds domain-specific defaults such as:

- an initial prompt tuned for Polish dance competition announcements
- hotwords like `numer`, `numer startowy`, `kategoria`

The repo does not currently define a standard ASR evaluation artifact such as WER/CER. Most evaluation here is operational or manual.

### Semantic announcement classifier

The semantic announcement path is experimental and transcript-driven:

- `demo_semantic_announcement_classifier.py`
- `benchmark_semantic_announcement_models.py`

Default classification model in the demo script:

- `gpt-5.4`

Supported backends:

- `openai-compatible`
- `local-openai`
- `codex-exec`
- `qwen-local`

The benchmark script accepts arbitrary `--models` and scores prepared transcript windows.

Recorded benchmark fields include:

- `strict_ok`
- `event_ok`
- `start_number_ok`
- `category_ordinal_ok`
- `response_status`
- `response_error`
- `finish_reason`
- `content_length`
- `reasoning_length`
- `prompt_eval_count`
- `eval_count`
- `elapsed_seconds`

The summary table reports per-model strict matches `X/N` and elapsed time.

## 5. Captioning and Schema Extraction Experiments

The repo also contains several experimental single-image captioning or structured extraction scripts. These are useful for model comparison and prompt development, but they are not part of the trained boundary verifier.

Model families explicitly referenced in the code:

- llama.cpp OpenAI-compatible:
  - `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`
- Ollama caption models:
  - `qwen3.5:2b`
  - `qwen3.5:4b`
  - `gemma4:e2b`
  - `gemma4:e4b`
- Hugging Face captioning baselines:
  - `Salesforce/blip-image-captioning-base`
  - `microsoft/Florence-2-base-ft`
  - `microsoft/git-base-coco`

Representative scripts:

- `benchmark_caption_llamacpp_qwen4b.py`
- `benchmark_schema_llamacpp_qwen4b.py`
- `test_caption_blip.py`
- `test_caption_florence2.py`
- `test_caption_git.py`
- `test_caption_ollama_qwen35_2b.py`
- `test_caption_ollama_qwen35_4b.py`
- `test_caption_ollama_gemma4_e2b.py`
- `test_caption_ollama_gemma4_e4b.py`

Metrics here are throughput-oriented rather than task-accuracy-oriented:

- `elapsed_seconds`
- `samples`
- `prompt_n`
- `prompt_ms`
- `predicted_n`
- `predicted_ms`
- `predicted_per_second`

## 6. Recommended Files to Inspect After a Run

For the verifier:

- `ml_boundary_corpus/ml_boundary_pipeline_summary.json`
- `ml_boundary_models/<run_id>/training_report.json`
- `ml_boundary_models/<run_id>/training_summary.json`
- `ml_boundary_models/<run_id>/feature_columns.json`
- `ml_boundary_eval/<run_id>/metrics.json`

For image-only feature generation:

- `photo_boundary_features.csv`
- `photo_boundary_scores.csv`
- `features/dinov2_index.csv`
- `features/dinov2_embeddings.npy`

For VLM and descriptor runs:

- `photo_pre_model_annotations/**/*.json`
- `vlm_boundary_results.csv`
- `vlm_runs/<run_id>.json`

For semantic announcement experiments:

- `semantic_announcement_windows.csv`
- `semantic_announcement_classification.csv`
- semantic benchmark CSV/JSONL outputs under `/tmp` or the requested output path

## Practical Summary

If you only need the short version:

- the repository has **one trainable ML system**: the AutoGluon-based ML boundary verifier
- its features combine gap geometry, heuristic image-only signals, and VLM-generated photo descriptors
- DINOv2, WhisperX, VLMs, and caption models are used as inference or benchmark components around that core system
- the main objective metrics currently implemented are for the boundary verifier and the semantic announcement benchmark
