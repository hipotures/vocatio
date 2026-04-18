## Goal

Stage A prepares the ML boundary verifier to become the structured signal source for the main visual boundary VLM.

This stage does **not** change the VLM prompt yet. It only expands the ML feature path so the two ML predictors can consume the same non-image signals that are currently injected directly into `probe_vlm_photo_boundaries.py`.

After Stage A is implemented, work stops. The user will run real ML experiments and review the results before any Stage B prompt integration begins.

## Scope

In scope:

- extend ML training features with the heuristic boundary signals currently passed to the VLM prompt
- keep existing descriptor-based feature extraction from `photo_pre_model_annotations`
- keep both ML predictors:
  - `segment_type`
  - `boundary`
- persist enough metadata to make Stage B possible later
- stop after ML retraining and evaluation are possible with the richer feature set

Out of scope for Stage A:

- changing `probe_vlm_photo_boundaries.py` prompt contents
- passing ML predictions into the VLM prompt
- removing heuristic or pre-model lines from the VLM prompt
- changing candidate selection
- changing review GUI behavior

## Current State

Today the main visual boundary VLM receives three categories of inputs:

1. the five attached images
2. heuristic gap hints derived from `photo_boundary_scores.csv`
3. per-photo pre-model annotations from `photo_pre_model_annotations`

The ML boundary verifier currently consumes:

1. gap/time features derived from the 5-frame candidate window
2. descriptor features derived from `photo_pre_model_annotations`
3. optional thumbnails in `tabular_plus_thumbnail`

The ML verifier does **not** currently consume the heuristic boundary features that are passed to the VLM prompt, and it does **not** consume DINO-derived distances in the active training path.

## Stage A Design

Stage A makes ML the aggregator of all structured, non-image signals that are relevant to a boundary candidate.

The resulting ML input should cover:

1. existing gap/time features
2. heuristic boundary features from `photo_boundary_scores.csv`
3. pre-model annotation descriptors from `photo_pre_model_annotations`

The prompt-facing VLM path remains unchanged during Stage A. This allows isolated evaluation of whether the richer ML feature space improves the two predictors before the prompt contract is changed.

## Feature Additions

### Existing feature blocks to keep

Keep the current ML feature families unchanged:

- gap/time features from the 5-frame candidate window
- flattened left/right descriptor features derived from `photo_pre_model_annotations`
- optional thumbnail columns for `tabular_plus_thumbnail`

### New heuristic boundary feature block

For each 5-frame candidate window, ML should ingest the same adjacent-gap heuristic signals that are currently summarized into the prompt:

- `dino_cosine_distance`
- `boundary_score`

Recommended additions from the same source row:

- `distance_zscore`
- `smoothed_distance_zscore`
- `time_gap_boost`
- `boundary_label`

These values should be attached per adjacent pair inside the 5-frame window. The feature schema should follow the existing naming style and remain explicit about pair position, for example:

- `heuristic_dino_dist_12`
- `heuristic_dino_dist_23`
- `heuristic_dino_dist_34`
- `heuristic_dino_dist_45`
- `heuristic_boundary_score_12`
- `heuristic_boundary_score_23`
- `heuristic_boundary_score_34`
- `heuristic_boundary_score_45`

And similarly for the additional z-score and label features.

This keeps the ML input close to the prompt semantics while preserving positional structure.

## Data Joining Contract

The source of truth for heuristic features is `photo_boundary_scores.csv`.

Stage A must add a deterministic join from each ML candidate row to the corresponding adjacent frame-pair rows in `photo_boundary_scores.csv`.

The join key should be based on candidate frame relative paths already present in the ML candidate dataset:

- `frame_01_relpath`
- `frame_02_relpath`
- `frame_03_relpath`
- `frame_04_relpath`
- `frame_05_relpath`

The four adjacent heuristic rows are therefore:

- `(frame_01_relpath, frame_02_relpath)`
- `(frame_02_relpath, frame_03_relpath)`
- `(frame_03_relpath, frame_04_relpath)`
- `(frame_04_relpath, frame_05_relpath)`

This contract mirrors how the VLM prompt path already looks up heuristic rows, so it minimizes ambiguity and avoids inventing a second matching scheme.

## Missing Data Handling

Missing heuristic rows must not silently crash the ML dataset builder.

Stage A should treat missing heuristic rows similarly to descriptor coverage:

- numeric heuristic fields fall back to a stable missing representation
- categorical heuristic fields fall back to `__missing__`
- missing rows are counted and reported in aggregate at the end, not logged one by one

Required counters:

- `missing_heuristic_pair_count`
- `missing_heuristic_candidate_count`

These should be carried alongside the existing descriptor coverage counters in ML training metadata and summary artifacts.

## Predictor Contract

Stage A does not change the predictor set.

The two predictors remain:

- `segment_type`
- `boundary`

They continue to share the same candidate rows and the same feature table, but with the richer heuristic block included.

The current metric policy remains in force:

- `segment_type` optimized for `f1_macro`
- `boundary` optimized for `f1`

## Artifacts

Stage A must preserve the current training and evaluation artifacts and extend them with the new heuristic coverage reporting.

At minimum, the following artifacts should reflect Stage A inputs:

- `feature_columns.json`
- `training_metadata.json`
- `training_report.json`
- `ml_boundary_pipeline_summary.json`

These should make it obvious that the ML run now includes heuristic boundary features and should expose missing-heuristic coverage counts.

## Testing Strategy

Stage A needs focused tests in three areas:

1. feature derivation
   - candidate rows correctly join to four adjacent heuristic rows
   - expected heuristic feature columns are emitted
2. missing-data behavior
   - missing `photo_boundary_scores` pairs do not crash the dataset build
   - aggregate missing-heuristic counters are correct
3. artifact/reporting integrity
   - training metadata and reports include heuristic coverage counts
   - feature column outputs include the new heuristic feature names

Real-day evaluation is intentionally deferred to the user after Stage A implementation.

## Rollout Boundary

Implementation ends when all of the following are true:

- ML training consumes heuristic boundary features from `photo_boundary_scores.csv`
- training/evaluation artifacts expose the richer feature set
- missing heuristic coverage is reported cleanly
- no VLM prompt changes have been made

After that point, work stops and the user tests the improved ML models on real data. Only after that review can Stage B begin.
