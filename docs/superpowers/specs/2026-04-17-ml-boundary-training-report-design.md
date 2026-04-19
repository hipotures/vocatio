# ML Boundary Training Report Design

## Goal

Add a short, human-readable end-of-training report for the ML boundary verifier, available both:

- in CLI output at the end of `train_ml_boundary_verifier.py`
- as a dedicated JSON artifact next to the existing training artifacts

This report is a summary layer. It does not replace:

- `training_plan.json`
- `training_metadata.json`
- `feature_columns.json`
- `training_summary.json`

## New Artifact

Add a new artifact:

- `training_report.json`

Location:

- inside the same output directory as the existing training artifacts

## Report Contents

The report is intentionally short and explicit. It should contain:

- `output_dir`
- `mode`
- `split_manifest_scope`
- `train_row_count`
- `validation_row_count`
- `shared_feature_count`
- `image_feature_count`
- `missing_annotation_photo_count`
- `missing_annotation_candidate_count`
- `segment_type`
  - `best_model`
  - `validation_score`
  - `model_dir`
- `boundary`
  - `best_model`
  - `validation_score`
  - `model_dir`
- `artifact_paths`
  - `training_plan`
  - `training_metadata`
  - `feature_columns`
  - `training_summary`
  - `training_report`
  - `segment_type_model_dir`
  - `boundary_model_dir`

## Data Sources

The report should be derived from already-available data:

- row counts and split scope from `TrainingDataBundle`
- feature counts from `TrainingDataBundle`
- missing-annotation counts from `TrainingDataBundle`
- winning model names and validation scores from the existing predictor training summary data
- artifact paths from the same artifact-path builder already used elsewhere in the training script

No additional model introspection is required beyond what is already returned by the current training flow.

## Best Model Definition

For each predictor:

- `best_model` is the `best_model` entry from the predictor fit summary excerpt when available
- if the predictor summary does not expose `best_model`, fall back to the predictor class name already used in the existing summary

This keeps the report robust across predictor backends while still preferring the actual winning AutoGluon model name when it exists.

## CLI Output

At the end of a successful run, after artifacts are written, print a short block to stderr.

It must include:

- output directory
- segment_type best model and validation score
- boundary best model and validation score
- shared/image feature counts
- missing-annotation coverage
- path to `training_report.json`

The block must be concise and appear once.

The existing success lines may remain, but this new block should be the clear end-of-run summary.

## Non-Goals

This change does not:

- change training behavior
- change evaluation behavior
- change dataset generation
- add new metrics
- expose full hyperparameter dumps

The purpose is readability and operational clarity, not exhaustive experiment logging.
