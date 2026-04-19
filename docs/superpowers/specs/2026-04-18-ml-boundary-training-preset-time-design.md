# ML Boundary Training Preset And Time Design

## Goal

Add a single pair of training controls that applies equally to both ML predictors:

- `segment_type`
- `boundary`

The user wants to compare training behavior under different AutoGluon presets and a per-predictor time budget without configuring the two predictors separately.

## User-Facing Changes

Add the same options to:

- `scripts/pipeline/train_ml_boundary_verifier.py`
- `scripts/pipeline/run_ml_boundary_pipeline.py`

New CLI options:

- `--preset`
- `--train-minutes`

Semantics:

- `--preset` is passed through to AutoGluon as-is.
- `--train-minutes N` means `N` minutes of training time **per predictor**, not shared across both predictors.

Defaults must preserve current behavior:

- preset defaults to current `medium_quality`
- no time limit by default

## Behavior

### Preset

The code does not introduce local alias mapping for presets.

Accepted values are whatever AutoGluon supports. The main intended examples are:

- `medium_quality`
- `best_quality`

The user may also pass shorthand values if AutoGluon itself accepts them.

### Time Limit

`--train-minutes` is optional.

When omitted:

- do not pass a time limit to AutoGluon

When provided:

- validate it as a positive numeric value
- convert minutes to seconds
- pass the resulting `time_limit` to each predictor's `.fit(...)`

If `--train-minutes 10` is passed:

- `segment_type` gets `time_limit=600`
- `boundary` gets `time_limit=600`

## Internal Design

Introduce a small shared training-options helper so training defaults are not duplicated across scripts.

The shared contract should cover:

- default preset
- preset passthrough value
- optional `train_minutes`
- optional `time_limit_seconds`

`train_ml_boundary_verifier.py` should:

- parse the new options
- resolve and validate training options once
- use the same resolved options for both predictors
- persist the resolved values in training artifacts

`run_ml_boundary_pipeline.py` should:

- parse the same options
- forward them unchanged to `train_ml_boundary_verifier.py`
- include them in the pipeline summary

## Artifacts And Reporting

Persist the resolved training options in:

- `training_plan.json`
- `training_report.json`
- `ml_boundary_pipeline_summary.json`

At minimum store:

- `training_preset`
- `train_minutes`
- `time_limit_seconds`

The training CLI console output should also show the preset and the effective time limit for each predictor so the run configuration is visible in terminal logs.

## Validation

Add or update tests to cover:

- default behavior remains `medium_quality` with no time limit
- explicit preset is forwarded to both predictors
- explicit `--train-minutes` is converted to seconds and applied to both predictors
- pipeline runner forwards both options to the training script
- persisted JSON artifacts include the resolved training options

## Non-Goals

This change does not:

- add separate settings per predictor
- change evaluation logic
- change predictor feature sets
- add local preset alias resolution beyond what AutoGluon already supports
