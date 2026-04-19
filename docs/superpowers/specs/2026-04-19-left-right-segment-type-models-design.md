# Left/Right Segment Type Models Design

## Goal

Replace the current ambiguous `segment_type` predictor with two explicit multiclass predictors:

- `left_segment_type`
- `right_segment_type`

while keeping the existing binary `boundary` predictor unchanged.

The public contract after this change is:

- `left_segment_type`
- `right_segment_type`
- `boundary`

There is no public `segment_type` alias after the migration.

## Motivation

The current ML setup trains:

- one multiclass predictor called `segment_type`
- one binary predictor called `boundary`

The multiclass predictor currently models only the right side of the gap, but its name does not make that clear. This is too ambiguous for downstream use in VLM hints, GUI output, evaluation reports, and artifact naming.

The desired outcome is a fully explicit three-predictor contract so downstream consumers can reason about:

- what is on the left side of the gap
- what is on the right side of the gap
- whether a boundary exists

without overloading one predictor name.

## Predictor Contract

After the change, the ML boundary verifier consists of three separate predictors:

1. `left_segment_type`
   - multiclass
   - predicts the segment type on the left side of the candidate gap
   - label values come from the existing `left_segment_type` column

2. `right_segment_type`
   - multiclass
   - predicts the segment type on the right side of the candidate gap
   - label values come from the existing `right_segment_type` column

3. `boundary`
   - binary
   - predicts whether the candidate gap is a true boundary
   - label values come from the existing `boundary` column

These predictors are independent. Neither left/right segment predictors imply the boundary decision, and the boundary predictor does not imply left/right segment types.

## Implementation Strategy

The current `segment_type` training path becomes the template for both left and right predictors.

Implementation approach:

- rename the existing `segment_type` predictor path to `right_segment_type`
- add a parallel `left_segment_type` predictor path using the same feature columns and the same split rows
- keep `boundary` unchanged

This means the architecture remains simple:

- same feature generation
- same train/validation/test splits
- same model family
- different label columns per predictor

The implementation must reuse the existing predictor pattern rather than inventing a separate special-case architecture.

## Training Data Contract

The candidate dataset already carries:

- `left_segment_type`
- `right_segment_type`
- `boundary`

The training bundle must expose three predictor payloads instead of two:

- `left_segment_type`
- `right_segment_type`
- `boundary`

Each predictor payload uses:

- the same feature columns
- the same row splits
- its own label column

There is no compatibility alias that maps `segment_type` to `right_segment_type` in the public training contract.

## Artifacts and Metadata

Model directories become:

- `left_segment_type_model`
- `right_segment_type_model`
- `boundary_model`

`feature_columns.json` becomes:

- `left_segment_type_feature_columns`
- `right_segment_type_feature_columns`
- `boundary_feature_columns`
- existing shared/image feature sections remain as needed

Training summary/report payloads must expose separate entries for:

- `left_segment_type`
- `right_segment_type`
- `boundary`

There must be no public `segment_type` artifact key after the migration.

## Evaluation Contract

Evaluation must report left and right segment predictors independently.

Replace the old single `segment_type` evaluation block with:

- `left_segment_type_macro_f1`
- `left_segment_type_accuracy`
- `left_segment_type_correct_count`
- `left_segment_type_incorrect_count`
- `left_segment_type_confusion_matrix`

and:

- `right_segment_type_macro_f1`
- `right_segment_type_accuracy`
- `right_segment_type_correct_count`
- `right_segment_type_incorrect_count`
- `right_segment_type_confusion_matrix`

`boundary` metrics remain unchanged.

Any summary rendering in console output and pipeline reports must render all three predictors explicitly.

## Probe, Prompt, and GUI Contract

ML hints returned to downstream VLM prompting must expose three separate signals:

- boundary
- left-side segment type
- right-side segment type

The prompt and manual GUI prediction must not collapse these into one inferred decision.

They must be presented as separate hints, for example:

- `Boundary: cut/no_cut (...)`
- `Left-side segment: ... (...)`
- `Right-side segment: ... (...)`

This affects at least:

- `probe_vlm_photo_boundaries.py`
- `build_vlm_photo_boundary_gui_index.py`
- `review_performance_proxy_gui.py`

## Files in Scope

Very likely in scope:

- `scripts/pipeline/lib/ml_boundary_training_data.py`
- `scripts/pipeline/train_ml_boundary_verifier.py`
- `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/probe_vlm_photo_boundaries.py`
- `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
- `scripts/pipeline/review_performance_proxy_gui.py`
- `scripts/pipeline/run_ml_boundary_pipeline.py`

Likely tests in scope:

- `scripts/pipeline/test_ml_boundary_training_data.py`
- `scripts/pipeline/test_train_ml_boundary_verifier.py`
- `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- `scripts/pipeline/test_run_ml_boundary_pipeline.py`

## Risks

The main risks are:

- incomplete rename from `segment_type` to `right_segment_type`
- old artifact/report keys surviving in part of the codebase
- probe/index/GUI still expecting `segment_type_feature_columns`
- mixed runtime logic where some places show one side and others show both

The safest migration rule is:

- no public alias
- no partial rename
- explicit three-predictor contract everywhere

## Testing Expectations

At minimum, tests must prove:

- training bundle exposes three predictors
- training artifacts write left/right feature-column sections explicitly
- evaluation writes left/right metrics and confusion matrices separately
- probe ML hints expose left and right segment predictions independently
- GUI/manual ML prediction renders boundary + left + right
- pipeline summary no longer depends on a public `segment_type` predictor key

## Out of Scope

This change does not introduce:

- a transition classifier like `performance->ceremony`
- a multilabel/multitask single-model setup
- any compatibility alias preserving public `segment_type`

The goal is only to replace the ambiguous single side predictor with two explicit side predictors plus the existing boundary predictor.
