# ML Boundary Verifier End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ML boundary verifier flow runnable end-to-end on real reviewed data, from existing review outputs to real AutoGluon training and evaluation.

**Architecture:** Reuse the current review GUI artifacts as the source of final truth, export them into a canonical `ml_boundary_reviewed_truth.csv`, build and validate candidate datasets, then replace the current train/eval scaffolds with real AutoGluon-backed training and evaluation. Keep the current candidate-gap verifier task unchanged, preserve fixed 5-photo windows, and add `.vocatio` / workspace-aware path resolution so the operational CLI is short enough to use in practice.

**Tech Stack:** Python CLI scripts, `pathlib.Path`, `csv`/`json`, existing review GUI state and index payloads, `rich.progress`, AutoGluon Tabular, AutoMM thumbnail experiment, `uv` dependency groups (`gpu` vs `autogluon`).

---

## File Map

### Non-negotiable operational requirement

The completed flow must work from `DAY` plus `.vocatio` workspace resolution, not only from explicit absolute paths.

For all operational ML verifier CLIs in this flow, path resolution must follow:

1. explicit CLI path override
2. `DAY/.vocatio` with `WORKSPACE_DIR=...`
3. `DAY/_workspace`

This applies to:

- `export_ml_boundary_reviewed_truth.py`
- `build_ml_boundary_candidate_dataset.py`
- `validate_ml_boundary_dataset.py`
- `build_ml_boundary_split_manifest.py` when applicable
- `train_ml_boundary_verifier.py`
- `evaluate_ml_boundary_verifier.py`

The operator should be able to run the flow primarily from `DAY`, with `.vocatio` supplying workspace indirection.

### Existing files to modify

- `scripts/pipeline/review_performance_proxy_gui.py`
  - existing source of reviewed split/merge state semantics
  - used as the reference for exporting final reviewed truth
- `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
  - already builds candidate rows once reviewed truth exists
  - should remain the first step **after** reviewed truth export
- `scripts/pipeline/validate_ml_boundary_dataset.py`
  - already validates candidate CSV + attrition JSON
  - needs workspace-aware defaults and corpus-level validation hooks
- `scripts/pipeline/train_ml_boundary_verifier.py`
  - currently scaffold only
  - must become a real AutoGluon train entrypoint
- `scripts/pipeline/evaluate_ml_boundary_verifier.py`
  - currently scaffold only
  - must become a real evaluation entrypoint
- `README.md`
  - top-level operator docs
- `scripts/pipeline/README.md`
  - pipeline-level operator docs

### New files to create

- `scripts/pipeline/export_ml_boundary_reviewed_truth.py`
  - export canonical per-photo final truth from existing review artifacts
- `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`
  - tests for truth export semantics
- `scripts/pipeline/build_ml_boundary_split_manifest.py`
  - optional helper to build explicit train/validation/test split manifests from per-day inputs
- `scripts/pipeline/test_build_ml_boundary_split_manifest.py`
  - tests for split-manifest generation
- `scripts/pipeline/lib/ml_boundary_review_truth_export.py`
  - pure helpers for rebuilding final reviewed sets and flattening them into per-photo truth rows
- `scripts/pipeline/test_ml_boundary_review_truth_export.py`
  - unit tests for export helpers
- `scripts/pipeline/lib/ml_boundary_training_data.py`
  - shared loader/helpers for train/eval dataset preparation, column checks, split joins, class balance summaries
- `scripts/pipeline/test_ml_boundary_training_data.py`
  - tests for train/eval dataset helpers

### Responsibilities by unit

- `export_ml_boundary_reviewed_truth.py`
  - operational missing link
  - converts the current review state into `ml_boundary_reviewed_truth.csv`
- `build_ml_boundary_candidate_dataset.py`
  - unchanged core task
  - becomes operational once reviewed truth export exists
- `build_ml_boundary_split_manifest.py`
  - creates explicit day-level split assignments
  - keeps split policy out of the train script
- `validate_ml_boundary_dataset.py`
  - validates single-day candidate datasets and split manifests
  - produces diagnostics before training
- `train_ml_boundary_verifier.py`
  - real training for:
    - `tabular_only`
    - `tabular_plus_thumbnail`
- `evaluate_ml_boundary_verifier.py`
  - real evaluation of saved run artifacts on a split-aware dataset

## Task 1: Export reviewed truth from the existing review workflow

**Files:**
- Create: `scripts/pipeline/lib/ml_boundary_review_truth_export.py`
- Create: `scripts/pipeline/export_ml_boundary_reviewed_truth.py`
- Test: `scripts/pipeline/test_ml_boundary_review_truth_export.py`
- Test: `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`
- Reference only: `scripts/pipeline/review_performance_proxy_gui.py`

- [ ] **Step 1: Write failing unit tests for final reviewed-truth flattening**

Cover:
- unsplit reviewed set -> one `segment_id`
- split reviewed set -> multiple `segment_id` values by `start_filename`
- merged reviewed sets -> one final `segment_id` spanning both sources
- exported rows preserve:
  - `photo_id`
  - final `segment_id`
  - final `segment_type`
- `segment_type` mapping in v1:
  - `performance`
  - `ceremony`
  - `warmup`

Run: `uv run pytest scripts/pipeline/test_ml_boundary_review_truth_export.py -v`
Expected: FAIL because helpers do not exist yet.

- [ ] **Step 2: Implement pure helpers for rebuilding final reviewed truth**

Implement small helpers in `scripts/pipeline/lib/ml_boundary_review_truth_export.py`:
- load review state JSON
- load review index payload JSON
- rebuild final display sets using the same semantics as current GUI:
  - original set
  - split specs
  - merge specs
- flatten final display sets into per-photo rows:
  - `photo_id`
  - `segment_id`
  - `segment_type`

Keep the exported `segment_id` deterministic:
- derived from final `display_set_id`
- or a stable normalized variant of it

- [ ] **Step 3: Write CLI-level failing test for truth export**

Create a temp workspace with:
- minimal review index payload
- minimal review state
- expected output path `ml_boundary_reviewed_truth.csv`

Run: `uv run pytest scripts/pipeline/test_export_ml_boundary_reviewed_truth.py -v`
Expected: FAIL because CLI does not exist yet.

- [ ] **Step 4: Implement `export_ml_boundary_reviewed_truth.py`**

CLI behavior:
- input: `DAY`
- resolve workspace using:
  1. `--workspace-dir`
  2. `DAY/.vocatio` `WORKSPACE_DIR`
  3. `DAY/_workspace`
- defaults:
  - review index: `performance_proxy_index.json`
  - review state: `review_state.json`
  - output: `ml_boundary_reviewed_truth.csv`
- write CSV:
  - `photo_id,segment_id,segment_type`

Run: `uv run pytest scripts/pipeline/test_ml_boundary_review_truth_export.py scripts/pipeline/test_export_ml_boundary_reviewed_truth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/ml_boundary_review_truth_export.py scripts/pipeline/export_ml_boundary_reviewed_truth.py scripts/pipeline/test_ml_boundary_review_truth_export.py scripts/pipeline/test_export_ml_boundary_reviewed_truth.py
git commit -m "Add ML reviewed truth export"
```

## Task 2: Make the single-day dataset flow operational from `DAY` and `.vocatio`

**Files:**
- Modify: `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
- Modify: `scripts/pipeline/validate_ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`
- Test: `scripts/pipeline/test_validate_ml_boundary_dataset.py`

- [ ] **Step 1: Write failing tests for workspace-aware defaults**

Cover:
- `build_ml_boundary_candidate_dataset.py DAY` works once reviewed truth export exists
- `validate_ml_boundary_dataset.py DAY` resolves:
  - candidate CSV
  - attrition JSON
  - report JSON
  from `.vocatio` / workspace defaults

Run: `uv run pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py -v`
Expected: FAIL on missing `DAY`-mode support in validator.

- [ ] **Step 2: Add `DAY` / workspace-aware path resolution to validator**

Validator should accept either:
- direct candidate CSV path
- or `DAY`, with default workspace artifact names

Keep existing explicit-path mode working.

- [ ] **Step 3: Add operator-friendly defaults and overwrite/report consistency**

Make the first three operational commands short:
- `export_ml_boundary_reviewed_truth.py DAY`
- `build_ml_boundary_candidate_dataset.py DAY`
- `validate_ml_boundary_dataset.py DAY`

Run: `uv run pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py
git commit -m "Add ML dataset workspace defaults"
```

## Task 3: Build explicit split-manifest generation for multi-day training

**Files:**
- Create: `scripts/pipeline/build_ml_boundary_split_manifest.py`
- Create: `scripts/pipeline/test_build_ml_boundary_split_manifest.py`

- [ ] **Step 1: Write failing tests for split-manifest generation**

Cover:
- input: multiple `day_id` rows with metadata such as year/camera/domain-shift hints
- output rows:
  - `day_id`
  - `split_name`
- day-level isolation only
- validation/test class coverage checks when feasible
- explicit failure when requested coverage cannot be satisfied without leakage

Run: `uv run pytest scripts/pipeline/test_build_ml_boundary_split_manifest.py -v`
Expected: FAIL because script does not exist yet.

- [ ] **Step 2: Implement minimal split-manifest builder**

CLI input can be:
- one CSV of day metadata
- or repeated `--day-metadata-csv`

Output:
- `ml_boundary_splits.csv`

Rules:
- no row-level random split
- at least one held-out split should represent domain shift when metadata supports it
- fail explicitly if requested class coverage cannot be satisfied

- [ ] **Step 3: Verify**

Run: `uv run pytest scripts/pipeline/test_build_ml_boundary_split_manifest.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/build_ml_boundary_split_manifest.py scripts/pipeline/test_build_ml_boundary_split_manifest.py
git commit -m "Add ML boundary split manifest builder"
```

## Task 4: Replace train scaffold with real AutoGluon training

**Files:**
- Create: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Create: `scripts/pipeline/test_ml_boundary_training_data.py`
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing dataset-loader tests for training**

Cover:
- load candidate CSV
- join day-level split manifest
- select train/validation rows
- verify required columns by mode:
  - `tabular_only`
  - `tabular_plus_thumbnail`
- reject missing labels

Run: `uv run pytest scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: FAIL because training-data helpers do not exist yet.

- [ ] **Step 2: Implement shared training-data helpers**

In `scripts/pipeline/lib/ml_boundary_training_data.py`:
- load CSV rows for AutoGluon
- enforce CSV-only or explicitly add Parquet support if chosen
- check split manifest coverage
- expose feature columns by mode
- produce train/validation tables for:
  - predictor A: `segment_type`
  - predictor B: `boundary`

- [ ] **Step 3: Replace scaffold-only train with real AutoGluon tabular training**

In `train_ml_boundary_verifier.py`:
- keep existing contract helpers
- add real training path under `uv --group autogluon`
- train two independent predictors:
  - `segment_type`
  - `boundary`
- save:
  - AutoGluon model directories
  - `training_plan.json`
  - `training_metadata.json`
  - feature config / column lists
  - summary metrics from training

- [ ] **Step 4: Add optional `tabular_plus_thumbnail` AutoMM path**

Use ordered image columns:
- `frame_01_thumb_path`
- `frame_02_thumb_path`
- `frame_03_thumb_path`
- `frame_04_thumb_path`
- `frame_05_thumb_path`

Do not collapse these into an unordered bag.

- [ ] **Step 5: Verify**

Run: `uv run pytest scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git commit -m "Implement ML verifier AutoGluon training"
```

## Task 5: Replace eval scaffold with real split-aware evaluation

**Files:**
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing tests for real eval inputs**

Cover:
- load saved model artifacts
- load candidate CSV
- join split manifest
- evaluate on held-out split only
- report:
  - `threshold_policy`
  - `final_boundary_threshold`
  - verifier metrics
  - GUI-cost metrics

Run: `uv run pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: FAIL because current eval is still scaffold truth-replay.

- [ ] **Step 2: Implement real predictor loading and held-out prediction**

Eval should:
- load trained predictor A and predictor B
- evaluate on validation/test split rows
- keep breakdown by ground-truth `segment_type`

- [ ] **Step 3: Compute operational metrics from real predictions**

Required outputs:
- `boundary_f1`
- `segment_type_accuracy` or equivalent multiclass metric
- `merge_run_count`
- `split_run_count`
- `estimated_correction_actions`

Review-cost metrics must be computed per `day_id` sequence, then aggregated.

- [ ] **Step 4: Keep scaffold mode only as explicit fallback**

If scaffold mode is preserved, it must be:
- explicit in CLI or metadata
- not the default once real training exists

- [ ] **Step 5: Verify**

Run: `uv run pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git commit -m "Implement ML verifier evaluation"
```

## Task 6: Add `.vocatio` / workspace-aware defaults to train, eval, and split-manifest flow

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing tests for `DAY`-mode train/eval**

Cover:
- `train_ml_boundary_verifier.py DAY --output-dir ...`
- `evaluate_ml_boundary_verifier.py DAY --model-dir ... --output-dir ...`
- `build_ml_boundary_split_manifest.py ...` can resolve output location through `.vocatio` when the command is day-rooted
- workspace resolution through:
  1. `--workspace-dir`
  2. `.vocatio` `WORKSPACE_DIR`
  3. `DAY/_workspace`

Run: `uv run pytest scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: FAIL on missing `DAY`-mode support.

- [ ] **Step 2: Implement path-resolution helpers**

Make train/eval able to infer:
- candidate CSV
- validation report
- split manifest
- model output roots

from `DAY` and workspace.

Do the same for split-manifest outputs when the operator uses a day-rooted invocation.

- [ ] **Step 3: Verify**

Run: `uv run pytest scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git commit -m "Add ML verifier workspace defaults"
```

## Task 7: Document the real end-to-end operator flow

**Files:**
- Modify: `README.md`
- Modify: `scripts/pipeline/README.md`

- [ ] **Step 1: Update docs to show the real first step**

The documented flow must start with:
- `export_ml_boundary_reviewed_truth.py DAY`

Then:
- `build_ml_boundary_candidate_dataset.py DAY`
- `validate_ml_boundary_dataset.py DAY`
- split-manifest generation
- train
- eval

- [ ] **Step 2: Document `uv` group usage clearly**

Show:
- dataset/export/validation in default env
- train/eval under:
  - `uv run --no-default-groups --group autogluon ...`

- [ ] **Step 3: Verify help-text and docs match**

Run:
- `python3 scripts/pipeline/export_ml_boundary_reviewed_truth.py --help`
- `python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py --help`
- `python3 scripts/pipeline/validate_ml_boundary_dataset.py --help`
- `python3 scripts/pipeline/build_ml_boundary_split_manifest.py --help`
- `python3 scripts/pipeline/train_ml_boundary_verifier.py --help`
- `python3 scripts/pipeline/evaluate_ml_boundary_verifier.py --help`

Expected: help text matches README examples.

- [ ] **Step 4: Commit**

```bash
git add README.md scripts/pipeline/README.md
git commit -m "Document ML verifier end-to-end flow"
```

## Task 8: Final verification on the complete real flow

**Files:**
- Modify: none unless fixes are required

- [ ] **Step 1: Run the full focused test suite**

Run:

```bash
uv run pytest scripts/pipeline/test_ml_boundary_truth.py scripts/pipeline/test_ml_boundary_dataset.py scripts/pipeline/test_ml_boundary_features.py scripts/pipeline/test_ml_boundary_review_truth_export.py scripts/pipeline/test_export_ml_boundary_reviewed_truth.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py scripts/pipeline/test_build_ml_boundary_split_manifest.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v
```

Expected: PASS.

- [ ] **Step 2: Run `py_compile` on the complete ML flow**

Run:

```bash
python3 -m py_compile scripts/pipeline/export_ml_boundary_reviewed_truth.py scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/build_ml_boundary_split_manifest.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/lib/ml_boundary_review_truth_export.py scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/lib/ml_boundary_truth.py scripts/pipeline/lib/ml_boundary_dataset.py scripts/pipeline/lib/ml_boundary_features.py
```

Expected: no output, exit code 0.

- [ ] **Step 3: Run the operator smoke chain on one small reviewed day**

Run in order:

```bash
python3 scripts/pipeline/export_ml_boundary_reviewed_truth.py DAY
python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py DAY
python3 scripts/pipeline/validate_ml_boundary_dataset.py DAY
python3 scripts/pipeline/build_ml_boundary_split_manifest.py ...
uv run --no-default-groups --group autogluon python3 scripts/pipeline/train_ml_boundary_verifier.py ...
uv run --no-default-groups --group autogluon python3 scripts/pipeline/evaluate_ml_boundary_verifier.py ...
```

Expected:
- reviewed truth CSV exists
- candidate dataset exists
- validation report exists
- trained model artifacts exist
- evaluation metrics exist

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add README.md scripts/pipeline/*.py scripts/pipeline/lib/*.py scripts/pipeline/test_*.py
git commit -m "Finish ML boundary verifier end-to-end flow"
```

## Self-Review

### Spec coverage

This plan covers the missing pieces that currently prevent real end-to-end use:
- missing reviewed-truth export
- missing explicit split-manifest generation
- scaffold-only train
- scaffold-only eval
- missing short operational `DAY`/workspace flow

It deliberately does not expand scope to:
- new GUI editing features
- replacing the candidate generator
- Gemma 4 tuning

### Placeholder scan

No `TODO`, `TBD`, or “implement later” placeholders remain. Every task names exact files, tests, and commands.

### Type consistency

The plan keeps the current canonical task and artifacts consistent:
- `ml_boundary_reviewed_truth.csv`
- `ml_boundary_candidates.csv`
- `ml_boundary_attrition.json`
- `ml_boundary_validation_report.json`
- `ml_boundary_splits.csv`
- train predictors:
  - `segment_type`
  - `boundary`
