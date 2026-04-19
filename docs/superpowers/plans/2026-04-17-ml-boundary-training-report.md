# ML Boundary Training Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `training_report.json` artifact and a concise end-of-run CLI summary block for `train_ml_boundary_verifier.py`.

**Architecture:** Reuse the existing predictor summaries and training bundle metadata to build one short report layer. Persist that layer as `training_report.json` and print the same information once at the end of training.

**Tech Stack:** Python 3, existing training CLI, pathlib/json helpers, pytest-style script tests.

---

## File Structure

- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
  - add the new report artifact filename
  - build the new report payload from existing training summary and training bundle data
  - write the new report JSON
  - print a concise final summary block to stderr
- Modify: `scripts/pipeline/test_train_ml_boundary_verifier.py`
  - verify the new artifact is written
  - verify its contents
  - verify the final CLI block contains the expected information

### Task 1: Add `training_report.json` to Artifact Handling

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Add the new artifact filename and path**

Add:

```python
TRAINING_REPORT_FILENAME = "training_report.json"
```

Update `_artifact_paths(...)` so it returns:

```python
"training_report": output_dir / TRAINING_REPORT_FILENAME
```

- [ ] **Step 2: Add a report payload builder**

Create a helper such as:

```python
def _build_training_report_payload(
    output_dir: Path,
    training_bundle: TrainingDataBundle,
    training_summary: dict[str, object],
    artifact_paths: dict[str, Path],
    mode: str,
) -> dict[str, object]:
```

It should collect:

- `output_dir`
- `mode`
- `split_manifest_scope`
- `train_row_count`
- `validation_row_count`
- `shared_feature_count`
- `image_feature_count`
- `missing_annotation_photo_count`
- `missing_annotation_candidate_count`
- `segment_type.best_model`
- `segment_type.validation_score`
- `segment_type.model_dir`
- `boundary.best_model`
- `boundary.validation_score`
- `boundary.model_dir`
- `artifact_paths`

`best_model` resolution rule:

- use `fit_summary_excerpt["best_model"]` when present
- otherwise fall back to `model_type`

- [ ] **Step 3: Write `training_report.json` during artifact generation**

Inside `main(...)`, after `training_summary.json` is written, add:

```python
atomic_write_json(
    staged_output_dir / TRAINING_REPORT_FILENAME,
    _build_training_report_payload(
        output_dir=output_dir,
        training_bundle=training_bundle,
        training_summary=training_summary,
        artifact_paths=final_artifact_paths,
        mode=args.mode,
    ),
)
```

- [ ] **Step 4: Add a focused artifact test**

In `test_train_ml_boundary_verifier.py`, verify:

- `training_report.json` exists
- it contains the expected top-level counts and predictor summaries
- `artifact_paths["training_report"]` points at the final artifact path

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected:

- report artifact written
- test green

### Task 2: Add the Final CLI Summary Block

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Create a formatter for the final CLI block**

Add a helper such as:

```python
def _training_report_console_lines(report_payload: dict[str, object]) -> list[str]:
```

It should return a short ordered block with lines for:

- output dir
- segment_type best model + validation score
- boundary best model + validation score
- shared/image feature counts
- missing-annotation coverage
- training report path

Example shape:

```text
Training report:
Output dir: ...
segment_type: RandomForestGini (validation_score=0.8864)
boundary: CatBoost (validation_score=0.6957)
Features: shared=52 image=0
Descriptor coverage: missing photos=1355 missing candidates=271
Report JSON: .../training_report.json
```

- [ ] **Step 2: Print the block once after artifacts are published**

After `_publish_staged_output(...)`, print:

```python
for line in _training_report_console_lines(report_payload):
    console.print(line, soft_wrap=True)
```

Keep the existing success lines unless they conflict with readability.

- [ ] **Step 3: Strengthen CLI output assertions**

Extend the existing CLI test to assert that stderr contains:

- the output directory line
- both predictor summary lines
- the feature-count line
- the descriptor coverage line
- the `training_report.json` path line

Make assertions robust to Rich line wrapping by checking substring presence and ordering rather than exact physical line widths.

- [ ] **Step 4: Run the focused CLI test**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected:

- final CLI block present once
- report JSON assertions green

### Task 3: End-to-End Verification

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Run the full targeted automated verification**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected:

- all training-report tests PASS

- [ ] **Step 2: Re-run the real training pipeline**

Run:

```bash
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py /arch03/V/DWC2026/20260323 --mode tabular_only --model-run-id day-20260323 --restart
```

Expected:

- pipeline completes successfully
- `training_report.json` is written under the model output directory
- stderr shows the new concise summary block at the end

- [ ] **Step 3: Inspect the new report artifact**

Run:

```bash
python3 -c "import json; from pathlib import Path; p=Path('/arch03/WORKSPACE/20260323DWC/ml_boundary_corpus/ml_boundary_models/day-20260323/training_report.json'); print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))"
```

Expected:

- report contains best model names, validation scores, feature counts, missing-annotation counts, and artifact paths

- [ ] **Step 4: Commit the completed implementation**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add ML training report artifact"
```
