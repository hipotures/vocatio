# Left/Right Segment Type Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public `segment_type` predictor contract with three explicit predictors: `left_segment_type`, `right_segment_type`, and `boundary`.

**Architecture:** Reuse the existing multiclass `segment_type` pipeline as the template for both segment-side predictors, renaming the current path to `right_segment_type` and adding a parallel `left_segment_type` path with the same features and splits. Then propagate the explicit three-predictor contract through training artifacts, evaluation, probe ML hints, GUI index precompute, and manual GUI prediction with no public `segment_type` alias left behind.

**Tech Stack:** Python 3, polars, JSON/CSV artifacts, AutoGluon predictors, PySide6 GUI, pytest/unittest script tests.

---

## File Map

### Existing files to modify
- `scripts/pipeline/lib/ml_boundary_training_data.py`
  - base training bundle contract
  - required columns
  - predictor payload assembly
  - `feature_columns.json` sections
- `scripts/pipeline/lib/ml_boundary_metrics.py`
  - predictor metric registry currently keyed by `segment_type`
- `scripts/pipeline/train_ml_boundary_verifier.py`
  - predictor training loop
  - model directory names
  - training summary payload keys
- `scripts/pipeline/evaluate_ml_boundary_verifier.py`
  - evaluation loop
  - metric payload keys
  - confusion matrix naming
- `scripts/pipeline/run_ml_boundary_pipeline.py`
  - pipeline summary rendering and report payload wiring
- `scripts/pipeline/validate_ml_boundary_dataset.py`
  - currently still validates legacy `segment_type == right_segment_type`
- `scripts/pipeline/lib/ml_boundary_truth.py`
  - current truth object still exposes a single `segment_type`
- `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - ML hint context loading
  - prediction result dataclass / rendering
  - prompt hint text
- `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
  - GUI precomputed ML hint payload shape
- `scripts/pipeline/review_performance_proxy_gui.py`
  - manual ML prediction rendering in the info panel

### Existing tests to modify
- `scripts/pipeline/test_ml_boundary_training_data.py`
- `scripts/pipeline/test_train_ml_boundary_verifier.py`
- `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/test_run_ml_boundary_pipeline.py`
- `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

### Optional existing tests to extend if needed
- `scripts/pipeline/test_validate_ml_boundary_dataset.py`

---

### Task 1: Replace the training bundle contract with left/right segment predictors

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Modify: `scripts/pipeline/lib/ml_boundary_metrics.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`

- [ ] **Step 1: Write the failing training-bundle tests**

Add tests to `scripts/pipeline/test_ml_boundary_training_data.py` that require the explicit three-predictor contract:

```python
def test_load_training_data_bundle_exposes_left_right_and_boundary_predictors(tmp_path: Path):
    bundle = ml_boundary_training_data.load_training_data_bundle(
        candidate_csv_path=_write_candidate_csv(tmp_path),
        split_csv_path=_write_split_csv(tmp_path),
        mode="tabular_only",
    )

    assert bundle.left_segment_type.label_column == "left_segment_type"
    assert bundle.right_segment_type.label_column == "right_segment_type"
    assert bundle.boundary.label_column == "boundary"


def test_feature_columns_manifest_uses_left_and_right_keys(tmp_path: Path):
    bundle = ml_boundary_training_data.load_training_data_bundle(
        candidate_csv_path=_write_candidate_csv(tmp_path),
        split_csv_path=_write_split_csv(tmp_path),
        mode="tabular_only",
    )

    columns = bundle.feature_columns_by_mode["tabular_only"]
    assert "left_segment_type_feature_columns" in columns
    assert "right_segment_type_feature_columns" in columns
    assert "segment_type_feature_columns" not in columns


def test_predictor_feature_sets_match_for_left_and_right(tmp_path: Path):
    bundle = ml_boundary_training_data.load_training_data_bundle(
        candidate_csv_path=_write_candidate_csv(tmp_path),
        split_csv_path=_write_split_csv(tmp_path),
        mode="tabular_only",
    )

    assert bundle.left_segment_type.feature_columns == bundle.right_segment_type.feature_columns
```

- [ ] **Step 2: Run the targeted training-data tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -q
```

Expected:
- FAIL because the bundle still exposes `segment_type` instead of explicit left/right predictors.

- [ ] **Step 3: Implement the new predictor bundle contract**

Update `scripts/pipeline/lib/ml_boundary_training_data.py` so the public bundle changes from:

```python
@dataclass(frozen=True)
class TrainingDataBundle:
    segment_type: PredictorTrainingData
    boundary: PredictorTrainingData
```

to:

```python
@dataclass(frozen=True)
class TrainingDataBundle:
    left_segment_type: PredictorTrainingData
    right_segment_type: PredictorTrainingData
    boundary: PredictorTrainingData
```

Then update the manifest payload and bundle assembly:

```python
columns_by_mode[mode] = {
    "left_segment_type_feature_columns": list(predictor_feature_columns),
    "right_segment_type_feature_columns": list(predictor_feature_columns),
    "boundary_feature_columns": list(predictor_feature_columns),
    "shared_feature_columns": list(shared_feature_columns),
    "image_feature_columns": list(image_feature_columns),
}
```

and:

```python
return TrainingDataBundle(
    left_segment_type=PredictorTrainingData(
        label_column="left_segment_type",
        feature_columns=left_segment_type_feature_columns,
        train_data=train_rows.select(left_segment_type_feature_columns + ["left_segment_type"]),
        validation_data=validation_rows.select(left_segment_type_feature_columns + ["left_segment_type"]),
        test_data=test_rows.select(left_segment_type_feature_columns + ["left_segment_type"]),
    ),
    right_segment_type=PredictorTrainingData(
        label_column="right_segment_type",
        feature_columns=right_segment_type_feature_columns,
        train_data=train_rows.select(right_segment_type_feature_columns + ["right_segment_type"]),
        validation_data=validation_rows.select(right_segment_type_feature_columns + ["right_segment_type"]),
        test_data=test_rows.select(right_segment_type_feature_columns + ["right_segment_type"]),
    ),
    boundary=PredictorTrainingData(
        label_column="boundary",
        feature_columns=boundary_feature_columns,
        train_data=train_rows.select(boundary_feature_columns + ["boundary"]),
        validation_data=validation_rows.select(boundary_feature_columns + ["boundary"]),
        test_data=test_rows.select(boundary_feature_columns + ["boundary"]),
    ),
)
```

Update `scripts/pipeline/lib/ml_boundary_metrics.py` so the metric registry explicitly contains:

```python
PREDICTOR_METRIC_SPECS = {
    "left_segment_type": PredictorMetricSpec(...),
    "right_segment_type": PredictorMetricSpec(...),
    "boundary": PredictorMetricSpec(...),
}
```

- [ ] **Step 4: Run the targeted training-data tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -q
```

Expected:
- PASS with the new left/right predictor bundle.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/lib/ml_boundary_metrics.py scripts/pipeline/test_ml_boundary_training_data.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: split segment predictors by side"
```

### Task 2: Train and evaluate three explicit predictors

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/validate_ml_boundary_dataset.py`
- Modify: `scripts/pipeline/lib/ml_boundary_truth.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_validate_ml_boundary_dataset.py`

- [ ] **Step 1: Write the failing training/evaluation tests**

Add tests requiring explicit left/right artifacts and metrics:

```python
def test_train_model_writes_left_right_and_boundary_directories(tmp_path: Path):
    train_ml_boundary_verifier.train_ml_boundary_verifier(
        candidate_csv_path=_candidate_csv(tmp_path),
        split_csv_path=_split_csv(tmp_path),
        output_dir=tmp_path / "models",
        mode="tabular_only",
    )

    assert (tmp_path / "models" / "left_segment_type_model").is_dir()
    assert (tmp_path / "models" / "right_segment_type_model").is_dir()
    assert (tmp_path / "models" / "boundary_model").is_dir()


def test_feature_columns_json_has_no_segment_type_key(tmp_path: Path):
    payload = _train_and_read_feature_columns(tmp_path)
    assert "left_segment_type_feature_columns" in payload
    assert "right_segment_type_feature_columns" in payload
    assert "segment_type_feature_columns" not in payload


def test_evaluation_metrics_include_left_and_right_predictors(tmp_path: Path):
    metrics = evaluate_ml_boundary_verifier.evaluate_ml_boundary_verifier(...)
    assert "left_segment_type_macro_f1" in metrics
    assert "right_segment_type_macro_f1" in metrics
    assert "segment_type_macro_f1" not in metrics
```

Add validator/truth tests:

```python
def test_validator_accepts_distinct_left_and_right_segment_types():
    row = _valid_candidate_row()
    row["left_segment_type"] = "performance"
    row["right_segment_type"] = "ceremony"
    validate_ml_boundary_dataset.validate_candidate_row(row, row_number=2)


def test_truth_loader_reads_left_and_right_segment_types():
    truth = ml_boundary_truth.load_truth_row({
        "left_segment_type": "performance",
        "right_segment_type": "ceremony",
        "boundary": "1",
    })
    assert truth.left_segment_type == "performance"
    assert truth.right_segment_type == "ceremony"
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_validate_ml_boundary_dataset.py -q
```

Expected:
- FAIL because training/evaluation still use the legacy public `segment_type` predictor.

- [ ] **Step 3: Implement explicit left/right training and evaluation**

Update `scripts/pipeline/train_ml_boundary_verifier.py` so the predictor loop becomes:

```python
predictor_training_data = {
    "left_segment_type": bundle.left_segment_type,
    "right_segment_type": bundle.right_segment_type,
    "boundary": bundle.boundary,
}
```

and the model directories become:

```python
model_dir = output_dir / f"{predictor_name}_model"
```

Update `scripts/pipeline/evaluate_ml_boundary_verifier.py` so the evaluation loop iterates over the same explicit names and writes payload keys like:

```python
metrics[f"{predictor_name}_macro_f1"] = macro_f1
metrics[f"{predictor_name}_accuracy"] = accuracy
metrics[f"{predictor_name}_correct_count"] = correct_count
metrics[f"{predictor_name}_incorrect_count"] = incorrect_count
metrics[f"{predictor_name}_confusion_matrix"] = confusion_matrix
```

Update `scripts/pipeline/validate_ml_boundary_dataset.py` to stop enforcing:

```python
if segment_type != right_segment_type:
    raise ValueError(...)
```

and instead validate both side columns directly as first-class labels.

Update `scripts/pipeline/lib/ml_boundary_truth.py` from a single field:

```python
segment_type: str
```

to:

```python
left_segment_type: str
right_segment_type: str
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_validate_ml_boundary_dataset.py -q
```

Expected:
- PASS with explicit left/right models and metrics.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/lib/ml_boundary_truth.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py scripts/pipeline/test_validate_ml_boundary_dataset.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: train and evaluate left and right segment models"
```

### Task 3: Replace pipeline/report wiring with the explicit three-predictor contract

**Files:**
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write the failing pipeline summary tests**

Add tests that require the summary payload to expose left/right keys and reject the old public key:

```python
def test_pipeline_summary_uses_left_and_right_metric_keys(tmp_path: Path):
    summary = _run_pipeline_and_read_summary(tmp_path)
    metrics = summary["evaluation_metrics"]
    assert "left_segment_type_macro_f1" in metrics
    assert "right_segment_type_macro_f1" in metrics
    assert "segment_type_macro_f1" not in metrics


def test_console_summary_renders_all_three_predictors(capsys):
    run_ml_boundary_pipeline.render_final_summary(
        evaluation_metrics={
            "left_segment_type_macro_f1": 0.61,
            "right_segment_type_macro_f1": 0.72,
            "boundary_f1": 0.85,
        },
        ...
    )
    text = capsys.readouterr().out
    assert "Left segment type" in text
    assert "Right segment type" in text
    assert "Boundary" in text
```

- [ ] **Step 2: Run the targeted pipeline tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected:
- FAIL because the pipeline summary still depends on `segment_type_*` keys.

- [ ] **Step 3: Update pipeline summaries and metric lookups**

Modify `scripts/pipeline/run_ml_boundary_pipeline.py` so any old lookup like:

```python
evaluation_metrics["segment_type_macro_f1"]
```

is replaced with explicit rendering for both segment-side predictors:

```python
left_macro_f1 = evaluation_metrics["left_segment_type_macro_f1"]
right_macro_f1 = evaluation_metrics["right_segment_type_macro_f1"]
boundary_f1 = evaluation_metrics["boundary_f1"]
```

and render summary text in the same explicit order:

```python
console.print(f"Left segment type: macro_f1={left_macro_f1:.4f}, accuracy={left_accuracy:.4f}, ...")
console.print(f"Right segment type: macro_f1={right_macro_f1:.4f}, accuracy={right_accuracy:.4f}, ...")
console.print(f"Boundary: f1={boundary_f1:.4f}, ...")
```

- [ ] **Step 4: Run the targeted pipeline tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected:
- PASS with left/right segment metrics wired through the pipeline summary and report payloads.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: report left and right segment predictors"
```

### Task 4: Expose left and right segment hints in probe and GUI index precompute

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Modify: `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`

- [ ] **Step 1: Write the failing downstream-hint tests**

Add tests requiring explicit left/right predictions in probe and GUI precompute:

```python
def test_predict_ml_hint_for_candidate_returns_left_right_and_boundary():
    prediction = probe.predict_ml_hint_for_candidate(...)
    assert prediction.left_segment_type_prediction == "performance"
    assert prediction.right_segment_type_prediction == "ceremony"
    assert prediction.boundary_prediction == "cut"


def test_build_ml_hint_lines_for_candidate_renders_three_independent_lines():
    lines = probe.build_ml_hint_lines_for_candidate(...)
    assert any("Left-side segment" in line for line in lines)
    assert any("Right-side segment" in line for line in lines)
    assert any("Boundary" in line for line in lines)


def test_build_gui_index_serializes_left_and_right_ml_predictions():
    payload = build_vlm_photo_boundary_gui_index.build_gui_index_for_run(...)
    ml_pair = payload["ml_hint_pairs"][0]
    assert ml_pair["left_segment_type_prediction"] == "performance"
    assert ml_pair["right_segment_type_prediction"] == "ceremony"
```

- [ ] **Step 2: Run the targeted probe/index tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py -q
```

Expected:
- FAIL because runtime prediction/result payloads still expose only one segment-side model.

- [ ] **Step 3: Implement explicit left/right ML hint payloads**

Update `scripts/pipeline/probe_vlm_photo_boundaries.py` so the ML prediction result dataclass changes from one segment-side prediction:

```python
segment_type_prediction: str
segment_type_confidence: float
```

to:

```python
left_segment_type_prediction: str
left_segment_type_confidence: float
right_segment_type_prediction: str
right_segment_type_confidence: float
boundary_prediction: str
boundary_confidence: float
```

Load predictors using explicit model directories:

```python
left_segment_type_model_dir = ml_model_dir / "left_segment_type_model"
right_segment_type_model_dir = ml_model_dir / "right_segment_type_model"
boundary_model_dir = ml_model_dir / "boundary_model"
```

and render three independent lines:

```python
lines = [
    f"ML hint for the main candidate gap in this window: likely {boundary_label} (confidence {boundary_confidence:.2f}).",
    f"ML hint for the left side of the candidate gap: likely {left_label} (confidence {left_confidence:.2f}).",
    f"ML hint for the right side of the candidate gap: likely {right_label} (confidence {right_confidence:.2f}).",
]
```

Then update `scripts/pipeline/build_vlm_photo_boundary_gui_index.py` to emit:

```python
{
    "boundary_prediction": prediction.boundary_prediction,
    "boundary_confidence": f"{prediction.boundary_confidence:.2f}",
    "left_segment_type_prediction": prediction.left_segment_type_prediction,
    "left_segment_type_confidence": f"{prediction.left_segment_type_confidence:.2f}",
    "right_segment_type_prediction": prediction.right_segment_type_prediction,
    "right_segment_type_confidence": f"{prediction.right_segment_type_confidence:.2f}",
}
```

- [ ] **Step 4: Run the targeted probe/index tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py -q
```

Expected:
- PASS with explicit left/right/downstream ML hint payloads.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/build_vlm_photo_boundary_gui_index.py scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: expose left and right ML hint predictions"
```

### Task 5: Update manual GUI prediction to show boundary + left + right explicitly

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing GUI diagnostics tests**

Add GUI diagnostics tests for the manual ML prediction panel:

```python
def test_manual_ml_prediction_section_renders_left_right_and_boundary(self):
    state = window._format_manual_ml_prediction_state(
        {
            "status": "result",
            "boundary_prediction": "cut",
            "boundary_confidence": 0.84,
            "left_segment_type_prediction": "performance",
            "left_segment_type_confidence": 0.71,
            "right_segment_type_prediction": "ceremony",
            "right_segment_type_confidence": 0.93,
        }
    )
    self.assertIn("Boundary: cut (0.84)", state)
    self.assertIn("Left-side segment: performance (0.71)", state)
    self.assertIn("Right-side segment: ceremony (0.93)", state)


def test_manual_ml_prediction_section_no_longer_uses_legacy_segment_type_label(self):
    state = window._format_manual_ml_prediction_state(
        {
            "status": "result",
            "boundary_prediction": "no_cut",
            "boundary_confidence": 0.52,
            "left_segment_type_prediction": "performance",
            "left_segment_type_confidence": 0.99,
            "right_segment_type_prediction": "performance",
            "right_segment_type_confidence": 0.99,
        }
    )
    self.assertNotIn("Right-side segment:", state.replace("Left-side segment:", ""))
```

- [ ] **Step 2: Run the targeted GUI diagnostics tests and verify failure**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- FAIL because manual ML prediction formatting still expects only one segment-side prediction.

- [ ] **Step 3: Update GUI rendering to show explicit left/right predictions**

Modify `scripts/pipeline/review_performance_proxy_gui.py` so the manual ML result formatter renders:

```python
result_lines = [
    f"Boundary: {payload['boundary_prediction']} ({payload['boundary_confidence']:.2f})",
    f"Left-side segment: {payload['left_segment_type_prediction']} ({payload['left_segment_type_confidence']:.2f})",
    f"Right-side segment: {payload['right_segment_type_prediction']} ({payload['right_segment_type_confidence']:.2f})",
    f"Gap seconds: {payload['gap_seconds']}",
    f"Anchors: {payload['left_anchor']} -> {payload['right_anchor']}",
]
```

and ensure any code that reads precomputed index hints or manual runtime results uses the explicit left/right keys and never the public `segment_type_*` keys.

- [ ] **Step 4: Run the targeted GUI diagnostics tests to verify they pass**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- PASS with explicit left/right GUI rendering.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: show left and right segment predictions in GUI"
```

### Task 6: Run the full regression for the three-predictor migration

**Files:**
- Modify if needed after failures: any of the files above
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Run the full regression suite**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- PASS with no remaining public `segment_type` predictor contract in those flows.

- [ ] **Step 2: Fix any remaining stale public keys**

If the regression finds leftover references, remove them in the owning file. Typical patterns to delete or rename:

```python
"segment_type_feature_columns"
"segment_type_macro_f1"
"segment_type_accuracy"
"segment_type_confusion_matrix"
"segment_type_prediction"
"segment_type_confidence"
```

Replace each with the explicit left/right forms already introduced in earlier tasks.

- [ ] **Step 3: Re-run the full regression suite**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- PASS again after any cleanup fix.

- [ ] **Step 4: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/lib/ml_boundary_metrics.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/lib/ml_boundary_truth.py scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/build_vlm_photo_boundary_gui_index.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py scripts/pipeline/test_run_ml_boundary_pipeline.py scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: migrate ML boundary verifier to left and right segment predictors"
```

## Self-Review

- Spec coverage:
  - predictor contract rename/replacement: covered by Tasks 1-2
  - artifact and metadata changes: covered by Tasks 1-3
  - evaluation contract: covered by Task 2
  - probe/prompt/GUI contract: covered by Tasks 4-5
  - no public alias / no stale keys: covered by Task 6
- Placeholder scan:
  - no `TODO`, `TBD`, “similar to Task N”, or unspecified “write tests” placeholders remain.
- Type consistency:
  - the plan uses the same public predictor names everywhere:
    - `left_segment_type`
    - `right_segment_type`
    - `boundary`
  - no later task reintroduces public `segment_type`.
