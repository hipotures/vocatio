# ML Boundary Training Preset And Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared `--preset` and `--train-minutes` training controls that apply equally to both ML predictors, preserve current defaults, and surface the resolved options in CLI output and JSON artifacts.

**Architecture:** Introduce one small shared training-options module that resolves defaults and converts minutes to per-predictor `time_limit` seconds. Wire that contract into `train_ml_boundary_verifier.py`, `run_ml_boundary_pipeline.py`, and the persisted training/pipeline artifacts so configuration and reporting stay in sync.

**Tech Stack:** Python 3, argparse, AutoGluon, rich, pytest

---

### Task 1: Add Shared Training Option Contract

**Files:**
- Create: `scripts/pipeline/lib/ml_boundary_training_options.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write the failing test**

Add a focused unit test block to `scripts/pipeline/test_train_ml_boundary_verifier.py` that imports the new helper functions and asserts the default and explicit option resolution:

```python
from lib.ml_boundary_training_options import (
    DEFAULT_TRAINING_PRESET,
    resolve_training_options,
)


def test_resolve_training_options_uses_current_defaults() -> None:
    assert DEFAULT_TRAINING_PRESET == "medium_quality"
    assert resolve_training_options(preset=None, train_minutes=None) == {
        "training_preset": "medium_quality",
        "train_minutes": None,
        "time_limit_seconds": None,
    }


def test_resolve_training_options_converts_minutes_to_seconds() -> None:
    assert resolve_training_options(preset="best_quality", train_minutes=10) == {
        "training_preset": "best_quality",
        "train_minutes": 10.0,
        "time_limit_seconds": 600,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import failure for `lib.ml_boundary_training_options`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/pipeline/lib/ml_boundary_training_options.py` with one shared contract:

```python
from __future__ import annotations


DEFAULT_TRAINING_PRESET = "medium_quality"


def resolve_training_options(
    *,
    preset: str | None,
    train_minutes: float | None,
) -> dict[str, object]:
    resolved_preset = str(preset or DEFAULT_TRAINING_PRESET).strip()
    if not resolved_preset:
        raise ValueError("training preset must not be blank")
    if train_minutes is None:
        return {
            "training_preset": resolved_preset,
            "train_minutes": None,
            "time_limit_seconds": None,
        }
    minutes_value = float(train_minutes)
    if minutes_value <= 0:
        raise ValueError("train_minutes must be greater than zero")
    return {
        "training_preset": resolved_preset,
        "train_minutes": minutes_value,
        "time_limit_seconds": int(minutes_value * 60),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: PASS for the new helper tests.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_options.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add shared ML training options"
```


### Task 2: Extend Training CLI And Predictor Fit Settings

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write the failing test**

Extend `scripts/pipeline/test_train_ml_boundary_verifier.py` so the existing real-artifact training test asserts:

```python
assert training_plan["training_preset"] == "medium_quality"
assert training_plan["train_minutes"] is None
assert training_plan["time_limit_seconds"] is None
assert training_report["training_preset"] == "medium_quality"
assert training_report["train_minutes"] is None
assert training_report["time_limit_seconds"] is None
assert FakeTabularPredictor.instances[0].fit_calls[0]["presets"] == "medium_quality"
assert FakeTabularPredictor.instances[0].fit_calls[0]["time_limit"] is None
assert FakeTabularPredictor.instances[1].fit_calls[0]["presets"] == "medium_quality"
assert FakeTabularPredictor.instances[1].fit_calls[0]["time_limit"] is None
```

Add a second test that calls the CLI with explicit options:

```python
exit_code = main(
    [
        str(dataset_path),
        "--split-manifest-csv",
        str(split_manifest_path),
        "--mode",
        "tabular_only",
        "--output-dir",
        str(output_dir),
        "--preset",
        "best_quality",
        "--train-minutes",
        "10",
    ]
)

assert exit_code == 0
assert FakeTabularPredictor.instances[0].fit_calls[0]["presets"] == "best_quality"
assert FakeTabularPredictor.instances[0].fit_calls[0]["time_limit"] == 600
assert FakeTabularPredictor.instances[1].fit_calls[0]["presets"] == "best_quality"
assert FakeTabularPredictor.instances[1].fit_calls[0]["time_limit"] == 600
```

Also assert the rendered console contains:

```python
assert "preset=best_quality" in rendered
assert "time_limit_seconds=600" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: FAIL because `train_ml_boundary_verifier.py` does not yet accept `--preset` or `--train-minutes`, does not persist the options, and does not pass `time_limit`.

- [ ] **Step 3: Write minimal implementation**

Update `scripts/pipeline/train_ml_boundary_verifier.py`:

1. Add CLI args:

```python
parser.add_argument(
    "--preset",
    default=None,
    help="AutoGluon preset passed through to each predictor. Default: medium_quality",
)
parser.add_argument(
    "--train-minutes",
    type=float,
    default=None,
    help="Optional training time limit in minutes applied separately to each predictor.",
)
```

2. Resolve options once in `main()`:

```python
from lib.ml_boundary_training_options import resolve_training_options

training_options = resolve_training_options(
    preset=args.preset,
    train_minutes=args.train_minutes,
)
```

3. Thread `training_options` through:
- `_build_training_plan_payload(...)`
- `_build_training_report_payload(...)`
- `_train_predictors(...)`
- `_fit_predictor(...)`
- `_final_console_block(...)`

4. Persist:

```python
"training_preset": training_options["training_preset"],
"train_minutes": training_options["train_minutes"],
"time_limit_seconds": training_options["time_limit_seconds"],
```

5. Use the resolved values in predictor fit:

```python
predictor.fit(
    _to_model_frame(predictor_data.train_data),
    tuning_data=_to_model_frame(predictor_data.validation_data),
    presets=str(training_options["training_preset"]),
    time_limit=training_options["time_limit_seconds"],
)
```

Apply the same `presets` and `time_limit` to the multimodal branch.

6. Show them in the console block:

```python
(
    "Training options: "
    f"preset={training_options['training_preset']}, "
    f"time_limit_seconds={training_options['time_limit_seconds']}"
)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: PASS, including the new explicit-preset/time-limit assertions.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add ML training preset and time limit CLI"
```


### Task 3: Forward Options Through Pipeline Runner

**Files:**
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write the failing test**

Extend the end-to-end pipeline test in `scripts/pipeline/test_run_ml_boundary_pipeline.py` to call:

```python
exit_code = main(
    [
        str(day_dir),
        "--mode",
        "tabular_only",
        "--model-run-id",
        "day-20260323",
        "--preset",
        "best_quality",
        "--train-minutes",
        "10",
    ]
)
```

Assert the recorded training command contains:

```python
assert "--preset" in train_command
assert "best_quality" in train_command
assert "--train-minutes" in train_command
assert "10" in train_command
```

Assert the pipeline summary records:

```python
assert summary_payload["training_preset"] == "best_quality"
assert summary_payload["train_minutes"] == 10.0
assert summary_payload["time_limit_seconds"] == 600
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: FAIL because the pipeline runner does not yet parse or forward these options.

- [ ] **Step 3: Write minimal implementation**

Modify `scripts/pipeline/run_ml_boundary_pipeline.py`:

1. Add CLI args:

```python
parser.add_argument(
    "--preset",
    default=None,
    help="AutoGluon preset forwarded to train_ml_boundary_verifier.py.",
)
parser.add_argument(
    "--train-minutes",
    type=float,
    default=None,
    help="Optional per-predictor training limit in minutes forwarded to the training step.",
)
```

2. Forward only when provided:

```python
if args.preset:
    train_command.extend(["--preset", args.preset])
if args.train_minutes is not None:
    train_command.extend(["--train-minutes", str(args.train_minutes)])
```

3. Persist the resolved values in the pipeline summary after training:

```python
if isinstance(training_metadata, dict):
    summary_payload["training_preset"] = training_metadata.get("training_preset")
    summary_payload["train_minutes"] = training_metadata.get("train_minutes")
    summary_payload["time_limit_seconds"] = training_metadata.get("time_limit_seconds")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: PASS, including the forwarded command and summary payload assertions.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: forward ML training options through pipeline"
```


### Task 4: Full Focused Verification And Real CLI Smoke Check

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: PASS for all tests.

- [ ] **Step 2: Run CLI help verification**

Run:

```bash
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/train_ml_boundary_verifier.py --help
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py --help
```

Expected: both help outputs include:

```text
--preset
--train-minutes
```

- [ ] **Step 3: Run one representative real invocation without changing defaults**

Run:

```bash
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py /arch03/V/DWC2026/20260323 --mode tabular_only --model-run-id day-20260323 --restart
```

Expected:
- current default behavior still works
- final summary still prints cleanly
- training artifacts include `training_preset=medium_quality`, `train_minutes=null`, `time_limit_seconds=null`

- [ ] **Step 4: Run one representative configured invocation**

Run:

```bash
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py /arch03/V/DWC2026/20260323 --mode tabular_only --model-run-id day-20260323-best-10m --preset best_quality --train-minutes 10 --restart
```

Expected:
- both predictors run with `best_quality`
- each predictor gets `time_limit=600`
- final artifacts record:
  - `training_preset=best_quality`
  - `train_minutes=10.0`
  - `time_limit_seconds=600`

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_options.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add shared ML preset and time controls"
```
