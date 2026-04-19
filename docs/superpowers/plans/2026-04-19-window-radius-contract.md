# Window Radius Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed `window_size` / `overlap` public contract with a single external `window_radius` contract and make VLM, ML, GUI precompute, and GUI manual prediction use the same symmetric window shape end-to-end.

**Architecture:** Introduce one shared radius-to-window helper contract, then migrate probe CLI/config, ML candidate schema generation, training metadata, and runtime consumers to radius-based validation. Remove hidden recomputation and all compatibility fallbacks so any radius mismatch fails immediately and explicitly.

**Tech Stack:** Python 3, argparse, CSV/JSON artifacts, AutoGluon tabular/multimodal predictors, PySide6 GUI, pytest/unittest script tests.

---

## File Map

### Existing files to modify
- `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - Public probe CLI/config contract
  - Run metadata/config hash fields
  - Candidate window construction
  - ML hint inference helpers currently hardcoded to fixed 5-frame window
- `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
  - Precomputed ML hint generation for GUI index artifacts
- `scripts/pipeline/review_performance_proxy_gui.py`
  - Manual ML prediction runtime path
  - `.vocatio` config loading
  - GUI-side ML metadata validation
- `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
  - Candidate CSV schema and row generation
- `scripts/pipeline/lib/ml_boundary_features.py`
  - Derived tabular feature generation, gap fields, side features
- `scripts/pipeline/lib/ml_boundary_training_data.py`
  - Training bundle loading, feature-column generation, required-column validation
- `scripts/pipeline/train_ml_boundary_verifier.py`
  - Training metadata and feature manifest emission
- `scripts/pipeline/evaluate_ml_boundary_verifier.py`
  - Model metadata loading and validation
- `scripts/pipeline/run_ml_boundary_pipeline.py`
  - Pipeline argument forwarding and summary reporting
- `scripts/pipeline/README.md`
  - Public docs for probe/train flow

### Existing tests to modify
- `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`
- `scripts/pipeline/test_ml_boundary_features.py`
- `scripts/pipeline/test_ml_boundary_training_data.py`
- `scripts/pipeline/test_train_ml_boundary_verifier.py`
- `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/test_run_ml_boundary_pipeline.py`

### New helper file to create
- `scripts/pipeline/lib/window_radius_contract.py`
  - Shared helpers for:
    - parsing/validating `window_radius`
    - computing internal `window_size = 2 * window_radius`
    - deriving symmetric center indices
    - validating model/runtime radius compatibility

---

### Task 1: Add shared radius contract helpers

**Files:**
- Create: `scripts/pipeline/lib/window_radius_contract.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write the failing tests for the shared radius helpers**

Add tests to `scripts/pipeline/test_probe_vlm_photo_boundaries.py` covering:

```python
def test_window_size_for_radius_is_symmetric():
    from lib.window_radius_contract import window_size_for_radius
    assert window_size_for_radius(1) == 2
    assert window_size_for_radius(2) == 4
    assert window_size_for_radius(3) == 6


def test_center_cut_index_for_radius():
    from lib.window_radius_contract import center_cut_index_for_radius
    assert center_cut_index_for_radius(1) == 0
    assert center_cut_index_for_radius(2) == 1
    assert center_cut_index_for_radius(3) == 2


def test_validate_matching_window_radius_allows_equal_values():
    from lib.window_radius_contract import validate_matching_window_radius
    validate_matching_window_radius(runtime_radius=3, artifact_radius=3, artifact_name="ml model")


def test_validate_matching_window_radius_rejects_mismatch():
    from lib.window_radius_contract import validate_matching_window_radius
    with pytest.raises(ValueError, match="window_radius mismatch"):
        validate_matching_window_radius(runtime_radius=3, artifact_radius=2, artifact_name="ml model")
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- FAIL because `lib.window_radius_contract` does not exist yet.

- [ ] **Step 3: Add the shared helper module**

Create `scripts/pipeline/lib/window_radius_contract.py`:

```python
from __future__ import annotations


def positive_window_radius(value: str) -> int:
    radius = int(str(value).strip())
    if radius <= 0:
        raise ValueError("window_radius must be greater than zero")
    return radius


def window_size_for_radius(window_radius: int) -> int:
    if window_radius <= 0:
        raise ValueError("window_radius must be greater than zero")
    return window_radius * 2


def center_cut_index_for_radius(window_radius: int) -> int:
    return window_radius - 1


def validate_matching_window_radius(*, runtime_radius: int, artifact_radius: int, artifact_name: str) -> None:
    if runtime_radius != artifact_radius:
        raise ValueError(
            f"{artifact_name} window_radius mismatch: runtime={runtime_radius}, artifact={artifact_radius}"
        )
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- PASS for the new helper tests.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/window_radius_contract.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add shared window radius helpers"
```

### Task 2: Replace public probe config with `window_radius`

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write failing probe contract tests**

Update probe tests to require the new public contract:

```python
def test_parse_args_accepts_window_radius():
    args = probe.parse_args(["/tmp/day", "--window-radius", "3"])
    assert args.window_radius == 3


def test_apply_vocatio_defaults_reads_window_radius_only(tmp_path: Path):
    day_dir = tmp_path / "20260323"
    day_dir.mkdir()
    (day_dir / ".vocatio").write_text("VLM_WINDOW_RADIUS=3\n", encoding="utf-8")
    args = probe.parse_args([str(day_dir)])
    args = probe.apply_vocatio_defaults(args, day_dir)
    assert args.window_radius == 3


def test_parse_args_rejects_removed_window_size_option():
    with pytest.raises(SystemExit):
        probe.parse_args(["/tmp/day", "--window-size", "6"])


def test_parse_args_rejects_removed_overlap_option():
    with pytest.raises(SystemExit):
        probe.parse_args(["/tmp/day", "--overlap", "3"])
```

- [ ] **Step 2: Run the targeted probe tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- FAIL because probe still exposes `window_size` and `overlap`.

- [ ] **Step 3: Update probe parsing/defaults/metadata to use radius only**

Modify `scripts/pipeline/probe_vlm_photo_boundaries.py` so that:

```python
from lib.window_radius_contract import (
    center_cut_index_for_radius,
    positive_window_radius,
    window_size_for_radius,
)

DEFAULT_WINDOW_RADIUS = 3
```

And in CLI/default handling:

```python
parser.add_argument(
    "--window-radius",
    type=positive_window_radius,
    default=DEFAULT_WINDOW_RADIUS,
    help=f"Number of context frames per side of the main gap. Default: {DEFAULT_WINDOW_RADIUS}",
)
```

And config defaults:

```python
apply_int("window_radius", DEFAULT_WINDOW_RADIUS, "VLM_WINDOW_RADIUS", positive_window_radius)
```

And internal use:

```python
window_radius = args.window_radius
window_size = window_size_for_radius(window_radius)
cut_index = center_cut_index_for_radius(window_radius)
```

And run metadata/config hash payloads must write:

```python
"window_radius": args.window_radius,
```

and must stop writing external `overlap`.

- [ ] **Step 4: Rewrite candidate-window positioning to be symmetric**

Replace overlap-based placement with centered gap placement:

```python
def build_candidate_windows(
    rows: Sequence[Mapping[str, str]],
    window_radius: int,
    boundary_gap_seconds: int,
) -> List[Dict[str, int]]:
    window_size = window_size_for_radius(window_radius)
    cut_local_index = center_cut_index_for_radius(window_radius)
    total_rows = len(rows)
    final_start = total_rows - window_size
    candidates_by_start: Dict[int, Dict[str, int]] = {}
    for cut_index in range(total_rows - 1):
        left_epoch_ms = int(str(rows[cut_index]["start_epoch_ms"]))
        right_epoch_ms = int(str(rows[cut_index + 1]["start_epoch_ms"]))
        time_gap_seconds = rounded_seconds(right_epoch_ms - left_epoch_ms)
        if time_gap_seconds <= boundary_gap_seconds:
            continue
        start_index = cut_index - cut_local_index
        if start_index < 0:
            start_index = 0
        if start_index > final_start:
            start_index = final_start
        candidate = {
            "start_index": start_index,
            "cut_index": cut_index,
            "time_gap_seconds": time_gap_seconds,
        }
        existing = candidates_by_start.get(start_index)
        if existing is None or candidate["time_gap_seconds"] > existing["time_gap_seconds"]:
            candidates_by_start[start_index] = candidate
    return [candidates_by_start[start_index] for start_index in sorted(candidates_by_start)]
```

- [ ] **Step 5: Run probe tests to verify pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- PASS with probe using `window_radius` externally and symmetric placement internally.

- [ ] **Step 6: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "refactor: use window radius in probe flow"
```

### Task 3: Make ML candidate schema dynamic by radius

**Files:**
- Modify: `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
- Modify: `scripts/pipeline/lib/ml_boundary_features.py`
- Test: `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`

- [ ] **Step 1: Write failing schema-generation tests**

Add tests that assert dynamic frame and gap columns:

```python
def test_candidate_headers_for_radius_two():
    headers = dataset.candidate_row_headers(window_radius=2, include_thumbnail=True)
    assert "frame_04_photo_id" in headers
    assert "frame_05_photo_id" not in headers
    assert "gap_34" in headers
    assert "gap_45" not in headers


def test_candidate_headers_for_radius_three():
    headers = dataset.candidate_row_headers(window_radius=3, include_thumbnail=True)
    assert "frame_06_photo_id" in headers
    assert "gap_56" in headers
```

And feature tests:

```python
def test_build_temporal_gap_features_for_radius_three():
    candidate = {
        "frame_01_timestamp": 1.0,
        "frame_02_timestamp": 2.0,
        "frame_03_timestamp": 3.0,
        "frame_04_timestamp": 8.0,
        "frame_05_timestamp": 9.0,
        "frame_06_timestamp": 10.0,
    }
    features = ml_boundary_features.build_gap_features(candidate, window_radius=3)
    assert features["gap_12"] == 1.0
    assert features["gap_34"] == 5.0
    assert features["gap_56"] == 1.0
```

- [ ] **Step 2: Run the dataset/feature tests and verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_ml_boundary_candidate_dataset.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py -q
```

Expected:
- FAIL because headers/features still assume fixed frame count `5`.

- [ ] **Step 3: Replace fixed frame/gap schema generation with radius-derived builders**

In `build_ml_boundary_candidate_dataset.py`, introduce helpers such as:

```python
def frame_numbers_for_radius(window_radius: int) -> list[int]:
    return list(range(1, window_size_for_radius(window_radius) + 1))


def candidate_row_headers(*, window_radius: int, include_thumbnail: bool) -> list[str]:
    headers = [
        "day_id",
        "window_radius",
        "candidate_id",
        "candidate_rule_name",
        "candidate_rule_params_json",
    ]
    for frame_index in frame_numbers_for_radius(window_radius):
        headers.extend(
            [
                f"frame_{frame_index:02d}_photo_id",
                f"frame_{frame_index:02d}_timestamp",
                f"frame_{frame_index:02d}_relpath",
                f"frame_{frame_index:02d}_preview_path",
            ]
        )
        if include_thumbnail:
            headers.append(f"frame_{frame_index:02d}_thumb_path")
    return headers
```

And in `lib/ml_boundary_features.py`, build gap and per-side features from radius-derived frame ranges instead of `frame_05` / `gap_45`.

- [ ] **Step 4: Run the dataset/feature tests to verify pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_ml_boundary_candidate_dataset.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py -q
```

Expected:
- PASS with dynamic frame and gap schema.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_ml_boundary_features.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "refactor: make ML candidate schema radius-driven"
```

### Task 4: Make training bundle and model metadata radius-aware

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing tests for metadata and feature manifests**

Add tests:

```python
def test_training_bundle_reads_window_radius_from_candidates():
    bundle = load_training_data_bundle(...)
    assert bundle.window_radius == 3


def test_training_metadata_records_window_radius(tmp_path: Path):
    metadata = json.loads((output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["window_radius"] == 3


def test_evaluate_rejects_model_window_radius_mismatch():
    with pytest.raises(ValueError, match="window_radius mismatch"):
        evaluate.load_predictors(...)
```

- [ ] **Step 2: Run training/eval tests to verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py -q
```

Expected:
- FAIL because metadata and bundle objects do not yet enforce `window_radius`.

- [ ] **Step 3: Persist and validate `window_radius`**

In `ml_boundary_training_data.py`:

```python
@dataclass(frozen=True)
class TrainingDataBundle:
    window_radius: int
    ...
```

And when loading candidate rows:

```python
window_radius_values = {int(str(row["window_radius"])) for row in rows}
if len(window_radius_values) != 1:
    raise ValueError(f"candidate corpus must contain exactly one window_radius, got {sorted(window_radius_values)}")
window_radius = next(iter(window_radius_values))
```

In `train_ml_boundary_verifier.py`, write:

```python
training_metadata["window_radius"] = bundle.window_radius
```

In `evaluate_ml_boundary_verifier.py`, load the metadata and validate with:

```python
validate_matching_window_radius(
    runtime_radius=current_window_radius,
    artifact_radius=int(training_metadata["window_radius"]),
    artifact_name="ml model",
)
```

- [ ] **Step 4: Run training/eval tests to verify pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py -q
```

Expected:
- PASS with persisted radius metadata and mismatch validation.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: persist ML window radius metadata"
```

### Task 5: Make probe-time ML hint inference radius-aware

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write failing tests for non-default radius ML hint inference**

Add tests:

```python
def test_build_ml_candidate_window_rows_uses_runtime_radius():
    rows = [{"relative_path": f"cam/{i}.jpg"} for i in range(8)]
    result = probe.build_ml_candidate_window_rows(rows, cut_index=3, window_radius=3)
    assert len(result) == 6


def test_predict_ml_hint_rejects_model_radius_mismatch():
    with pytest.raises(ValueError, match="window_radius mismatch"):
        ...
```

- [ ] **Step 2: Run probe tests to verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- FAIL because ML hint inference still uses `ML_BOUNDARY_WINDOW_SIZE = 5`.

- [ ] **Step 3: Replace fixed ML hint window helpers with radius-derived helpers**

In `probe_vlm_photo_boundaries.py`, replace fixed constants and helpers with:

```python
def build_ml_candidate_window_rows(
    joined_rows: Sequence[Mapping[str, str]],
    *,
    cut_index: int,
    window_radius: int,
) -> Optional[list[Mapping[str, str]]]:
    window_size = window_size_for_radius(window_radius)
    cut_local_index = center_cut_index_for_radius(window_radius)
    window_start = cut_index - cut_local_index
    window_end = window_start + window_size
    if window_start < 0 or window_end > len(joined_rows):
        return None
    rows = list(joined_rows[window_start:window_end])
    if len(rows) != window_size:
        return None
    return rows
```

And `_build_ml_candidate_row(...)` must derive frame and gap fields from `window_radius`, not from a fixed `5`.

- [ ] **Step 4: Run probe tests to verify pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q
```

Expected:
- PASS with radius-aware ML hint inference.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "refactor: align ML hint inference with window radius"
```

### Task 6: Fix GUI index precompute and manual GUI prediction to use dynamic radius

**Files:**
- Modify: `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write failing tests for GUI precompute/manual prediction on radius 3**

Add tests:

```python
def test_build_ml_hint_pairs_for_run_supports_vlm_window_radius_three():
    run_metadata = {"args": {"window_radius": 3, "effective_ml_model_run_id": "ml-run-001"}}
    ...
    assert payload["ml_hints_error"] == ""
    assert payload["ml_hint_pairs"]


def test_manual_ml_prediction_uses_vocatio_window_radius():
    ...
    assert result["window_radius"] == 3
```

- [ ] **Step 2: Run GUI builder / GUI diagnostics tests to verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- FAIL because builder and GUI still assume the old fixed-width path.

- [ ] **Step 3: Update GUI precompute and manual prediction to use `window_radius`**

In `build_vlm_photo_boundary_gui_index.py`:

```python
window_radius = int(run_metadata["args"]["window_radius"])
candidate_rows = probe.build_ml_candidate_window_rows(
    joined_rows,
    cut_index=cut_index,
    window_radius=window_radius,
)
```

In `review_performance_proxy_gui.py`:

```python
def resolve_manual_prediction_window_config(vocatio_config: Mapping[str, str]) -> Dict[str, int]:
    window_radius = positive_window_radius(str(vocatio_config.get("VLM_WINDOW_RADIUS", "") or DEFAULT_WINDOW_RADIUS))
    return {"window_radius": window_radius}
```

And manual prediction runtime must pass `window_radius` into candidate window construction and mismatch validation.

- [ ] **Step 4: Run GUI builder / GUI diagnostics tests to verify pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected:
- PASS with dynamic precompute and manual prediction.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/build_vlm_photo_boundary_gui_index.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "fix: use window radius in GUI ML paths"
```

### Task 7: Update pipeline wiring, docs, and full regression coverage

**Files:**
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Modify: `scripts/pipeline/README.md`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write failing pipeline/docs tests**

Add tests that assert:

```python
def test_pipeline_forwards_window_radius():
    ...
    assert "--window-radius" in command


def test_pipeline_summary_uses_window_radius():
    ...
    assert summary["window_radius"] == 3
```

- [ ] **Step 2: Run pipeline tests to verify failure**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected:
- FAIL because old size/overlap wiring is still referenced.

- [ ] **Step 3: Update pipeline/docs**

Modify `run_ml_boundary_pipeline.py` so forwarded commands and summaries use:

```python
["--window-radius", str(args.window_radius)]
```

Update `scripts/pipeline/README.md` examples to use `--window-radius` and `VLM_WINDOW_RADIUS`, and remove mentions of public `VLM_WINDOW_SIZE` / `VLM_OVERLAP`.

- [ ] **Step 4: Run the focused pipeline tests**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected:
- PASS with the updated public contract.

- [ ] **Step 5: Run the cross-cutting regression suite**

Run:

```bash
uv run pytest \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_build_ml_boundary_candidate_dataset.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_evaluate_ml_boundary_verifier.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py \
  -q
```

Expected:
- PASS across probe, ML, GUI, and pipeline wiring.

- [ ] **Step 6: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/README.md scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "refactor: finalize window radius contract"
```

## Self-Review

### Spec coverage
- Public `window_radius` only: covered in Tasks 1, 2, and 7.
- Dynamic ML schema and inference: covered in Tasks 3, 4, and 5.
- GUI precompute/manual prediction: covered in Task 6.
- Fail-fast mismatch rules: covered in Tasks 1, 4, 5, and 6.
- Migration/regression coverage: covered in Task 7.

### Placeholder scan
- No `TODO` / `TBD` placeholders remain in task steps.
- Each code-changing step includes explicit code shape or exact commands.

### Type consistency
- Public contract consistently uses `window_radius`.
- Internal derived `window_size` is described as local-only.
- No task reintroduces public `window_size` or `overlap`.
