# ML Boundary Global Corpus Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the ML boundary verifier pipeline to accept from 1 to many day directories, merge all candidate rows into one corpus, and then split that corpus into `train / validation / test` at the candidate-row level with explicit configurable fractions and deterministic shuffling.

**Architecture:** Keep per-day preprocessing exactly as it is now: reviewed truth export, candidate build, day-level validation. Replace the current `day_id -> split_name` orchestration with a global corpus split stage that operates on merged candidate rows. The split manifest becomes candidate-keyed (`candidate_id -> split_name`) and is consumed uniformly by validation, train, and evaluation. Multi-day input remains supported, but days are only data sources, not split units.

**Tech Stack:** Python 3, CLI-first pipeline scripts under `scripts/pipeline/`, CSV manifests, `pathlib.Path`, `rich`, AutoGluon train/eval flow.

---

### Task 1: Redefine Split Strategy Contract Around the Merged Corpus

**Files:**
- Create: `docs/superpowers/plans/2026-04-16-ml-boundary-global-corpus-splits.md`
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Modify: `README.md`
- Modify: `scripts/pipeline/README.md`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write the failing test for single-day corpus split support**

```python
def test_main_single_day_global_random_is_allowed(tmp_path: Path, monkeypatch) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            rows = []
            for index in range(20):
                row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=index + 1)
                row["candidate_id"] = f"c{index:02d}"
                rows.append(row)
            _write_candidate_csv(workspace_dir / "ml_boundary_candidates.csv", rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main([str(day_dir), "--split-strategy", "global_random"])
    assert exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest scripts/pipeline/test_run_ml_boundary_pipeline.py::test_main_single_day_global_random_is_allowed -v`
Expected: FAIL because current runner still assumes `day_id`-level split construction.

- [ ] **Step 3: Replace split-strategy surface in `run_ml_boundary_pipeline.py`**

Use this CLI contract:

```python
parser.add_argument(
    "--split-strategy",
    choices=("global_random", "global_stratified"),
    default=None,
    help="Corpus split strategy after merging candidate rows. Default: global_stratified when possible, else global_random.",
)
parser.add_argument(
    "--train-fraction",
    type=float,
    help="Fraction of merged candidate rows assigned to train. Default: 0.70",
)
parser.add_argument(
    "--validation-fraction",
    type=float,
    help="Fraction of merged candidate rows assigned to validation. Default: 0.15",
)
parser.add_argument(
    "--test-fraction",
    type=float,
    help="Fraction of merged candidate rows assigned to test. Default: 0.15",
)
parser.add_argument(
    "--split-seed",
    type=int,
    help="Deterministic seed for corpus split shuffling. Default: 42",
)
```

Add `.vocatio` equivalents:

```text
ML_SPLIT_STRATEGY=global_stratified
ML_SPLIT_TRAIN_FRACTION=0.70
ML_SPLIT_VALIDATION_FRACTION=0.15
ML_SPLIT_TEST_FRACTION=0.15
ML_SPLIT_SEED=42
```

Rule:

```python
if resolved_split_strategy is None:
    resolved_split_strategy = "global_stratified"
```

- [ ] **Step 4: Remove the 3-day orchestration assumption**

Delete the current logic that stops because fewer than 3 `day_id` values are present, and replace it with corpus-size validation:

```python
if len(merged_rows) < 3:
    raise ValueError(
        "ML boundary corpus split requires at least three candidate rows to produce train, validation, and test splits"
    )
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest scripts/pipeline/test_run_ml_boundary_pipeline.py -v`
Expected: PASS for the new single-day contract after implementation.

- [ ] **Step 6: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_run_ml_boundary_pipeline.py README.md scripts/pipeline/README.md
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "Redefine ML boundary split strategy around merged corpus"
```

### Task 2: Support Candidate-Keyed Split Manifests Everywhere

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Modify: `scripts/pipeline/validate_ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_validate_ml_boundary_dataset.py`

- [ ] **Step 1: Write the failing test for candidate-level split manifests in training**

```python
def test_load_training_data_bundle_accepts_candidate_level_split_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250325", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
            _candidate_row(day_id="20250325", segment_type="warmup", boundary="0"),
        ],
    )
    with split_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "split_name"])
        writer.writeheader()
        writer.writerows(
            [
                {"candidate_id": _candidate_row(day_id="20250325", segment_type="performance", boundary="0")["candidate_id"], "split_name": "train"},
                {"candidate_id": _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1")["candidate_id"], "split_name": "validation"},
                {"candidate_id": _candidate_row(day_id="20250325", segment_type="warmup", boundary="0")["candidate_id"], "split_name": "test"},
            ]
        )

    bundle = load_training_data_bundle(dataset_path, split_manifest_path=split_manifest_path, mode="tabular_only")
    assert bundle.split_counts_by_name == {"train": 1, "validation": 1, "test": 1}
```

- [ ] **Step 2: Write the failing test for candidate-level split manifests in validation**

```python
def test_validate_split_manifest_accepts_candidate_level_assignments() -> None:
    candidate_rows = [
        _candidate_row(day_id="20250325"),
        _candidate_row(day_id="20250325", candidate_rule_version="gap-v2"),
    ]
    split_rows = [
        {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
        {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "validation"},
    ]

    validate_split_manifest(split_rows, candidate_rows)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
- `uv run pytest scripts/pipeline/test_ml_boundary_training_data.py::test_load_training_data_bundle_accepts_candidate_level_split_manifest -v`
- `uv run pytest scripts/pipeline/test_validate_ml_boundary_dataset.py::test_validate_split_manifest_accepts_candidate_level_assignments -v`

Expected: FAIL because current code only supports `day_id`.

- [ ] **Step 4: Extend split manifest schema detection**

Implement schema detection in both files:

```python
if {"candidate_id", "split_name"} <= set(frame.columns):
    manifest_key = "candidate_id"
elif {"day_id", "split_name"} <= set(frame.columns):
    manifest_key = "day_id"
else:
    raise ValueError("split manifest must contain either day_id/split_name or candidate_id/split_name")
```

Update join logic in `ml_boundary_training_data.py`:

```python
if manifest_key == "candidate_id":
    split_by_candidate = {
        str(row["candidate_id"]).strip(): str(row["split_name"]).strip()
        for row in split_manifest_frame.rows
    }
    missing_candidate_ids = sorted(candidate_id for candidate_id in candidate_ids if candidate_id not in split_by_candidate)
    ...
    joined_row["split_name"] = split_by_candidate[row["candidate_id"]]
```

Update validation in `validate_ml_boundary_dataset.py` with the same key-detection and coverage checks.

- [ ] **Step 5: Run focused tests**

Run:
- `uv run pytest scripts/pipeline/test_ml_boundary_training_data.py -v`
- `uv run pytest scripts/pipeline/test_validate_ml_boundary_dataset.py -v`

Expected: PASS, with old `day_id` tests still green and new `candidate_id` tests green.

- [ ] **Step 6: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_validate_ml_boundary_dataset.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "Support candidate-keyed ML boundary split manifests"
```

### Task 3: Implement Global Corpus Split Builder

**Files:**
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write the failing test for deterministic global random split**

```python
def test_main_global_random_writes_candidate_level_split_manifest(tmp_path: Path, monkeypatch) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = []
            for index in range(10):
                row = _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=index + 1)
                row["candidate_id"] = f"{day_dir.name}-c{index:02d}"
                rows.append(row)
            _write_candidate_csv(workspace_dir / "ml_boundary_candidates.csv", rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    corpus_workspace = tmp_path / "corpus"
    exit_code = main(
        [str(day) for day in day_dirs]
        + [
            "--corpus-workspace",
            str(corpus_workspace),
            "--split-strategy",
            "global_random",
            "--train-fraction",
            "0.70",
            "--validation-fraction",
            "0.15",
            "--test-fraction",
            "0.15",
            "--split-seed",
            "42",
            "--prepare-only",
        ]
    )
    assert exit_code == 0

    with (corpus_workspace / "ml_boundary_splits.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0].keys()) == {"candidate_id", "split_name"}
    assert len(rows) == 20
```

- [ ] **Step 2: Write the failing test for default stratified behavior**

```python
def test_main_global_stratified_preserves_boundary_presence_in_all_splits(tmp_path: Path, monkeypatch) -> None:
    ...
    assert {"train", "validation", "test"} == {row["split_name"] for row in split_rows}
    assert each_split_has_boundary_0_and_1
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest scripts/pipeline/test_run_ml_boundary_pipeline.py -v`
Expected: FAIL because current runner still builds day-level manifests.

- [ ] **Step 4: Implement `global_random`**

Add helper:

```python
def _build_global_random_split_rows(
    candidate_rows: Sequence[dict[str, str]],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> list[dict[str, str]]:
    total = train_fraction + validation_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.0")
    if len(candidate_rows) < 3:
        raise ValueError("global_random requires at least three candidate rows")

    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    rng = random.Random(split_seed)
    rng.shuffle(candidate_ids)
```

Then assign counts:

```python
train_count = max(1, int(round(len(candidate_ids) * train_fraction)))
validation_count = max(1, int(round(len(candidate_ids) * validation_fraction)))
test_count = len(candidate_ids) - train_count - validation_count
if test_count < 1:
    test_count = 1
    if train_count > validation_count:
        train_count -= 1
    else:
        validation_count -= 1
```

- [ ] **Step 5: Implement `global_stratified`**

Use a simple, deterministic v1 approach:

```python
strata_key = (
    str(row["boundary"]).strip().lower(),
    str(row["segment_type"]).strip(),
)
```

Algorithm:

```python
group rows by strata_key
shuffle each group with random.Random(split_seed)
split each group proportionally into train/validation/test
merge per-split assignments
sort output rows by candidate_id for stable manifest diffs
```

Fallback:

```python
if any(stratum too small to populate all splits):
    fall back to global_random
```

Record chosen strategy in summary JSON as:

```python
"requested_split_strategy": resolved_split_strategy,
"effective_split_strategy": effective_split_strategy,
"train_fraction": resolved_train_fraction,
"validation_fraction": resolved_validation_fraction,
"test_fraction": resolved_test_fraction,
"split_seed": resolved_split_seed,
```

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest scripts/pipeline/test_run_ml_boundary_pipeline.py -v`
Expected: PASS for single-day and multi-day corpus split runs.

- [ ] **Step 7: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "Implement global corpus ML boundary split builders"
```

### Task 4: Make Train and Eval Consume the New Manifest Style Without Caring About Day Count

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing train/eval tests for candidate-keyed manifests**

Add tests with:

```python
writer = csv.DictWriter(handle, fieldnames=["candidate_id", "split_name"])
```

and assert:

```python
assert training_plan_payload["split_manifest_path"] == str(split_manifest_path)
assert metrics_payload["split_manifest_path"] == str(split_manifest_path)
assert metrics_payload["split_name"] == "test"
```

- [ ] **Step 2: Run tests to verify failure**

Run:
- `uv run pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
- `uv run pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`

Expected: FAIL if any train/eval path still assumes only `day_id` manifests.

- [ ] **Step 3: Keep train/eval generic over manifest key**

No new public CLI flags are needed; rely on the loader abstraction from Task 2.

Add manifest-scope reporting:

```python
"split_manifest_scope": "candidate_id" or "day_id"
```

This value should be inferred from `load_split_manifest_frame(...)`.

- [ ] **Step 4: Run focused tests**

Run:
- `uv run pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
- `uv run pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`

Expected: PASS with both manifest types supported.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "Propagate global corpus split manifest scope through train and eval"
```

### Task 5: Document Defaults and Verify the Whole ML Boundary Stack

**Files:**
- Modify: `README.md`
- Modify: `scripts/pipeline/README.md`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_validate_ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Update operator docs with the global corpus model**

Add these examples to `README.md`:

```bash
python3 scripts/pipeline/run_ml_boundary_pipeline.py DAY --split-strategy global_random --train-fraction 0.70 --validation-fraction 0.15 --test-fraction 0.15 --split-seed 42 --mode tabular_only
python3 scripts/pipeline/run_ml_boundary_pipeline.py DAY_A DAY_B DAY_C --split-strategy global_stratified --mode tabular_only
python3 scripts/pipeline/run_ml_boundary_pipeline.py DAY_A DAY_B DAY_C DAY_D --prepare-only
```

Document the key rule explicitly:

```text
All input days are merged into one candidate corpus first.
The train/validation/test split is then applied to the merged candidate rows.
Input days are data sources, not split units.
```

- [ ] **Step 2: Document exact defaults**

Record:

```text
global corpus split defaults:
- split_strategy = global_stratified
- train_fraction = 0.70
- validation_fraction = 0.15
- test_fraction = 0.15
- split_seed = 42
```

Document `.vocatio` keys:

```text
ML_SPLIT_STRATEGY=global_stratified
ML_SPLIT_TRAIN_FRACTION=0.70
ML_SPLIT_VALIDATION_FRACTION=0.15
ML_SPLIT_TEST_FRACTION=0.15
ML_SPLIT_SEED=42
```

- [ ] **Step 3: Run full ML boundary verification suite**

Run:

```bash
uv run pytest \
  scripts/pipeline/test_export_ml_boundary_reviewed_truth.py \
  scripts/pipeline/test_ml_boundary_review_truth_export.py \
  scripts/pipeline/test_build_ml_boundary_candidate_dataset.py \
  scripts/pipeline/test_validate_ml_boundary_dataset.py \
  scripts/pipeline/test_build_ml_boundary_split_manifest.py \
  scripts/pipeline/test_ml_boundary_dataset.py \
  scripts/pipeline/test_ml_boundary_features.py \
  scripts/pipeline/test_ml_boundary_truth.py \
  scripts/pipeline/test_ml_boundary_training_data.py \
  scripts/pipeline/test_train_ml_boundary_verifier.py \
  scripts/pipeline/test_evaluate_ml_boundary_verifier.py \
  scripts/pipeline/test_run_ml_boundary_pipeline.py -v
```

Expected: PASS for all tests.

- [ ] **Step 4: Run CLI help smoke**

Run:

```bash
python3 scripts/pipeline/run_ml_boundary_pipeline.py --help
python3 scripts/pipeline/train_ml_boundary_verifier.py --help
python3 scripts/pipeline/evaluate_ml_boundary_verifier.py --help
python3 scripts/pipeline/validate_ml_boundary_dataset.py --help
```

Expected: each command exits `0` and documents the corpus split behavior clearly.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add README.md scripts/pipeline/README.md
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "Document global corpus ML boundary split workflow"
```

## Self-Review

- Spec coverage: the plan now matches the intended workflow:
  - input can be `1..100` days,
  - all days are merged into one candidate corpus,
  - split is applied to candidate rows, not to days,
  - split sizes are explicit and configurable,
  - current train/eval/validation code is extended to consume candidate-keyed manifests.
- Placeholder scan: no `TBD` or deferred logic remains; split defaults, config keys, file paths, commands, and test targets are explicit.
- Type consistency: the plan consistently uses `candidate_id`, `split_name`, `split_strategy`, `train_fraction`, `validation_fraction`, `test_fraction`, and `split_seed`.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-16-ml-boundary-global-corpus-splits.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
