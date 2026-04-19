# ML Boundary VLM Handoff Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ML boundary training so both predictors consume the heuristic boundary signals currently passed directly to the visual boundary VLM prompt.

**Architecture:** Keep the current ML candidate dataset and predictor split intact, but enrich the derived feature view by joining each 5-frame candidate window to its four adjacent `photo_boundary_scores.csv` rows. Store the new heuristic features and missing-coverage counters in the training bundle and reporting artifacts, then stop before any VLM prompt integration.

**Tech Stack:** Python 3, CSV-based pipeline scripts, Rich CLI, AutoGluon training wrappers, pytest script tests

---

## File Map

- Modify: [scripts/pipeline/lib/ml_boundary_training_data.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/lib/ml_boundary_training_data.py)
  - Add heuristic boundary row loading, candidate-window joins, and missing-coverage counters.
- Modify: [scripts/pipeline/lib/ml_boundary_features.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/lib/ml_boundary_features.py)
  - Add explicit heuristic feature expansion for the four adjacent frame pairs.
- Modify: [scripts/pipeline/train_ml_boundary_verifier.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/train_ml_boundary_verifier.py)
  - Persist heuristic coverage and reflect the richer feature set in training metadata/report artifacts.
- Modify: [scripts/pipeline/run_ml_boundary_pipeline.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py)
  - Surface the new heuristic coverage counts in the final pipeline summary JSON.
- Test: [scripts/pipeline/test_ml_boundary_training_data.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py)
  - Add failing coverage for heuristic joins and missing-heuristic accounting.
- Test: [scripts/pipeline/test_ml_boundary_features.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py)
  - Add failing coverage for heuristic feature-name and value expansion.
- Test: [scripts/pipeline/test_train_ml_boundary_verifier.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py)
  - Assert heuristic coverage fields are written to training artifacts.
- Test: [scripts/pipeline/test_run_ml_boundary_pipeline.py](/home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py)
  - Assert pipeline summary carries heuristic coverage fields through the end-to-end wrapper.

### Task 1: Lock Heuristic Feature Contract In Tests

**Files:**
- Modify: `scripts/pipeline/test_ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`

- [ ] **Step 1: Write the failing feature-expansion test**

Add a test that proves a candidate row can emit four pairwise heuristic feature groups:

```python
def test_build_candidate_feature_row_includes_pairwise_heuristic_features() -> None:
    candidate = {
        "frame_01_timestamp": "0",
        "frame_02_timestamp": "5",
        "frame_03_timestamp": "10",
        "frame_04_timestamp": "40",
        "frame_05_timestamp": "45",
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    heuristic_features = {
        "12": {
            "dino_cosine_distance": 0.101,
            "boundary_score": 0.202,
            "distance_zscore": 0.303,
            "smoothed_distance_zscore": 0.404,
            "time_gap_boost": 0.0,
            "boundary_label": "none",
        },
        "23": {
            "dino_cosine_distance": 0.111,
            "boundary_score": 0.222,
            "distance_zscore": 0.333,
            "smoothed_distance_zscore": 0.444,
            "time_gap_boost": 0.0,
            "boundary_label": "none",
        },
        "34": {
            "dino_cosine_distance": 0.777,
            "boundary_score": 0.888,
            "distance_zscore": 1.111,
            "smoothed_distance_zscore": 1.222,
            "time_gap_boost": 1.0,
            "boundary_label": "hard",
        },
        "45": {
            "dino_cosine_distance": 0.123,
            "boundary_score": 0.234,
            "distance_zscore": 0.345,
            "smoothed_distance_zscore": 0.456,
            "time_gap_boost": 0.0,
            "boundary_label": "none",
        },
    }

    row = build_candidate_feature_row(
        candidate,
        descriptors={},
        embeddings=None,
        heuristic_features=heuristic_features,
    )

    assert row["heuristic_dino_dist_12"] == 0.101
    assert row["heuristic_boundary_score_34"] == 0.888
    assert row["heuristic_distance_zscore_34"] == 1.111
    assert row["heuristic_smoothed_distance_zscore_45"] == 0.456
    assert row["heuristic_time_gap_boost_34"] == 1.0
    assert row["heuristic_boundary_label_34"] == "hard"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py -q
```

Expected: FAIL because `build_candidate_feature_row()` does not yet accept or emit heuristic features.

- [ ] **Step 3: Write the minimal feature-builder implementation**

Extend the function signature and add one helper for the four pair slots:

```python
def build_candidate_feature_row(
    candidate: Mapping[str, object],
    descriptors: Mapping[str, object],
    embeddings: Mapping[str, Sequence[float]] | None,
    descriptor_field_registry: Mapping[str, str] | None = None,
    heuristic_features: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, float | int | str]:
```

And add a helper like:

```python
def build_heuristic_feature_block(
    heuristic_features: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for pair_name in ("12", "23", "34", "45"):
        feature_row = dict(heuristic_features.get(pair_name, {})) if heuristic_features else {}
        row[f"heuristic_dino_dist_{pair_name}"] = _normalize_heuristic_float(
            feature_row.get("dino_cosine_distance")
        )
        row[f"heuristic_boundary_score_{pair_name}"] = _normalize_heuristic_float(
            feature_row.get("boundary_score")
        )
        row[f"heuristic_distance_zscore_{pair_name}"] = _normalize_heuristic_float(
            feature_row.get("distance_zscore")
        )
        row[f"heuristic_smoothed_distance_zscore_{pair_name}"] = _normalize_heuristic_float(
            feature_row.get("smoothed_distance_zscore")
        )
        row[f"heuristic_time_gap_boost_{pair_name}"] = _normalize_heuristic_float(
            feature_row.get("time_gap_boost")
        )
        row[f"heuristic_boundary_label_{pair_name}"] = _normalize_heuristic_label(
            feature_row.get("boundary_label")
        )
    return row
```

Then merge it into `row` before descriptor and embedding logic returns.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py -q
```

Expected: PASS, including the new heuristic feature test.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/test_ml_boundary_features.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add heuristic boundary feature block"
```

### Task 2: Join Candidates To `photo_boundary_scores.csv`

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`

- [ ] **Step 1: Write the failing join-and-missing-coverage test**

Add a test that builds a tiny candidate dataset plus a matching `photo_boundary_scores.csv`, then asserts both feature values and missing counts:

```python
def test_load_training_data_bundle_joins_heuristic_boundary_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    dataset_path = workspace_dir / "ml_boundary_candidates.corpus.csv"
    split_manifest_path = workspace_dir / "ml_boundary_splits.csv"
    boundary_scores_path = workspace_dir / "photo_boundary_scores.csv"

    dataset_rows = [
        {
            "candidate_id": "c-001",
            "day_id": "20260323",
            "segment_type": "performance",
            "boundary": "1",
            "split_name": "train",
            "frame_01_timestamp": "0",
            "frame_02_timestamp": "5",
            "frame_03_timestamp": "10",
            "frame_04_timestamp": "40",
            "frame_05_timestamp": "45",
            "frame_01_photo_id": "p1",
            "frame_02_photo_id": "p2",
            "frame_03_photo_id": "p3",
            "frame_04_photo_id": "p4",
            "frame_05_photo_id": "p5",
            "frame_01_relpath": "cam/p1.jpg",
            "frame_02_relpath": "cam/p2.jpg",
            "frame_03_relpath": "cam/p3.jpg",
            "frame_04_relpath": "cam/p4.jpg",
            "frame_05_relpath": "cam/p5.jpg",
        },
    ]

    boundary_rows = [
        {"left_relative_path": "cam/p1.jpg", "right_relative_path": "cam/p2.jpg", "dino_cosine_distance": "0.101", "distance_zscore": "0.201", "smoothed_distance_zscore": "0.301", "time_gap_boost": "0.0", "boundary_score": "0.401", "boundary_label": "none", "boundary_reason": "distance_only", "model_source": "bootstrap", "time_gap_seconds": "5.0"},
        {"left_relative_path": "cam/p2.jpg", "right_relative_path": "cam/p3.jpg", "dino_cosine_distance": "0.102", "distance_zscore": "0.202", "smoothed_distance_zscore": "0.302", "time_gap_boost": "0.0", "boundary_score": "0.402", "boundary_label": "none", "boundary_reason": "distance_only", "model_source": "bootstrap", "time_gap_seconds": "5.0"},
        {"left_relative_path": "cam/p3.jpg", "right_relative_path": "cam/p4.jpg", "dino_cosine_distance": "0.903", "distance_zscore": "1.203", "smoothed_distance_zscore": "1.303", "time_gap_boost": "1.0", "boundary_score": "0.953", "boundary_label": "hard", "boundary_reason": "hard_gap", "model_source": "bootstrap", "time_gap_seconds": "30.0"},
    ]

    write_candidate_csv(dataset_path, dataset_rows)
    write_split_manifest_csv(split_manifest_path, [{"candidate_id": "c-001", "split_name": "train"}])
    write_boundary_scores_csv(boundary_scores_path, boundary_rows)

    bundle = load_training_data_bundle(dataset_path, split_manifest_path=split_manifest_path, mode="tabular_only")

    assert bundle.train_rows.rows[0]["heuristic_boundary_score_34"] == 0.953
    assert bundle.train_rows.rows[0]["heuristic_boundary_label_34"] == "hard"
    assert bundle.train_rows.rows[0]["heuristic_boundary_label_45"] == "__missing__"
    assert bundle.missing_heuristic_pair_count == 1
    assert bundle.missing_heuristic_candidate_count == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -q
```

Expected: FAIL because the training bundle does not yet load or join `photo_boundary_scores.csv`, and the new coverage fields do not exist.

- [ ] **Step 3: Implement heuristic-row loading and candidate joins**

In `ml_boundary_training_data.py`, extend `TrainingDataBundle`:

```python
@dataclass(frozen=True)
class TrainingDataBundle:
    ...
    missing_annotation_photo_count: int
    missing_annotation_candidate_count: int
    missing_heuristic_pair_count: int
    missing_heuristic_candidate_count: int
    ...
```

Add a resolver and loader:

```python
def _resolve_boundary_scores_path(dataset_path: Path) -> Path | None:
    candidate_dirs = [dataset_path.parent / "photo_boundary_scores.csv"]
    if dataset_path.parent.name == "ml_boundary_corpus":
        candidate_dirs.insert(0, dataset_path.parent.parent / "photo_boundary_scores.csv")
    for candidate_path in candidate_dirs:
        if candidate_path.exists():
            return candidate_path
    return None
```

```python
def _load_boundary_scores_by_pair(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                str(row.get("left_relative_path", "")).strip(),
                str(row.get("right_relative_path", "")).strip(),
            ): dict(row)
            for row in reader
        }
```

Then wire `_derive_feature_view()` to build the per-candidate pair map:

```python
heuristic_rows_by_pair = _load_boundary_scores_by_pair(_resolve_boundary_scores_path(dataset_path))
...
heuristic_features, candidate_missing_heuristic_count = _build_candidate_heuristic_features(
    row,
    boundary_rows_by_pair=heuristic_rows_by_pair,
)
derived_features = build_candidate_feature_row(
    row,
    descriptors=descriptors,
    embeddings=None,
    descriptor_field_registry=descriptor_field_registry,
    heuristic_features=heuristic_features,
)
```

And add:

```python
def _build_candidate_heuristic_features(
    row: dict[str, object],
    *,
    boundary_rows_by_pair: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, dict[str, str]], int]:
    pair_specs = {
        "12": ("frame_01_relpath", "frame_02_relpath"),
        "23": ("frame_02_relpath", "frame_03_relpath"),
        "34": ("frame_03_relpath", "frame_04_relpath"),
        "45": ("frame_04_relpath", "frame_05_relpath"),
    }
    feature_rows: dict[str, dict[str, str]] = {}
    missing_count = 0
    for pair_name, (left_key, right_key) in pair_specs.items():
        pair = (str(row.get(left_key, "")).strip(), str(row.get(right_key, "")).strip())
        boundary_row = boundary_rows_by_pair.get(pair)
        if boundary_row is None:
            missing_count += 1
            continue
        feature_rows[pair_name] = boundary_row
    return feature_rows, missing_count
```

- [ ] **Step 4: Run the training-data test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -q
```

Expected: PASS, including the new join-and-missing-coverage test.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/test_ml_boundary_training_data.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: join heuristic boundary rows into ML bundle"
```

### Task 3: Persist Heuristic Coverage In Training Artifacts

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write the failing artifact test**

Add a test that asserts the new counters are written into both metadata and report payloads:

```python
def test_training_artifacts_include_heuristic_coverage(tmp_path: Path, monkeypatch) -> None:
    ...
    metadata = json.loads((output_dir / "training_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))

    assert metadata["missing_heuristic_pair_count"] == 4
    assert metadata["missing_heuristic_candidate_count"] == 2
    assert report["heuristic_boundary_coverage"]["missing_pair_count"] == 4
    assert report["heuristic_boundary_coverage"]["missing_candidate_count"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: FAIL because those fields are not yet present in the written artifacts.

- [ ] **Step 3: Implement metadata/report persistence**

Update the metadata builder in `train_ml_boundary_verifier.py`:

```python
"missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
"missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
"missing_heuristic_pair_count": training_bundle.missing_heuristic_pair_count,
"missing_heuristic_candidate_count": training_bundle.missing_heuristic_candidate_count,
```

And extend the training report payload:

```python
"heuristic_boundary_coverage": {
    "missing_pair_count": training_bundle.missing_heuristic_pair_count,
    "missing_candidate_count": training_bundle.missing_heuristic_candidate_count,
},
```

Also include one short terminal line in the final training report block:

```python
Text(
    f"Heuristic coverage: missing_pairs={training_bundle.missing_heuristic_pair_count}, "
    f"missing_candidates={training_bundle.missing_heuristic_candidate_count}"
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -q
```

Expected: PASS, including the new heuristic coverage artifact test.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: report ML heuristic coverage"
```

### Task 4: Carry Heuristic Coverage Through Pipeline Summary

**Files:**
- Modify: `scripts/pipeline/run_ml_boundary_pipeline.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Write the failing pipeline-summary test**

Add a test that feeds training metadata with heuristic coverage fields and asserts they survive in `ml_boundary_pipeline_summary.json`:

```python
def test_pipeline_summary_includes_heuristic_coverage(tmp_path: Path, monkeypatch) -> None:
    ...
    summary_payload = json.loads((corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["training_metadata"]["missing_heuristic_pair_count"] == 4
    assert summary_payload["training_metadata"]["missing_heuristic_candidate_count"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: FAIL because the summary path does not yet assert or display the heuristic coverage fields.

- [ ] **Step 3: Implement pipeline-summary forwarding**

Update the compact final summary and JSON assembly in `run_ml_boundary_pipeline.py` so the loaded training metadata is forwarded unchanged, and add one summary line:

```python
console.print(
    "Heuristic coverage: "
    f"missing_pairs={training_metadata.get('missing_heuristic_pair_count', 0)}, "
    f"missing_candidates={training_metadata.get('missing_heuristic_candidate_count', 0)}"
)
```

If the summary already stores the full `training_metadata` object, keep that contract and only tighten tests around the new fields rather than adding a second copy.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: PASS, including the new heuristic coverage summary test.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: surface heuristic coverage in pipeline summary"
```

### Task 5: Run Focused Verification For Stage A Stop Point

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_run_ml_boundary_pipeline.py`

- [ ] **Step 1: Run the focused Stage A test set**

Run:

```bash
uv run pytest \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_run_ml_boundary_pipeline.py -q
```

Expected: PASS with all Stage A feature, artifact, and summary tests green.

- [ ] **Step 2: Verify no Stage B prompt integration slipped in**

Run:

```bash
rg -n "ML hint|ml_boundary_prediction|ml_segment_type_prediction|build_gap_hint_lines|build_photo_pre_model_lines" /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/probe_vlm_photo_boundaries.py
```

Expected:

- existing `build_gap_hint_lines` and `build_photo_pre_model_lines` references still present
- no new prompt-time ML hint strings added yet

- [ ] **Step 3: Commit the final Stage A checkpoint**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/run_ml_boundary_pipeline.py scripts/pipeline/test_ml_boundary_features.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_run_ml_boundary_pipeline.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add stage A ML heuristic boundary inputs"
```

- [ ] **Step 4: Stop and hand back to the user**

Do not modify `probe_vlm_photo_boundaries.py`.

Hand back the branch with:

- richer ML feature inputs
- updated metadata and reports
- no prompt changes

The user then runs real ML experiments and decides whether Stage B should begin.
