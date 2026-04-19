# ML Boundary Descriptor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `costume_type_*` ML features with real flattened descriptor features loaded from `photo_pre_model_annotations/*.json`.

**Architecture:** Keep the existing candidate-gap ML pipeline intact and change only the training feature view. Load per-photo annotation JSON from the workspace, flatten `payload["data"]` into deterministic left/right categorical columns, and keep the current numeric gap features unchanged.

**Tech Stack:** Python 3, csv/json/pathlib, existing pipeline CLI scripts, AutoGluon Tabular, pytest-style script tests.

---

## File Structure

- Modify: `scripts/pipeline/lib/photo_pre_model_annotations.py`
  - Reuse the existing annotation loader and add a helper that returns descriptor records keyed by `photo_id` / `relative_path` in the format expected by ML.
- Modify: `scripts/pipeline/lib/ml_boundary_features.py`
  - Remove the placeholder `costume_type_*` feature block and replace it with generic flattened descriptor feature builders.
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
  - Load annotation descriptors from the workspace, pass them into `build_candidate_feature_row(...)`, preserve missing-annotation counters, and expose them through the training bundle.
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
  - Write and print descriptor missing-count summary at the end of training artifact generation.
- Modify: `scripts/pipeline/test_ml_boundary_features.py`
  - Replace tests tied to `costume_type_*` with tests for flattened descriptor aggregation, token splitting, ordering, and missingness.
- Modify: `scripts/pipeline/test_ml_boundary_training_data.py`
  - Cover end-to-end training bundle derivation from real annotation JSON fixtures.
- Modify: `scripts/pipeline/test_train_ml_boundary_verifier.py`
  - Verify missing-annotation counters are persisted into training artifacts and reported in CLI output.

### Task 1: Add Failing Tests for Descriptor Flattening

**Files:**
- Modify: `scripts/pipeline/test_ml_boundary_features.py`
- Modify: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`

- [ ] **Step 1: Replace the old placeholder descriptor expectations with new failing tests**

```python
def test_build_candidate_feature_row_flattens_scalar_descriptor_fields() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"upper_garment": "Top", "lower_garment": "Skirt"},
        "p2": {"upper_garment": "top", "lower_garment": "skirt"},
        "p3": {"upper_garment": "TOP", "lower_garment": "Skirt"},
        "p4": {"upper_garment": "Top", "lower_garment": "Tutu"},
        "p5": {"upper_garment": "Top", "lower_garment": "Tutu"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_upper_garment"] == "top"
    assert row["right_upper_garment"] == "top"
    assert row["left_lower_garment"] == "skirt"
    assert row["right_lower_garment"] == "tutu"
```

```python
def test_build_candidate_feature_row_flattens_multivalue_descriptor_fields() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"dominant_colors": ["White", "Purple"]},
        "p2": {"dominant_colors": ["purple"]},
        "p3": {"dominant_colors": ["white", "purple"]},
        "p4": {"dominant_colors": ["Blue", "White"]},
        "p5": {"dominant_colors": ["white"]},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_dominant_colors_01"] == "purple"
    assert row["left_dominant_colors_02"] == "white"
    assert row["left_dominant_colors_03"] == "__missing__"
    assert row["right_dominant_colors_01"] == "blue"
    assert row["right_dominant_colors_02"] == "white"
```

```python
def test_build_candidate_feature_row_splits_text_values_on_list_delimiters_only() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 0.1,
        "frame_03_timestamp": 0.2,
        "frame_04_timestamp": 20.2,
        "frame_05_timestamp": 20.3,
        "frame_01_photo_id": "p1",
        "frame_02_photo_id": "p2",
        "frame_03_photo_id": "p3",
        "frame_04_photo_id": "p4",
        "frame_05_photo_id": "p5",
    }
    descriptors = {
        "p1": {"footwear": "ballet_shoes"},
        "p2": {"footwear": "dance_shoes"},
        "p3": {"footwear": "ballet_shoes"},
        "p4": {"props": "fan; ribbon"},
        "p5": {"props": "ribbon/banner"},
    }

    row = build_candidate_feature_row(candidate, descriptors=descriptors, embeddings=None)

    assert row["left_footwear"] == "ballet_shoes"
    assert row["right_props_01"] == "banner"
    assert row["right_props_02"] == "fan"
    assert row["right_props_03"] == "ribbon"
```

```python
def test_load_training_data_bundle_reports_missing_annotation_counts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    split_manifest_path = tmp_path / "splits.csv"
    annotation_dir = tmp_path / "photo_pre_model_annotations"
    annotation_dir.mkdir()

    dataset_path.write_text(
        \"\"\"candidate_id,day_id,segment_type,boundary,frame_01_photo_id,frame_02_photo_id,frame_03_photo_id,frame_04_photo_id,frame_05_photo_id,frame_01_timestamp,frame_02_timestamp,frame_03_timestamp,frame_04_timestamp,frame_05_timestamp,frame_01_thumb_path,frame_02_thumb_path,frame_03_thumb_path,frame_04_thumb_path,frame_05_thumb_path,frame_01_relpath,frame_02_relpath,frame_03_relpath,frame_04_relpath,frame_05_relpath\nc1,20260323,performance,1,p1,p2,p3,p4,p5,0.0,0.1,0.2,20.2,20.3,a,b,c,d,e,p-a/1.hif,p-a/2.hif,p-a/3.hif,p-a/4.hif,p-a/5.hif\n\"\"\",
        encoding="utf-8",
    )
    split_manifest_path.write_text("candidate_id,split_name\nc1,train\n", encoding="utf-8")
    (annotation_dir / "p-a").mkdir()
    (annotation_dir / "p-a" / "1.hif.json").write_text(
        json.dumps({"schema_version": "photo_pre_model_v1", "relative_path": "p-a/1.hif", "data": {"upper_garment": "top"}}),
        encoding="utf-8",
    )

    bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode="tabular_only",
        annotation_dir=annotation_dir,
    )

    assert bundle.missing_annotation_photo_count == 4
    assert bundle.missing_annotation_candidate_count == 1
```

- [ ] **Step 2: Run the targeted tests to confirm they fail before implementation**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -v
```

Expected:

- failures because `left_upper_garment`, `right_dominant_colors_01`, and missing-annotation counters do not exist yet

- [ ] **Step 3: Commit the failing tests**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/test_ml_boundary_features.py scripts/pipeline/test_ml_boundary_training_data.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "test: cover ML descriptor flattening"
```

### Task 2: Implement Real Descriptor Loading and Flattened Feature Generation

**Files:**
- Modify: `scripts/pipeline/lib/photo_pre_model_annotations.py`
- Modify: `scripts/pipeline/lib/ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`

- [ ] **Step 1: Add a loader that reads annotation `data` by relative path**

```python
def load_photo_pre_model_data_by_relative_path(
    output_dir: Path,
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for relative_path in relative_paths:
        output_path = build_annotation_output_path(output_dir, relative_path)
        if not output_path.exists():
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        data = payload.get("data")
        if isinstance(data, Mapping):
            loaded[relative_path] = dict(data)
    return loaded
```

- [ ] **Step 2: Replace the placeholder costume feature block with generic descriptor flattening**

```python
DESCRIPTOR_LIST_DELIMITERS = (",", ";", "|", "/")
DESCRIPTOR_MAX_VALUES_PER_FIELD = 5
CANONICAL_MISSING = "__missing__"

def normalize_descriptor_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tokens = [str(part) for part in value]
    else:
        tokens = [str(value)]

    normalized: list[str] = []
    for token in tokens:
        pending = [token.lower()]
        for delimiter in DESCRIPTOR_LIST_DELIMITERS:
            next_pending: list[str] = []
            for part in pending:
                next_pending.extend(part.split(delimiter))
            pending = next_pending
        normalized.extend(part.strip() for part in pending if part.strip())

    return sorted(set(normalized))[:DESCRIPTOR_MAX_VALUES_PER_FIELD]
```

```python
def build_candidate_feature_row(
    candidate: Mapping[str, object],
    descriptors: Mapping[str, Mapping[str, object]],
    embeddings: Mapping[str, Sequence[float]] | None,
) -> dict[str, float | int | str]:
    row = {
        "gap_12": gaps[0],
        "gap_23": gaps[1],
        "gap_34": gaps[2],
        "gap_45": gaps[3],
        "left_internal_gap_mean": (gaps[0] + gaps[1]) / 2.0,
        "local_gap_median": local_gap_median,
        "gap_ratio": safe_divide(gaps[2], local_gap_median),
        "gap_variance": float(pvariance(gaps)),
    }
    row.update(build_side_descriptor_features("left", left_photo_ids, descriptors, tie_break_index=-1))
    row.update(build_side_descriptor_features("right", right_photo_ids, descriptors, tie_break_index=0))
    if embeddings is not None:
        row.update(build_embedding_features(...))
    return row
```

Remove the old placeholder descriptor outputs entirely:

```python
# delete these legacy placeholder columns
"costume_type_left_value"
"costume_type_right_value"
"costume_type_changed"
"costume_type_left_missing"
"costume_type_right_missing"
"costume_type_left_consistency"
"costume_type_right_consistency"
```

```python
def build_side_descriptor_features(
    side_prefix: str,
    photo_ids: Sequence[str],
    descriptors: Mapping[str, Mapping[str, object]],
    *,
    tie_break_index: int,
) -> dict[str, str]:
    descriptor_keys = sorted({
        key
        for photo_id in photo_ids
        for key in descriptors.get(photo_id, {}).keys()
    })
    features: dict[str, str] = {}
    for key in descriptor_keys:
        per_frame_tokens = [normalize_descriptor_tokens(descriptors.get(photo_id, {}).get(key)) for photo_id in photo_ids]
        if any(len(tokens) > 1 for tokens in per_frame_tokens):
            merged = sorted({token for tokens in per_frame_tokens for token in tokens})[:DESCRIPTOR_MAX_VALUES_PER_FIELD]
            for index in range(DESCRIPTOR_MAX_VALUES_PER_FIELD):
                features[f\"{side_prefix}_{key}_{index + 1:02d}\"] = merged[index] if index < len(merged) else CANONICAL_MISSING
        else:
            candidates = [tokens[0] if tokens else CANONICAL_MISSING for tokens in per_frame_tokens]
            features[f\"{side_prefix}_{key}\"] = majority_vote(candidates, tie_break_value=candidates[tie_break_index])
    return features
```

- [ ] **Step 3: Run descriptor-feature tests and make them pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py -v
```

Expected:

- PASS for the new scalar, multivalue, delimiter, and missingness tests

- [ ] **Step 4: Commit the descriptor feature implementation**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/photo_pre_model_annotations.py scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/test_ml_boundary_features.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: flatten photo descriptors for ML features"
```

### Task 3: Integrate Descriptor Loading Into the Training Bundle

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`

- [ ] **Step 1: Extend the training bundle to carry annotation coverage counters**

```python
@dataclass(frozen=True)
class TrainingDataBundle:
    train_rows: "TrainingTable"
    validation_rows: "TrainingTable"
    test_rows: "TrainingTable"
    split_manifest_scope: str
    split_counts_by_name: dict[str, int]
    shared_feature_columns: list[str]
    image_feature_columns: list[str]
    missing_annotation_photo_count: int
    missing_annotation_candidate_count: int
    segment_type: PredictorTrainingData
    boundary: PredictorTrainingData
```

- [ ] **Step 2: Load annotation JSON using frame relative paths before deriving the feature view**

```python
def load_training_data_bundle(
    dataset_path: Path,
    *,
    split_manifest_path: Path,
    mode: str,
    annotation_dir: Path | None = None,
    require_train_validation: bool = True,
) -> TrainingDataBundle:
    ...
    resolved_annotation_dir = annotation_dir or dataset_path.parent / "photo_pre_model_annotations"
    joined_frame, derived_feature_columns, missing_counts = _derive_feature_view(
        joined_frame,
        annotation_dir=resolved_annotation_dir,
    )
    ...
    return TrainingDataBundle(
        ...,
        missing_annotation_photo_count=missing_counts["missing_annotation_photo_count"],
        missing_annotation_candidate_count=missing_counts["missing_annotation_candidate_count"],
        ...
    )
```

```python
def _derive_feature_view(
    table: TrainingTable,
    *,
    annotation_dir: Path,
) -> tuple[TrainingTable, list[str], dict[str, int]]:
    derived_rows: list[dict[str, object]] = []
    missing_annotation_photo_count = 0
    missing_annotation_candidate_count = 0

    for row in table.rows:
        relative_paths = [str(row[f"frame_{index:02d}_relpath"]).strip() for index in range(1, 6)]
        descriptors_by_path = load_photo_pre_model_data_by_relative_path(annotation_dir, relative_paths)
        descriptors_by_photo_id = {
            str(row[f"frame_{index:02d}_photo_id"]).strip(): descriptors_by_path.get(relative_paths[index - 1], {})
            for index in range(1, 6)
        }
        missing_for_candidate = sum(1 for relative_path in relative_paths if relative_path not in descriptors_by_path)
        missing_annotation_photo_count += missing_for_candidate
        if missing_for_candidate:
            missing_annotation_candidate_count += 1
        derived_features = build_candidate_feature_row(row, descriptors=descriptors_by_photo_id, embeddings=None)
        ...

    return derived_table, derived_feature_columns, {
        "missing_annotation_photo_count": missing_annotation_photo_count,
        "missing_annotation_candidate_count": missing_annotation_candidate_count,
    }
```

- [ ] **Step 3: Run the training-data tests and confirm the bundle carries real descriptor features**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py -v
```

Expected:

- PASS with `left_*` / `right_*` descriptor columns present
- PASS with correct missing-annotation counters

- [ ] **Step 4: Commit the training-bundle integration**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/test_ml_boundary_training_data.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: load photo annotations into ML training bundle"
```

### Task 4: Persist and Print Missing-Annotation Summary

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Add missing-annotation counters to the training metadata and summary payloads**

```python
def _build_training_metadata_payload(
    output_dir: Path,
    mode: str,
    training_bundle: TrainingDataBundle,
    final_artifact_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "mode": mode,
        "split_manifest_scope": training_bundle.split_manifest_scope,
        "split_counts_by_name": training_bundle.split_counts_by_name,
        "missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
        "missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
        ...
    }
```

```python
training_summary["descriptor_annotation_coverage"] = {
    "missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
    "missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
}
```

- [ ] **Step 2: Print the counts only once at the end of training artifact generation**

```python
console.print(
    "Descriptor annotation coverage: "
    f"missing_annotation_photo_count={training_bundle.missing_annotation_photo_count} "
    f"missing_annotation_candidate_count={training_bundle.missing_annotation_candidate_count}"
)
```

- [ ] **Step 3: Add a focused test for metadata/reporting**

```python
def test_train_writes_descriptor_missing_annotation_counts(tmp_path: Path) -> None:
    ...
    metadata = json.loads((output_dir / "training_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))

    assert metadata["missing_annotation_photo_count"] == 4
    assert metadata["missing_annotation_candidate_count"] == 1
    assert summary["descriptor_annotation_coverage"]["missing_annotation_photo_count"] == 4
```

- [ ] **Step 4: Run the training CLI test and verify the report behavior**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -v
```

Expected:

- PASS with metadata and summary counts present

- [ ] **Step 5: Commit the reporting changes**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: report ML descriptor annotation coverage"
```

### Task 5: End-to-End Verification on Real Data

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_training_data.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Run the focused automated verification**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_features.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_training_data.py /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_train_ml_boundary_verifier.py -v
```

Expected:

- all targeted tests PASS

- [ ] **Step 2: Re-run the real one-day training pipeline**

Run:

```bash
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/run_ml_boundary_pipeline.py /arch03/V/DWC2026/20260323 --mode tabular_only --model-run-id day-20260323 --restart
```

Expected:

- pipeline completes successfully
- training no longer reports placeholder `costume_type_*` columns as constant features
- final training metadata/summary include missing annotation counts

- [ ] **Step 3: Inspect the written training feature column manifest**

Run:

```bash
python3 -c "import json; from pathlib import Path; p=Path('/arch03/WORKSPACE/20260323DWC/ml_boundary_corpus/ml_boundary_models/day-20260323/feature_columns.json'); print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))"
```

Expected:

- `left_upper_garment`, `right_upper_garment`, and similar flattened descriptor columns are present
- old placeholder `costume_type_*` columns are absent

- [ ] **Step 4: Commit the completed integration after verification**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/photo_pre_model_annotations.py scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/lib/ml_boundary_training_data.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_ml_boundary_features.py scripts/pipeline/test_ml_boundary_training_data.py scripts/pipeline/test_train_ml_boundary_verifier.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: integrate photo pre-model descriptors into ML boundary features"
```
