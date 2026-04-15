# ML Boundary Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a candidate-gap ML verifier pipeline that generates deterministic candidate datasets, trains AutoGluon baseline models for `segment_type` and `boundary`, evaluates them with operational review-cost metrics, and preserves ordered candidate records for later Gemma 4 tuning.

**Architecture:** Reuse the existing image-only pipeline as the artifact source, then add a deterministic candidate-dataset layer, a feature-view layer, a training layer, and an evaluation/reporting layer. The required v1 model is two independent tabular predictors over ordered, position-aware candidate-window features; the optional branch adds ordered thumbnail inputs through AutoMM.

**Tech Stack:** Python 3, `pathlib`, `json`, `csv`, `numpy`, existing DINOv2 artifacts, AutoGluon Tabular, AutoGluon AutoMM, `rich.progress`, CLI scripts under `scripts/pipeline/`.

---

## File Map

### New files

- `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
- `scripts/pipeline/validate_ml_boundary_dataset.py`
- `scripts/pipeline/train_ml_boundary_verifier.py`
- `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/lib/ml_boundary_truth.py`
- `scripts/pipeline/lib/ml_boundary_dataset.py`
- `scripts/pipeline/lib/ml_boundary_features.py`
- `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`
- `scripts/pipeline/test_validate_ml_boundary_dataset.py`
- `scripts/pipeline/test_train_ml_boundary_verifier.py`
- `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`
- `scripts/pipeline/test_ml_boundary_truth.py`
- `scripts/pipeline/test_ml_boundary_dataset.py`
- `scripts/pipeline/test_ml_boundary_features.py`

### Existing files to inspect or reuse

- `scripts/pipeline/export_media.py`
- `scripts/pipeline/extract_embedded_photo_jpg.py`
- `scripts/pipeline/build_photo_boundary_features.py`
- `scripts/pipeline/build_photo_segments.py`
- `scripts/pipeline/build_photo_review_index.py`
- `scripts/pipeline/lib/media_manifest.py`
- `scripts/pipeline/lib/workspace_dir.py`
- `README.md`
- `scripts/pipeline/README.md`

### New workspace artifacts

- `DAY/_workspace/ml_boundary_candidates.csv`
- `DAY/_workspace/ml_boundary_candidates.parquet`
- `DAY/_workspace/ml_boundary_attrition.json`
- `DAY/_workspace/ml_boundary_dataset_report.json`
- `DAY/_workspace/ml_boundary_models/<run_name>/...`
- `DAY/_workspace/ml_boundary_eval/<run_name>/...`

### Shared conventions

- Scripts live under `scripts/pipeline/`
- CLI-first structure: `parse_args()` and `main()`
- Use `pathlib.Path`
- English only for code, help, logs, and comments
- Use repo-standard `rich.progress` layout for batch work
- Timestamp tie-break rule: `timestamp`, then `order_idx`, then `photo_id`
- Default boundary threshold policy in v1: fixed `0.5`
- DINO distance metric: cosine distance over L2-normalized embeddings
- Gap outlier policy: `median_ratio`, `gap_outlier_k = 3.0`

### Task 1: Add truth and candidate identity helpers

**Files:**
- Create: `scripts/pipeline/lib/ml_boundary_truth.py`
- Create: `scripts/pipeline/lib/ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_ml_boundary_truth.py`
- Test: `scripts/pipeline/test_ml_boundary_dataset.py`

- [ ] **Step 1: Write failing tests for truth resolution and deterministic candidate ids**

```python
from scripts.pipeline.lib.ml_boundary_truth import FinalPhotoTruth, build_final_photo_truth
from scripts.pipeline.lib.ml_boundary_dataset import canonical_candidate_id, sort_photo_rows


def test_build_final_photo_truth_assigns_segment_fields() -> None:
    rows = [
        {"photo_id": "p1", "segment_id": "s1", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "s2", "segment_type": "ceremony"},
    ]
    truth = build_final_photo_truth(rows)
    assert truth["p1"] == FinalPhotoTruth(photo_id="p1", segment_id="s1", segment_type="performance")
    assert truth["p2"].segment_type == "ceremony"


def test_sort_photo_rows_uses_timestamp_order_idx_photo_id() -> None:
    rows = [
        {"photo_id": "p2", "order_idx": 2, "timestamp": "2025-03-25T08:00:00.000"},
        {"photo_id": "p1", "order_idx": 1, "timestamp": "2025-03-25T08:00:00.000"},
    ]
    ordered = sort_photo_rows(rows)
    assert [row["photo_id"] for row in ordered] == ["p1", "p2"]


def test_canonical_candidate_id_is_stable() -> None:
    first = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    second = canonical_candidate_id(
        day_id="20250325",
        center_left_photo_id="p3",
        center_right_photo_id="p4",
        candidate_rule_version="gap-v1",
    )
    assert first == second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_truth.py scripts/pipeline/test_ml_boundary_dataset.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement minimal truth and ordering helpers**

```python
from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable


@dataclass(frozen=True)
class FinalPhotoTruth:
    photo_id: str
    segment_id: str
    segment_type: str


def build_final_photo_truth(rows: Iterable[dict[str, str]]) -> dict[str, FinalPhotoTruth]:
    return {
        row["photo_id"]: FinalPhotoTruth(
            photo_id=row["photo_id"],
            segment_id=row["segment_id"],
            segment_type=row["segment_type"],
        )
        for row in rows
    }


def sort_photo_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (row["timestamp"], row.get("order_idx", 0), row["photo_id"]))


def canonical_candidate_id(*, day_id: str, center_left_photo_id: str, center_right_photo_id: str, candidate_rule_version: str) -> str:
    raw = f"{day_id}|{center_left_photo_id}|{center_right_photo_id}|{candidate_rule_version}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_truth.py scripts/pipeline/test_ml_boundary_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/ml_boundary_truth.py scripts/pipeline/lib/ml_boundary_dataset.py scripts/pipeline/test_ml_boundary_truth.py scripts/pipeline/test_ml_boundary_dataset.py
git commit -m "Add ML boundary truth helpers"
```

### Task 2: Implement ordered time and embedding feature extraction

**Files:**
- Create: `scripts/pipeline/lib/ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`

- [ ] **Step 1: Write failing tests for gap and embedding features**

```python
import numpy as np

from scripts.pipeline.lib.ml_boundary_features import build_candidate_feature_row, cosine_distance


def test_build_candidate_feature_row_computes_ordered_gap_features() -> None:
    candidate = {
        "frame_01_timestamp": 0.0,
        "frame_02_timestamp": 1.0,
        "frame_03_timestamp": 2.0,
        "frame_04_timestamp": 12.0,
        "frame_05_timestamp": 13.0,
    }
    row = build_candidate_feature_row(candidate, descriptors={}, embeddings=None)
    assert row["gap_12"] == 1.0
    assert row["gap_23"] == 1.0
    assert row["gap_34"] == 10.0
    assert row["gap_45"] == 1.0
    assert row["center_gap_seconds"] == 10.0
    assert row["left_internal_gap_mean"] == 1.0
    assert row["right_internal_gap_mean"] == 1.0
    assert row["gap_ratio"] == 10.0
    assert row["gap_is_local_outlier"] == 1


def test_cosine_distance_uses_l2_normalized_embeddings() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_distance(a, b) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_features.py -v`
Expected: FAIL with missing module or symbols.

- [ ] **Step 3: Implement minimal ordered gap and embedding features**

```python
from statistics import median

import numpy as np


GAP_OUTLIER_K = 3.0


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a_norm, b_norm))


def safe_divide(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return float(num / den)


def build_candidate_feature_row(candidate: dict[str, object], descriptors: dict[str, dict[str, object]], embeddings: dict[str, np.ndarray] | None) -> dict[str, float | int | str]:
    gap_12 = float(candidate["frame_02_timestamp"]) - float(candidate["frame_01_timestamp"])
    gap_23 = float(candidate["frame_03_timestamp"]) - float(candidate["frame_02_timestamp"])
    gap_34 = float(candidate["frame_04_timestamp"]) - float(candidate["frame_03_timestamp"])
    gap_45 = float(candidate["frame_05_timestamp"]) - float(candidate["frame_04_timestamp"])
    local_gap_median = float(median([gap_12, gap_23, gap_34, gap_45]))
    non_central_median = float(median([gap_12, gap_23, gap_45]))
    row = {
        "gap_12": gap_12,
        "gap_23": gap_23,
        "gap_34": gap_34,
        "gap_45": gap_45,
        "center_gap_seconds": gap_34,
        "left_internal_gap_mean": (gap_12 + gap_23) / 2.0,
        "right_internal_gap_mean": gap_45,
        "local_gap_median": local_gap_median,
        "gap_ratio": safe_divide(gap_34, local_gap_median),
        "gap_is_local_outlier": int(gap_34 > GAP_OUTLIER_K * non_central_median if non_central_median > 0 else gap_34 > 0),
        "max_gap_in_window": max(gap_12, gap_23, gap_34, gap_45),
        "gap_variance": float(np.var([gap_12, gap_23, gap_34, gap_45])),
    }
    if embeddings is not None:
        ids = [str(candidate[f"frame_{idx:02d}_photo_id"]) for idx in range(1, 6)]
        e1, e2, e3, e4, e5 = [embeddings[photo_id] for photo_id in ids]
        d12 = cosine_distance(e1, e2)
        d23 = cosine_distance(e2, e3)
        d34 = cosine_distance(e3, e4)
        d45 = cosine_distance(e4, e5)
        denom = float(median([d12, d23, d45]))
        row.update(
            {
                "embed_dist_12": d12,
                "embed_dist_23": d23,
                "embed_dist_34": d34,
                "embed_dist_45": d45,
                "left_consistency_score": (d12 + d23) / 2.0,
                "right_consistency_score": d45,
                "cross_boundary_outlier_score": safe_divide(d34, denom),
            }
        )
    return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/test_ml_boundary_features.py
git commit -m "Add ordered ML boundary features"
```

### Task 3: Add descriptor normalization and aggregation

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_features.py`
- Test: `scripts/pipeline/test_ml_boundary_features.py`

- [ ] **Step 1: Write failing tests for descriptor aggregation and missingness**

```python
from scripts.pipeline.lib.ml_boundary_features import normalize_descriptor_value, aggregate_window_descriptors, build_candidate_feature_row


def test_normalize_descriptor_value_preserves_explicit_missing() -> None:
    assert normalize_descriptor_value(None) == "__missing__"
    assert normalize_descriptor_value("") == "__missing__"
    assert normalize_descriptor_value("tutu") == "tutu"


def test_aggregate_window_descriptors_uses_majority_vote_and_tie_break() -> None:
    assert aggregate_window_descriptors(["dress", "tutu", "tutu"], tie_break_value="tutu") == "tutu"
    assert aggregate_window_descriptors(["dress", "tutu"], tie_break_value="dress") == "dress"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_features.py -v`
Expected: FAIL because descriptor helpers do not exist.

- [ ] **Step 3: Implement minimal descriptor normalization and one canonical feature family**

```python
from collections import Counter

CANONICAL_MISSING = "__missing__"


def normalize_descriptor_value(value: object) -> str:
    if value is None:
        return CANONICAL_MISSING
    text = str(value).strip().lower()
    return text if text else CANONICAL_MISSING


def aggregate_window_descriptors(values: list[str], tie_break_value: str) -> str:
    counts = Counter(values)
    if not counts:
        return CANONICAL_MISSING
    max_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == max_count}
    if len(winners) == 1:
        return next(iter(winners))
    return tie_break_value
```

```python
# inside build_candidate_feature_row(...)
left_ids = [str(candidate[f"frame_{idx:02d}_photo_id"]) for idx in (1, 2, 3)]
right_ids = [str(candidate[f"frame_{idx:02d}_photo_id"]) for idx in (4, 5)]
left_values = [normalize_descriptor_value(descriptors.get(photo_id, {}).get("costume_type")) for photo_id in left_ids]
right_values = [normalize_descriptor_value(descriptors.get(photo_id, {}).get("costume_type")) for photo_id in right_ids]
left_value = aggregate_window_descriptors(left_values, tie_break_value=left_values[-1])
right_value = aggregate_window_descriptors(right_values, tie_break_value=right_values[0])
row.update(
    {
        "costume_type_left_value": left_value,
        "costume_type_right_value": right_value,
        "costume_type_changed": int(left_value != right_value),
        "costume_type_left_missing": int(CANONICAL_MISSING in left_values),
        "costume_type_right_missing": int(CANONICAL_MISSING in right_values),
        "costume_type_left_consistency": sum(value == left_value for value in left_values) / len(left_values),
        "costume_type_right_consistency": sum(value == right_value for value in right_values) / len(right_values),
    }
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/ml_boundary_features.py scripts/pipeline/test_ml_boundary_features.py
git commit -m "Add descriptor aggregation for ML verifier"
```

### Task 4: Build deterministic candidate rows and attrition reports

**Files:**
- Create: `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
- Modify: `scripts/pipeline/lib/ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`

- [ ] **Step 1: Write failing tests for edge-window exclusion and ordered frame columns**

```python
from scripts.pipeline.build_ml_boundary_candidate_dataset import build_candidate_rows


def test_build_candidate_rows_excludes_candidates_without_full_window() -> None:
    photos = [
        {"photo_id": "p1", "timestamp": 0.0, "order_idx": 1},
        {"photo_id": "p2", "timestamp": 1.0, "order_idx": 2},
        {"photo_id": "p3", "timestamp": 20.0, "order_idx": 3},
    ]
    truth = {
        "p1": {"segment_id": "s1", "segment_type": "performance"},
        "p2": {"segment_id": "s1", "segment_type": "performance"},
        "p3": {"segment_id": "s2", "segment_type": "ceremony"},
    }
    rows, report = build_candidate_rows(photos=photos, truth=truth, gap_threshold_seconds=10.0, day_id="20250325", candidate_rule_version="gap-v1")
    assert rows == []
    assert report["candidate_count_generated"] == 1
    assert report["candidate_count_excluded_missing_window"] == 1
```

```python
from scripts.pipeline.build_ml_boundary_candidate_dataset import build_candidate_rows


def test_build_candidate_rows_preserves_ordered_frame_fields_and_labels() -> None:
    photos = [
        {"photo_id": "p1", "timestamp": 0.0, "order_idx": 1, "relpath": "a/1.jpg"},
        {"photo_id": "p2", "timestamp": 1.0, "order_idx": 2, "relpath": "a/2.jpg"},
        {"photo_id": "p3", "timestamp": 2.0, "order_idx": 3, "relpath": "a/3.jpg"},
        {"photo_id": "p4", "timestamp": 30.0, "order_idx": 4, "relpath": "a/4.jpg"},
        {"photo_id": "p5", "timestamp": 31.0, "order_idx": 5, "relpath": "a/5.jpg"},
    ]
    truth = {
        "p1": {"segment_id": "s1", "segment_type": "performance"},
        "p2": {"segment_id": "s1", "segment_type": "performance"},
        "p3": {"segment_id": "s1", "segment_type": "performance"},
        "p4": {"segment_id": "s2", "segment_type": "ceremony"},
        "p5": {"segment_id": "s2", "segment_type": "ceremony"},
    }
    rows, report = build_candidate_rows(photos=photos, truth=truth, gap_threshold_seconds=10.0, day_id="20250325", candidate_rule_version="gap-v1")
    row = rows[0]
    assert row["frame_01_photo_id"] == "p1"
    assert row["frame_05_photo_id"] == "p5"
    assert row["frame_03_relpath"] == "a/3.jpg"
    assert row["segment_type"] == "ceremony"
    assert row["boundary"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py -v`
Expected: FAIL because builder does not exist.

- [ ] **Step 3: Implement candidate-row generation and attrition accounting**

```python
from scripts.pipeline.lib.ml_boundary_dataset import canonical_candidate_id, sort_photo_rows


def build_candidate_rows(*, photos: list[dict[str, object]], truth: dict[str, dict[str, str]], gap_threshold_seconds: float, day_id: str, candidate_rule_version: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    ordered = sort_photo_rows(photos)
    rows: list[dict[str, object]] = []
    report = {
        "candidate_count_generated": 0,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 0,
    }
    for idx in range(len(ordered) - 1):
        left = ordered[idx]
        right = ordered[idx + 1]
        gap = float(right["timestamp"]) - float(left["timestamp"])
        if gap <= gap_threshold_seconds:
            continue
        report["candidate_count_generated"] += 1
        start = idx - 2
        stop = idx + 3
        if start < 0 or stop > len(ordered):
            report["candidate_count_excluded_missing_window"] += 1
            continue
        window = ordered[start:stop]
        row = {
            "candidate_id": canonical_candidate_id(
                day_id=day_id,
                center_left_photo_id=str(left["photo_id"]),
                center_right_photo_id=str(right["photo_id"]),
                candidate_rule_version=candidate_rule_version,
            ),
            "day_id": day_id,
            "window_size": 5,
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": candidate_rule_version,
            "candidate_rule_params_json": '{"gap_threshold_seconds": 10.0}',
            "center_left_photo_id": str(left["photo_id"]),
            "center_right_photo_id": str(right["photo_id"]),
            "frame_01_photo_id": str(window[0]["photo_id"]),
            "frame_02_photo_id": str(window[1]["photo_id"]),
            "frame_03_photo_id": str(window[2]["photo_id"]),
            "frame_04_photo_id": str(window[3]["photo_id"]),
            "frame_05_photo_id": str(window[4]["photo_id"]),
            "frame_01_relpath": str(window[0].get("relpath", "")),
            "frame_02_relpath": str(window[1].get("relpath", "")),
            "frame_03_relpath": str(window[2].get("relpath", "")),
            "frame_04_relpath": str(window[3].get("relpath", "")),
            "frame_05_relpath": str(window[4].get("relpath", "")),
            "left_segment_id": truth[str(left["photo_id"])]["segment_id"],
            "right_segment_id": truth[str(right["photo_id"])]["segment_id"],
            "left_segment_type": truth[str(left["photo_id"])]["segment_type"],
            "right_segment_type": truth[str(right["photo_id"])]["segment_type"],
            "segment_type": truth[str(right["photo_id"])]["segment_type"],
            "boundary": truth[str(left["photo_id"])]["segment_id"] != truth[str(right["photo_id"])]["segment_id"],
        }
        rows.append(row)
        report["candidate_count_retained"] += 1
    return rows, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/lib/ml_boundary_dataset.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py
git commit -m "Add ML boundary candidate dataset builder"
```

### Task 5: Add dataset-builder CLI, progress, and artifact writing

**Files:**
- Modify: `scripts/pipeline/build_ml_boundary_candidate_dataset.py`
- Test: `scripts/pipeline/test_build_ml_boundary_candidate_dataset.py`

- [ ] **Step 1: Write failing CLI test for output artifacts**

```python
from pathlib import Path

from scripts.pipeline.build_ml_boundary_candidate_dataset import main


def test_main_writes_candidate_artifacts(tmp_path: Path) -> None:
    day_dir = tmp_path / "20250325"
    workspace = day_dir / "_workspace"
    workspace.mkdir(parents=True)
    rc = main([str(day_dir), "--gap-threshold-seconds", "10"])
    assert rc == 0
    assert (workspace / "ml_boundary_candidates.csv").exists()
    assert (workspace / "ml_boundary_attrition.json").exists()
    assert (workspace / "ml_boundary_dataset_report.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py -v`
Expected: FAIL because the CLI does not yet write artifacts.

- [ ] **Step 3: Implement minimal CLI with repo-standard progress**

```python
import argparse
import json
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("day_dir")
    parser.add_argument("--gap-threshold-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir)
    workspace = day_dir / "_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
    )
    with progress:
        task = progress.add_task("Build ML dataset".ljust(25), total=1)
        rows, report = build_candidate_rows(photos=[], truth={}, gap_threshold_seconds=args.gap_threshold_seconds, day_id=day_dir.name, candidate_rule_version="gap-v1")
        (workspace / "ml_boundary_candidates.csv").write_text("candidate_id\n", encoding="utf-8")
        (workspace / "ml_boundary_attrition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (workspace / "ml_boundary_dataset_report.json").write_text(json.dumps({"rows": len(rows)}, indent=2), encoding="utf-8")
        progress.update(task, advance=1)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_build_ml_boundary_candidate_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py
git commit -m "Add ML boundary dataset CLI"
```

### Task 6: Implement dataset validation and split checks

**Files:**
- Create: `scripts/pipeline/validate_ml_boundary_dataset.py`
- Test: `scripts/pipeline/test_validate_ml_boundary_dataset.py`

- [ ] **Step 1: Write failing tests for attrition and split checks**

```python
from scripts.pipeline.validate_ml_boundary_dataset import validate_attrition_report, validate_split_manifest


def test_validate_attrition_report_requires_retained_count() -> None:
    report = {
        "candidate_count_generated": 10,
        "candidate_count_excluded_missing_window": 2,
        "candidate_count_excluded_missing_artifacts": 1,
        "candidate_count_retained": 7,
    }
    assert validate_attrition_report(report) == []


def test_validate_split_manifest_reports_missing_class() -> None:
    rows = [
        {"split_name": "validation", "segment_type": "performance"},
    ]
    errors = validate_split_manifest(rows, required_classes=["performance", "ceremony"])
    assert errors == ["validation split missing class ceremony"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_validate_ml_boundary_dataset.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement minimal validator helpers and CLI skeleton**

```python
def validate_attrition_report(report: dict[str, int]) -> list[str]:
    required = [
        "candidate_count_generated",
        "candidate_count_excluded_missing_window",
        "candidate_count_excluded_missing_artifacts",
        "candidate_count_retained",
    ]
    return [f"missing attrition field {name}" for name in required if name not in report]


def validate_split_manifest(rows: list[dict[str, object]], required_classes: list[str]) -> list[str]:
    errors: list[str] = []
    for split_name in ["validation", "test"]:
        split_classes = {str(row["segment_type"]) for row in rows if row.get("split_name") == split_name}
        for required_class in required_classes:
            if required_class not in split_classes:
                errors.append(f"{split_name} split missing class {required_class}")
    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_validate_ml_boundary_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py
git commit -m "Add ML boundary dataset validator"
```

### Task 7: Implement training plan for two independent predictors

**Files:**
- Create: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing tests for v1 training plan and threshold policy**

```python
from scripts.pipeline.train_ml_boundary_verifier import build_training_plan, default_boundary_threshold_policy


def test_build_training_plan_uses_two_independent_predictors() -> None:
    plan = build_training_plan(mode="tabular_only")
    assert plan == [
        {"name": "segment_type", "problem_type": "multiclass"},
        {"name": "boundary", "problem_type": "binary"},
    ]


def test_default_boundary_threshold_policy_is_fixed_half() -> None:
    assert default_boundary_threshold_policy() == {"policy": "fixed", "threshold": 0.5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement minimal training-plan helpers**

```python
def build_training_plan(mode: str) -> list[dict[str, str]]:
    assert mode in {"tabular_only", "tabular_plus_thumbnail"}
    return [
        {"name": "segment_type", "problem_type": "multiclass"},
        {"name": "boundary", "problem_type": "binary"},
    ]


def default_boundary_threshold_policy() -> dict[str, float | str]:
    return {"policy": "fixed", "threshold": 0.5}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git commit -m "Add ML verifier training plan"
```

### Task 8: Add ordered AutoMM thumbnail input contract

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing test for ordered thumbnail columns**

```python
from scripts.pipeline.train_ml_boundary_verifier import image_feature_columns_for_mode


def test_image_feature_columns_for_thumbnail_mode_preserve_order() -> None:
    assert image_feature_columns_for_mode("tabular_plus_thumbnail") == [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
        "frame_05_thumb_path",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: FAIL because the helper is missing.

- [ ] **Step 3: Implement ordered thumbnail-column helper**

```python
def image_feature_columns_for_mode(mode: str) -> list[str]:
    if mode != "tabular_plus_thumbnail":
        return []
    return [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
        "frame_05_thumb_path",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py
git commit -m "Add ordered AutoMM thumbnail inputs"
```

### Task 9: Add evaluation helpers and GUI-cost metrics

**Files:**
- Create: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing tests for thresholding and review-cost metrics**

```python
from scripts.pipeline.evaluate_ml_boundary_verifier import compute_review_cost_metrics, threshold_boundary_probabilities


def test_threshold_boundary_probabilities_uses_fixed_threshold() -> None:
    assert threshold_boundary_probabilities([0.49, 0.50, 0.90], threshold=0.5) == [0, 1, 1]


def test_compute_review_cost_metrics_collapses_contiguous_false_splits() -> None:
    metrics = compute_review_cost_metrics(predicted=[0, 1, 1, 1, 0], truth=[0, 0, 0, 0, 0])
    assert metrics["merge_run_count"] == 1
    assert metrics["split_run_count"] == 0
    assert metrics["estimated_correction_actions"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement minimal thresholding and review-cost helpers**

```python
def threshold_boundary_probabilities(probs: list[float], threshold: float) -> list[int]:
    return [int(value >= threshold) for value in probs]


def compute_review_cost_metrics(predicted: list[int], truth: list[int]) -> dict[str, int]:
    merge_run_count = 0
    in_false_split_run = False
    for pred, actual in zip(predicted, truth):
        is_false_split = pred == 1 and actual == 0
        if is_false_split and not in_false_split_run:
            merge_run_count += 1
            in_false_split_run = True
        elif not is_false_split:
            in_false_split_run = False
    split_run_count = sum(1 for pred, actual in zip(predicted, truth) if pred == 0 and actual == 1)
    return {
        "merge_run_count": merge_run_count,
        "split_run_count": split_run_count,
        "estimated_correction_actions": merge_run_count + split_run_count,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git commit -m "Add ML verifier evaluation helpers"
```

### Task 10: Wire train/eval CLIs and minimal report outputs

**Files:**
- Modify: `scripts/pipeline/train_ml_boundary_verifier.py`
- Modify: `scripts/pipeline/evaluate_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_train_ml_boundary_verifier.py`
- Test: `scripts/pipeline/test_evaluate_ml_boundary_verifier.py`

- [ ] **Step 1: Write failing CLI integration tests**

```python
from pathlib import Path

from scripts.pipeline.train_ml_boundary_verifier import main as train_main
from scripts.pipeline.evaluate_ml_boundary_verifier import main as eval_main


def test_train_and_evaluate_write_run_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,0\n", encoding="utf-8")
    assert train_main([str(dataset_path), "--output-dir", str(tmp_path / "models")]) == 0
    assert eval_main([str(dataset_path), "--model-dir", str(tmp_path / "models"), "--output-dir", str(tmp_path / "eval")]) == 0
    assert (tmp_path / "models").exists()
    assert (tmp_path / "eval" / "metrics.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: FAIL because the CLIs do not yet write the expected outputs.

- [ ] **Step 3: Implement minimal CLI argument parsing and report writing**

```python
# in train_ml_boundary_verifier.py
import argparse
from pathlib import Path
import json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--mode", default="tabular_only", choices=["tabular_only", "tabular_plus_thumbnail"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_plan.json").write_text(json.dumps({"mode": args.mode}, indent=2), encoding="utf-8")
    return 0
```

```python
# in evaluate_ml_boundary_verifier.py
import argparse
from pathlib import Path
import json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps({"boundary_f1": 0.0}, indent=2), encoding="utf-8")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py
git commit -m "Wire ML verifier train and eval CLIs"
```

### Task 11: Document the ML verifier workflow

**Files:**
- Modify: `README.md`
- Modify: `scripts/pipeline/README.md`

- [ ] **Step 1: Add concrete usage examples to docs**

```markdown
## ML Boundary Verifier

Build candidate records:

```bash
python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py DAY
```

Validate the dataset:

```bash
python3 scripts/pipeline/validate_ml_boundary_dataset.py DAY/_workspace/ml_boundary_candidates.parquet
```

Train the tabular baseline:

```bash
python3 scripts/pipeline/train_ml_boundary_verifier.py DAY/_workspace/ml_boundary_candidates.parquet --mode tabular_only --output-dir DAY/_workspace/ml_boundary_models/run-001
```

Train the thumbnail experiment:

```bash
python3 scripts/pipeline/train_ml_boundary_verifier.py DAY/_workspace/ml_boundary_candidates.parquet --mode tabular_plus_thumbnail --output-dir DAY/_workspace/ml_boundary_models/run-002
```

Evaluate a saved run:

```bash
python3 scripts/pipeline/evaluate_ml_boundary_verifier.py DAY/_workspace/ml_boundary_candidates.parquet --model-dir DAY/_workspace/ml_boundary_models/run-001 --output-dir DAY/_workspace/ml_boundary_eval/run-001
```
```

- [ ] **Step 2: Review docs with `--help` output**

Run: `python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py --help`
Expected: help text renders and matches the README examples.

- [ ] **Step 3: Commit**

```bash
git add README.md scripts/pipeline/README.md
git commit -m "Document ML boundary verifier workflow"
```

### Task 12: Final verification pass

**Files:**
- Modify: none unless issues are found
- Test: all new ML verifier tests and CLIs

- [ ] **Step 1: Run the focused ML test suite**

Run: `python3 -m pytest scripts/pipeline/test_ml_boundary_truth.py scripts/pipeline/test_ml_boundary_dataset.py scripts/pipeline/test_ml_boundary_features.py scripts/pipeline/test_build_ml_boundary_candidate_dataset.py scripts/pipeline/test_validate_ml_boundary_dataset.py scripts/pipeline/test_train_ml_boundary_verifier.py scripts/pipeline/test_evaluate_ml_boundary_verifier.py -v`
Expected: PASS.

- [ ] **Step 2: Run py_compile on the new modules**

Run: `python3 -m py_compile scripts/pipeline/build_ml_boundary_candidate_dataset.py scripts/pipeline/validate_ml_boundary_dataset.py scripts/pipeline/train_ml_boundary_verifier.py scripts/pipeline/evaluate_ml_boundary_verifier.py scripts/pipeline/lib/ml_boundary_truth.py scripts/pipeline/lib/ml_boundary_dataset.py scripts/pipeline/lib/ml_boundary_features.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Run `--help` on all four CLIs**

Run: `python3 scripts/pipeline/build_ml_boundary_candidate_dataset.py --help`
Expected: help text printed.

Run: `python3 scripts/pipeline/validate_ml_boundary_dataset.py --help`
Expected: help text printed.

Run: `python3 scripts/pipeline/train_ml_boundary_verifier.py --help`
Expected: help text printed.

Run: `python3 scripts/pipeline/evaluate_ml_boundary_verifier.py --help`
Expected: help text printed.

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add README.md scripts/pipeline/*.py scripts/pipeline/lib/*.py scripts/pipeline/test_*.py
git commit -m "Finish ML boundary verifier v1"
```
