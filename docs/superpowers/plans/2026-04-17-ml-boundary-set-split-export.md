# ML Boundary Set-Split Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ML boundary truth export accept numeric GUI set splits such as `86 -> 86 + 87` and export them as `performance -> performance` boundaries instead of raising `Unknown explicit split name`.

**Architecture:** Keep the existing review-state schema and GUI unchanged. Fix only the ML truth exporter by introducing one narrow classifier for split semantics that uses both `new_name` and `is_set_split`, then route flattening/export through that classifier and lock the behavior down with focused tests.

**Tech Stack:** Python 3, pytest, existing ML-boundary exporter modules in `scripts/pipeline/`

---

### Task 1: Lock Down Current GUI Split Semantics In Unit Tests

**Files:**
- Modify: `scripts/pipeline/test_ml_boundary_review_truth_export.py`
- Test: `scripts/pipeline/test_ml_boundary_review_truth_export.py`

- [ ] **Step 1: Write the failing test for numeric set splits**

Add this test near the existing split-name coverage in `scripts/pipeline/test_ml_boundary_review_truth_export.py`:

```python
def test_flatten_final_display_sets_treats_numeric_set_split_as_performance() -> None:
    review_index_payload = {
        "performances": [
            _performance(
                performance_number="86",
                set_id="set-86",
                photos=[
                    _photo("p1", "IMG_0001.JPG", "2026-03-23T14:58:53"),
                    _photo("p2", "IMG_0002.JPG", "2026-03-23T15:00:00"),
                    _photo("p3", "IMG_0003.JPG", "2026-03-23T15:01:00"),
                ],
            )
        ]
    }
    review_state = {
        "splits": {
            "set-86": [
                {
                    "start_filename": "IMG_0002.JPG",
                    "new_name": "87",
                    "is_set_split": True,
                }
            ]
        },
        "merges": [],
    }

    rows = flatten_final_display_sets(rebuild_final_display_sets(review_index_payload, review_state))

    assert rows == [
        {"photo_id": "p1", "segment_id": "set-86", "segment_type": "performance"},
        {"photo_id": "p2", "segment_id": "set-86::IMG_0002.JPG", "segment_type": "performance"},
        {"photo_id": "p3", "segment_id": "set-86::IMG_0002.JPG", "segment_type": "performance"},
    ]
```

- [ ] **Step 2: Keep the invalid semantic-name case explicit**

Update the existing unknown-name test so it documents the intended contract: invalid names fail only for semantic splits.

Use this review state in `test_flatten_final_display_sets_rejects_unknown_explicit_split_names`:

```python
review_state = {
    "splits": {
        "set-105": [
            {
                "start_filename": "IMG_0002.JPG",
                "new_name": "Awards",
                "is_set_split": False,
            }
        ]
    },
    "merges": [],
}
```

- [ ] **Step 3: Run the focused unit tests and confirm the new one fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_review_truth_export.py -v
```

Expected:
- the new numeric split test fails with the current `Unknown explicit split name` behavior
- existing tests still run

- [ ] **Step 4: Commit the failing-test checkpoint**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/test_ml_boundary_review_truth_export.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "test: cover ML numeric set split export"
```

### Task 2: Classify Split Semantics Using `is_set_split`

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_review_truth_export.py`
- Test: `scripts/pipeline/test_ml_boundary_review_truth_export.py`

- [ ] **Step 1: Add a dedicated split-to-segment-type helper**

In `scripts/pipeline/lib/ml_boundary_review_truth_export.py`, add a helper above `rebuild_final_display_sets`:

```python
def _split_spec_to_segment_type(
    *,
    new_name: str,
    is_set_split: bool,
    original_performance_number: str,
    set_id: str,
) -> str:
    normalized = new_name.strip().lower()
    if is_set_split:
        return "performance"
    if normalized == "ceremony":
        return "ceremony"
    if normalized == "warmup":
        return "warmup"
    original_normalized = original_performance_number.strip().lower()
    if normalized == original_normalized:
        return "performance"
    raise ValueError(
        f"Unknown explicit semantic split name for display set {set_id}: {new_name}. "
        "Use ceremony or warmup for semantic splits, or mark the split as a set split."
    )
```

- [ ] **Step 2: Preserve `is_set_split` when rebuilding display sets**

In the `valid_specs.append(...)` block inside `rebuild_final_display_sets`, include:

```python
"is_set_split": bool(spec.get("is_set_split", False)),
```

Then replace the current `segment_names` construction with explicit segment metadata:

```python
segment_display_names = [original_number] + [spec["new_name"] or original_number for spec in valid_specs]
segment_types = ["performance"] + [
    _split_spec_to_segment_type(
        new_name=spec["new_name"],
        is_set_split=spec["is_set_split"],
        original_performance_number=original_number,
        set_id=f"{base_set_id}::{spec['start_filename']}",
    )
    for spec in valid_specs
]
```

When appending each rebuilt display set, include:

```python
"display_name": segment_display_names[index],
"segment_type": segment_types[index],
```

- [ ] **Step 3: Stop inferring segment type from display name during flattening**

Replace the current `flatten_final_display_sets` segment-type resolution:

```python
segment_type = str(display_set.get("segment_type", "") or "").strip().lower()
if not segment_type:
    segment_type = _display_name_to_segment_type(
        display_name,
        original_performance_number=original_performance_number,
        set_id=display_set_id,
    )
```

Keep the existing `VALID_SEGMENT_TYPES` check immediately after that.

This preserves compatibility for any caller that still passes legacy display sets without `segment_type`, while making rebuilt split semantics authoritative.

- [ ] **Step 4: Run the focused unit tests and confirm they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_review_truth_export.py -v
```

Expected:
- numeric set split test passes
- ceremony/warmup tests still pass
- invalid semantic split test still fails in the intended case

- [ ] **Step 5: Commit the exporter logic**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_review_truth_export.py scripts/pipeline/test_ml_boundary_review_truth_export.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "fix: support numeric ML set split export"
```

### Task 3: Cover The CLI Export Path

**Files:**
- Modify: `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`
- Test: `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`

- [ ] **Step 1: Add a real CLI-path regression test**

Add this test to `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`:

```python
def test_main_writes_performance_rows_for_numeric_set_split() -> None:
    with TemporaryDirectory() as tmp:
        root_dir = Path(tmp)
        day_dir = root_dir / "20260323"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        for relative_path in [
            "cam_a/IMG_0001.ARW",
            "cam_a/IMG_0002.ARW",
            "cam_a/IMG_0003.ARW",
        ]:
            source_path = day_dir / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"raw")
            proxy_path = workspace_dir / "proxy_jpg" / f"{Path(relative_path).name}.jpg"
            proxy_path.parent.mkdir(parents=True, exist_ok=True)
            proxy_path.write_bytes(b"jpg")

        index_payload = {
            "day": day_dir.name,
            "workspace_dir": str(workspace_dir),
            "performance_count": 1,
            "photo_count": 3,
            "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
            "performances": [
                {
                    "performance_number": "86",
                    "set_id": "set-86",
                    "timeline_status": "normal",
                    "performance_start_local": "2026-03-23T14:58:53",
                    "performance_end_local": "2026-03-23T15:01:00",
                    "photos": [
                        _photo("cam_a/IMG_0001.ARW", "2026-03-23T14:58:53"),
                        _photo("cam_a/IMG_0002.ARW", "2026-03-23T15:00:00"),
                        _photo("cam_a/IMG_0003.ARW", "2026-03-23T15:01:00"),
                    ],
                }
            ],
        }
        review_state = {
            "version": 2,
            "day": day_dir.name,
            "performances": {},
            "splits": {
                "set-86": [
                    {
                        "start_filename": "IMG_0002.ARW",
                        "new_name": "87",
                        "is_set_split": True,
                    }
                ]
            },
            "merges": [],
        }

        (workspace_dir / "performance_proxy_index.json").write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        (workspace_dir / "review_state.json").write_text(
            json.dumps(review_state, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        exit_code = main([str(day_dir), "--workspace-dir", str(workspace_dir)])

        assert exit_code == 0
        with (workspace_dir / "ml_boundary_reviewed_truth.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert rows == [
            {"photo_id": "cam_a/IMG_0001.ARW", "segment_id": "set-86", "segment_type": "performance"},
            {"photo_id": "cam_a/IMG_0002.ARW", "segment_id": "set-86::IMG_0002.ARW", "segment_type": "performance"},
            {"photo_id": "cam_a/IMG_0003.ARW", "segment_id": "set-86::IMG_0002.ARW", "segment_type": "performance"},
        ]
```

- [ ] **Step 2: Run only the export CLI tests**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_export_ml_boundary_reviewed_truth.py -v
```

Expected:
- the new CLI regression passes
- existing CLI export tests still pass

- [ ] **Step 3: Commit the CLI regression coverage**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/test_export_ml_boundary_reviewed_truth.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "test: cover CLI export for numeric set splits"
```

### Task 4: Verify The End-To-End Fix On Real And Local Test Data

**Files:**
- Modify: `scripts/pipeline/lib/ml_boundary_review_truth_export.py` only if error text needs final tightening
- Test: `scripts/pipeline/test_ml_boundary_review_truth_export.py`
- Test: `scripts/pipeline/test_export_ml_boundary_reviewed_truth.py`

- [ ] **Step 1: Run the full exporter-related test slice**

Run:

```bash
uv run pytest \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_ml_boundary_review_truth_export.py \
  /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_export_ml_boundary_reviewed_truth.py \
  -v
```

Expected:
- all selected tests pass

- [ ] **Step 2: Run the real failing command against the reviewed day**

Run from the worktree:

```bash
cd /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier
python3 scripts/pipeline/export_ml_boundary_reviewed_truth.py /arch03/V/DWC2026/20260323 --workspace-dir /arch03/WORKSPACE/20260323DWC
```

Expected:
- command exits `0`
- no `Unknown explicit split name` error for the `86 -> 87` split
- `/arch03/WORKSPACE/20260323DWC/ml_boundary_reviewed_truth.csv` is written

- [ ] **Step 3: If needed, tighten the final error wording**

If the remaining invalid-name failure message still implies that only `ceremony`/`warmup` are ever valid explicit split names, update it to mention the actual rule:

```python
raise ValueError(
    f"Unknown explicit semantic split name for display set {set_id}: {new_name}. "
    "Use ceremony or warmup for semantic splits, or mark the split as a set split."
)
```

- [ ] **Step 4: Re-run the real command after any wording change**

Run:

```bash
cd /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier
python3 scripts/pipeline/export_ml_boundary_reviewed_truth.py /arch03/V/DWC2026/20260323 --workspace-dir /arch03/WORKSPACE/20260323DWC
```

Expected:
- exit `0`
- output CSV refreshed

- [ ] **Step 5: Commit the final verified fix**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/lib/ml_boundary_review_truth_export.py scripts/pipeline/test_ml_boundary_review_truth_export.py scripts/pipeline/test_export_ml_boundary_reviewed_truth.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "fix: export ML truth for numeric set splits"
```
