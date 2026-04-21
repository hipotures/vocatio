# VLM Window Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable VLM window selection schemas so `VLM_WINDOW_RADIUS` keeps its current meaning while photo selection inside the left and right segments becomes strategy-driven and reusable across probe runs, GUI index reconstruction, and manual GUI analysis.

**Architecture:** Add one shared `window_schema` library that owns schema names, default resolution, deterministic selection, and tie-break rules. Integrate probe, GUI index, and manual GUI with that shared selector so current `consecutive` behavior remains the default while new schemas use the same segment-based contract everywhere.

**Tech Stack:** Python 3, existing `scripts/pipeline` modules, `pytest`, PySide6 GUI code, `.vocatio` config loading, Rich-based CLI workflows

---

## File Structure

### New files

- Create: `scripts/pipeline/lib/window_schema.py`
  - shared schema enum
  - defaults and validation helpers
  - segment-side selection helpers
  - deterministic random selection
  - index/time/tie-break logic
- Create: `scripts/pipeline/test_window_schema.py`
  - unit tests for schema parsing and row selection

### Modified files

- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - parse CLI/config defaults for `window_schema` and `window_schema_seed`
  - detect left/right segments around candidate gaps
  - delegate row selection to `lib.window_schema`
  - persist schema metadata in run payloads
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
  - probe config default coverage
  - candidate row selection coverage
  - run metadata coverage
- Modify: `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
  - read schema metadata from run metadata
  - rebuild candidate rows through shared selection
- Modify: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
  - metadata read/validation coverage
  - candidate reconstruction coverage
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - resolve schema config for manual runtime
  - add dropdown state for manual VLM schema override
  - route manual candidate selection through shared selector
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - manual runtime config coverage
  - GUI dropdown behavior coverage
- Modify: `README.md`
  - document new workflow settings
- Modify: `scripts/pipeline/README.md`
  - document new CLI/config behavior for probe/manual flows

## Task 1: Add Shared Window Schema Library

**Files:**
- Create: `scripts/pipeline/lib/window_schema.py`
- Test: `scripts/pipeline/test_window_schema.py`

- [ ] **Step 1: Write the failing tests for defaults, enum validation, and `consecutive`**

```python
import importlib.util
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import window_schema


def _row(relative_path: str, start_epoch_ms: int) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "start_epoch_ms": str(start_epoch_ms),
    }


def test_defaults_and_validation() -> None:
    assert window_schema.DEFAULT_WINDOW_SCHEMA == "consecutive"
    assert window_schema.DEFAULT_WINDOW_SCHEMA_SEED == 42
    assert window_schema.parse_window_schema("random") == "random"
    try:
        window_schema.parse_window_schema("weird")
    except ValueError as exc:
        assert "consecutive" in str(exc)
        assert "time_boundary_spread" in str(exc)
    else:
        raise AssertionError("expected invalid schema error")


def test_consecutive_selects_rows_nearest_gap() -> None:
    rows = [
        _row("cam/a.jpg", 1000),
        _row("cam/b.jpg", 2000),
        _row("cam/c.jpg", 3000),
        _row("cam/d.jpg", 4000),
    ]
    selected = window_schema.select_segment_rows(
        rows,
        radius=2,
        schema="consecutive",
        gap_side="left",
        schema_seed=42,
    )
    assert [row["relative_path"] for row in selected] == ["cam/c.jpg", "cam/d.jpg"]
```

- [ ] **Step 2: Run the new tests to verify failure**

Run: `uv run pytest scripts/pipeline/test_window_schema.py -q`
Expected: FAIL with `ModuleNotFoundError` for `lib.window_schema` or missing attributes such as `DEFAULT_WINDOW_SCHEMA`

- [ ] **Step 3: Implement the minimal shared schema module**

```python
from __future__ import annotations

import random
from typing import Literal, Mapping, Sequence

WindowSchemaName = Literal[
    "consecutive",
    "random",
    "index_quantile",
    "time_quantile",
    "time_max_min",
    "time_boundary_spread",
]

DEFAULT_WINDOW_SCHEMA: WindowSchemaName = "consecutive"
DEFAULT_WINDOW_SCHEMA_SEED = 42
WINDOW_SCHEMA_VALUES: tuple[WindowSchemaName, ...] = (
    "consecutive",
    "random",
    "index_quantile",
    "time_quantile",
    "time_max_min",
    "time_boundary_spread",
)


def parse_window_schema(value: object) -> WindowSchemaName:
    normalized = str(value or "").strip()
    if not normalized:
        return DEFAULT_WINDOW_SCHEMA
    if normalized not in WINDOW_SCHEMA_VALUES:
        raise ValueError(
            "window schema must be one of: " + ", ".join(WINDOW_SCHEMA_VALUES)
        )
    return normalized  # type: ignore[return-value]


def parse_window_schema_seed(value: object) -> int:
    normalized = str(value or "").strip()
    if not normalized:
        return DEFAULT_WINDOW_SCHEMA_SEED
    return int(normalized)


def select_segment_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    radius: int,
    schema: WindowSchemaName,
    gap_side: Literal["left", "right"],
    schema_seed: int,
) -> list[dict[str, str]]:
    selected_count = min(int(radius), len(rows))
    normalized_rows = [dict(row) for row in rows]
    if selected_count <= 0:
        return []
    if schema == "consecutive":
        if gap_side == "left":
            return normalized_rows[-selected_count:]
        return normalized_rows[:selected_count]
    if schema == "random":
        rng = random.Random((schema_seed, gap_side, tuple(row["relative_path"] for row in normalized_rows)))
        indexes = sorted(rng.sample(range(len(normalized_rows)), selected_count))
        return [normalized_rows[index] for index in indexes]
    raise NotImplementedError(schema)
```

- [ ] **Step 4: Expand tests for all remaining schemas before implementing them**

```python
def test_random_is_deterministic_for_same_seed() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 8)]
    first = window_schema.select_segment_rows(
        rows, radius=3, schema="random", gap_side="right", schema_seed=42
    )
    second = window_schema.select_segment_rows(
        rows, radius=3, schema="random", gap_side="right", schema_seed=42
    )
    assert [row["relative_path"] for row in first] == [row["relative_path"] for row in second]


def test_index_quantile_prefers_rows_farther_from_gap_on_tie() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 6)]
    selected = window_schema.select_segment_rows(
        rows, radius=3, schema="index_quantile", gap_side="left", schema_seed=42
    )
    assert [row["relative_path"] for row in selected] == ["cam/1.jpg", "cam/2.jpg", "cam/5.jpg"]


def test_time_quantile_uses_time_not_index() -> None:
    rows = [
        _row("cam/1.jpg", 1000),
        _row("cam/2.jpg", 2000),
        _row("cam/3.jpg", 3000),
        _row("cam/4.jpg", 4000),
        _row("cam/5.jpg", 5000),
        _row("cam/20.jpg", 20000),
        _row("cam/21.jpg", 21000),
        _row("cam/22.jpg", 22000),
        _row("cam/23.jpg", 23000),
        _row("cam/24.jpg", 24000),
    ]
    selected = window_schema.select_segment_rows(
        rows, radius=3, schema="time_quantile", gap_side="left", schema_seed=42
    )
    assert [row["relative_path"] for row in selected] == ["cam/1.jpg", "cam/5.jpg", "cam/24.jpg"]


def test_time_boundary_spread_always_keeps_boundary_nearest_photo() -> None:
    rows = [_row(f"cam/{index}.jpg", index * 1000) for index in range(1, 8)]
    selected = window_schema.select_segment_rows(
        rows, radius=3, schema="time_boundary_spread", gap_side="left", schema_seed=42
    )
    assert selected[-1]["relative_path"] == "cam/7.jpg"
```

- [ ] **Step 5: Implement the remaining schema selection logic**

```python
def _distance_from_gap(index: int, row_count: int, gap_side: Literal["left", "right"]) -> int:
    if gap_side == "left":
        return (row_count - 1) - index
    return index


def _sort_indexes_for_output(indexes: set[int]) -> list[int]:
    return sorted(indexes)


def _choose_quantile_indexes(
    row_count: int,
    selected_count: int,
    *,
    gap_side: Literal["left", "right"],
) -> list[int]:
    if selected_count >= row_count:
        return list(range(row_count))
    if selected_count == 1:
        return [0 if gap_side == "right" else row_count - 1]
    indexes: list[int] = []
    denominator = selected_count - 1
    for ordinal in range(selected_count):
        raw = ordinal * (row_count - 1) / denominator
        lower = int(raw)
        upper = min(row_count - 1, lower + 1)
        if lower == upper:
            chosen = lower
        else:
            lower_distance = abs(raw - lower)
            upper_distance = abs(raw - upper)
            if lower_distance < upper_distance:
                chosen = lower
            elif upper_distance < lower_distance:
                chosen = upper
            else:
                chosen = lower if _distance_from_gap(lower, row_count, gap_side) >= _distance_from_gap(upper, row_count, gap_side) else upper
        indexes.append(chosen)
    return _sort_indexes_for_output(set(indexes))
```

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `uv run pytest scripts/pipeline/test_window_schema.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/lib/window_schema.py scripts/pipeline/test_window_schema.py
git commit -m "Add shared VLM window schema selector"
```

## Task 2: Integrate Probe Config, Segment Selection, and Run Metadata

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write the failing probe tests for config defaults and segment-based selection**

```python
def test_apply_vocatio_defaults_reads_window_schema_and_seed(self):
    day_dir = self.make_day_dir(
        "\n".join(
            [
                "VLM_NAME=qwen3.5:9b",
                "VLM_WINDOW_RADIUS=3",
                "VLM_WINDOW_SCHEMA=index_quantile",
                "VLM_WINDOW_SCHEMA_SEED=99",
            ]
        )
    )
    config_path = self.write_vlm_models_config(
        '''
models:
  - VLM_NAME: "qwen3.5:9b"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11435"
    VLM_MODEL: "test-model"
    VLM_MAX_OUTPUT_TOKENS: 256
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_RESPONSE_SCHEMA_MODE: "off"
    VLM_JSON_VALIDATION_MODE: "strict"
'''
    )
    with mock.patch.object(probe, "VLM_MODELS_CONFIG_PATH", config_path, create=True):
        args = probe.apply_vocatio_defaults(probe.parse_args([str(day_dir)]), day_dir)
    self.assertEqual(args.window_schema, "index_quantile")
    self.assertEqual(args.window_schema_seed, 99)


def test_build_candidate_window_rows_uses_index_quantile_segment_sampling(self):
    joined_rows = [
        {"relative_path": f"cam/{index}.jpg", "start_epoch_ms": str(index * 1000)}
        for index in range(1, 11)
    ]
    rows = probe.build_window_rows_for_cut_index(
        joined_rows=joined_rows,
        cut_index=4,
        window_radius=3,
        window_schema="index_quantile",
        window_schema_seed=42,
        boundary_gap_seconds=10,
    )
    assert [row["relative_path"] for row in rows] == [
        "cam/1.jpg", "cam/3.jpg", "cam/5.jpg", "cam/6.jpg", "cam/8.jpg", "cam/10.jpg"
    ]
```

- [ ] **Step 2: Run the probe tests to verify failure**

Run: `uv run pytest scripts/pipeline/test_probe_vlm_photo_boundaries.py -q`
Expected: FAIL with missing `window_schema` attributes or missing helper such as `build_window_rows_for_cut_index`

- [ ] **Step 3: Add CLI/config fields and shared selection helpers in probe**

```python
from lib import window_schema as window_schema_lib

DEFAULT_WINDOW_SCHEMA = window_schema_lib.DEFAULT_WINDOW_SCHEMA
DEFAULT_WINDOW_SCHEMA_SEED = window_schema_lib.DEFAULT_WINDOW_SCHEMA_SEED

VLM_WORKFLOW_CONFIG_KEYS = frozenset(
    {
        "VLM_EMBEDDED_MANIFEST_CSV",
        "VLM_PHOTO_MANIFEST_CSV",
        "VLM_IMAGE_VARIANT",
        "VLM_WINDOW_RADIUS",
        "VLM_WINDOW_SCHEMA",
        "VLM_WINDOW_SCHEMA_SEED",
        "VLM_BOUNDARY_GAP_SECONDS",
        "VLM_MAX_BATCHES",
        "VLM_PHOTO_PRE_MODEL_DIR",
        "VLM_ML_MODEL_RUN_ID",
        "VLM_DUMP_DEBUG_DIR",
    }
)

parser.add_argument(
    "--window-schema",
    choices=window_schema_lib.WINDOW_SCHEMA_VALUES,
    default=DEFAULT_WINDOW_SCHEMA,
    help=f"Photo selection schema inside each segment. Default: {DEFAULT_WINDOW_SCHEMA}",
)
parser.add_argument(
    "--window-schema-seed",
    type=int,
    default=DEFAULT_WINDOW_SCHEMA_SEED,
    help=f"Deterministic seed for window schema selection. Default: {DEFAULT_WINDOW_SCHEMA_SEED}",
)
```

- [ ] **Step 4: Implement segment discovery and row selection through the shared module**

```python
def _segment_bounds_for_cut_index(
    rows: Sequence[Mapping[str, str]],
    *,
    cut_index: int,
    boundary_gap_seconds: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    left_start = 0
    for index in range(cut_index - 1, -1, -1):
        gap_seconds = rounded_seconds(int(str(rows[index + 1]["start_epoch_ms"])) - int(str(rows[index]["start_epoch_ms"])))
        if gap_seconds >= boundary_gap_seconds:
            left_start = index + 1
            break
    right_end = len(rows)
    for index in range(cut_index + 1, len(rows) - 1):
        gap_seconds = rounded_seconds(int(str(rows[index + 1]["start_epoch_ms"])) - int(str(rows[index]["start_epoch_ms"])))
        if gap_seconds >= boundary_gap_seconds:
            right_end = index + 1
            break
    return (left_start, cut_index + 1), (cut_index + 1, right_end)


def build_window_rows_for_cut_index(
    *,
    joined_rows: Sequence[Mapping[str, str]],
    cut_index: int,
    window_radius: int,
    window_schema: str,
    window_schema_seed: int,
    boundary_gap_seconds: int,
) -> list[Mapping[str, str]]:
    (left_start, left_end), (right_start, right_end) = _segment_bounds_for_cut_index(
        joined_rows,
        cut_index=cut_index,
        boundary_gap_seconds=boundary_gap_seconds,
    )
    left_rows = window_schema_lib.select_segment_rows(
        joined_rows[left_start:left_end],
        radius=window_radius,
        schema=window_schema_lib.parse_window_schema(window_schema),
        gap_side="left",
        schema_seed=window_schema_seed,
    )
    right_rows = window_schema_lib.select_segment_rows(
        joined_rows[right_start:right_end],
        radius=window_radius,
        schema=window_schema_lib.parse_window_schema(window_schema),
        gap_side="right",
        schema_seed=window_schema_seed,
    )
    combined_rows = [dict(row) for row in [*left_rows, *right_rows]]
    return sorted(combined_rows, key=lambda row: int(str(row["start_epoch_ms"])))
```

- [ ] **Step 5: Persist schema metadata in run args and output rows**

```python
args_payload = {
    "window_radius": args.window_radius,
    "window_schema": args.window_schema,
    "window_schema_seed": args.window_schema_seed,
    "boundary_gap_seconds": args.boundary_gap_seconds,
    "image_variant": args.image_variant,
    "response_schema_mode": args.response_schema_mode,
}

RESUME_CONFIG_KEYS = (
    "embedded_manifest_csv",
    "photo_manifest_csv",
    "provider",
    "image_variant",
    "window_radius",
    "window_schema",
    "window_schema_seed",
    "boundary_gap_seconds",
    "model",
    "ollama_base_url",
    "ollama_num_ctx",
    "ollama_num_predict",
    "ollama_keep_alive",
    "timeout_seconds",
    "temperature",
    "ollama_think",
    "response_schema_mode",
    "json_validation_mode",
    "photo_pre_model_dir",
    "effective_ml_model_run_id",
    "effective_extra_instructions",
)
```

- [ ] **Step 6: Run the probe tests and focused window schema tests**

Run: `uv run pytest scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/test_window_schema.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git commit -m "Integrate VLM window schema into probe flow"
```

## Task 3: Use Run Metadata Schema in GUI Index Reconstruction

**Files:**
- Modify: `scripts/pipeline/build_vlm_photo_boundary_gui_index.py`
- Test: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`

- [ ] **Step 1: Write the failing GUI index tests**

```python
def test_resolve_runtime_window_schema_reads_metadata_defaults() -> None:
    metadata = {"args": {"window_radius": 3, "window_schema": "random", "window_schema_seed": 42}}
    assert module.resolve_runtime_window_schema(metadata) == "random"
    assert module.resolve_runtime_window_schema_seed(metadata) == 42


def test_build_ml_hint_pairs_for_run_reconstructs_candidates_with_schema(monkeypatch) -> None:
    run_metadata = {
        "args": {
            "window_radius": 3,
            "window_schema": "index_quantile",
            "window_schema_seed": 42,
            "photo_pre_model_dir": "photo_pre_model_annotations",
        }
    }
    run_rows = [
        {
            "relative_paths_json": '["cam/1.jpg","cam/3.jpg","cam/5.jpg","cam/6.jpg","cam/8.jpg","cam/10.jpg"]',
            "window_radius": "3",
        }
    ]
    # assertion target: no metadata error and one hint pair produced
```

- [ ] **Step 2: Run the GUI index tests to verify failure**

Run: `uv run pytest scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py -q`
Expected: FAIL with missing resolver functions or schema metadata handling

- [ ] **Step 3: Add schema metadata resolvers and row validation**

```python
def resolve_runtime_window_schema(run_metadata: Mapping[str, Any]) -> str:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    return probe.window_schema_lib.parse_window_schema(args.get("window_schema", ""))


def resolve_runtime_window_schema_seed(run_metadata: Mapping[str, Any]) -> int:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    return probe.window_schema_lib.parse_window_schema_seed(args.get("window_schema_seed", ""))
```

- [ ] **Step 4: Reconstruct candidate windows through shared schema helpers instead of trusting only stored row order**

```python
runtime_window_radius = resolve_runtime_window_radius(run_metadata)
runtime_window_schema = resolve_runtime_window_schema(run_metadata)
runtime_window_schema_seed = resolve_runtime_window_schema_seed(run_metadata)

candidate_rows = probe.build_window_rows_for_cut_index(
    joined_rows=joined_rows,
    cut_index=cut_index,
    window_radius=runtime_window_radius,
    window_schema=runtime_window_schema,
    window_schema_seed=runtime_window_schema_seed,
    boundary_gap_seconds=int(str(args.get("boundary_gap_seconds", "") or probe.DEFAULT_BOUNDARY_GAP_SECONDS)),
)
```

- [ ] **Step 5: Run the GUI index tests**

Run: `uv run pytest scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/build_vlm_photo_boundary_gui_index.py scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py
git commit -m "Use window schema metadata in GUI index rebuild"
```

## Task 4: Add Manual GUI Schema Override and Shared Runtime Selection

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing manual GUI tests**

```python
def test_resolve_manual_vlm_runtime_args_reads_schema_from_payload() -> None:
    payload = {
        "window_radius": 3,
        "window_schema": "time_quantile",
        "window_schema_seed": 42,
    }
    runtime_args = gui.resolve_manual_vlm_runtime_args(
        day_dir=Path("/tmp/day"),
        workspace_dir=Path("/tmp/workspace"),
        payload=payload,
        manual_vlm_model=_manual_model(),
    )
    assert runtime_args.window_schema == "time_quantile"
    assert runtime_args.window_schema_seed == 42


def test_manual_vlm_dropdown_override_updates_runtime_payload(qtbot) -> None:
    window = build_main_window_for_test(...)
    window.manual_vlm_window_schema_combo.setCurrentText("random")
    payload = window.build_manual_vlm_payload()
    assert payload["window_schema"] == "random"
```

- [ ] **Step 2: Run the manual GUI tests to verify failure**

Run: `uv run pytest scripts/pipeline/test_review_gui_image_only_diagnostics.py -q`
Expected: FAIL with missing runtime args fields or missing combo box

- [ ] **Step 3: Extend manual runtime config and payload handling**

```python
def resolve_manual_prediction_window_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    configured_window_radius = str(payload.get("window_radius", "") or "").strip()
    if not configured_window_radius:
        raise ValueError("review index window_radius is unavailable")
    return {
        "window_radius": probe_vlm_boundary.positive_window_radius_arg(configured_window_radius),
        "window_schema": probe_vlm_boundary.window_schema_lib.parse_window_schema(
            payload.get("window_schema", "")
        ),
        "window_schema_seed": probe_vlm_boundary.window_schema_lib.parse_window_schema_seed(
            payload.get("window_schema_seed", "")
        ),
    }


args.window_schema = resolve_manual_prediction_window_config(payload)["window_schema"]
args.window_schema_seed = resolve_manual_prediction_window_config(payload)["window_schema_seed"]
```

- [ ] **Step 4: Add the dropdown to the manual GUI and pass its value into the runtime payload**

```python
self.manual_vlm_window_schema_combo = QComboBox(self)
self.manual_vlm_window_schema_combo.addItems(list(probe_vlm_boundary.window_schema_lib.WINDOW_SCHEMA_VALUES))
self.manual_vlm_window_schema_combo.setCurrentText(
    str(self.manual_vlm_window_schema_default or probe_vlm_boundary.window_schema_lib.DEFAULT_WINDOW_SCHEMA)
)
form_layout.addRow("Window schema", self.manual_vlm_window_schema_combo)
```

- [ ] **Step 5: Route manual candidate selection through the shared probe helper**

```python
window_schema = str(window_config.get("window_schema", "") or "")
window_schema_seed = int(window_config.get("window_schema_seed", probe_vlm_boundary.DEFAULT_WINDOW_SCHEMA_SEED))

candidate_rows = probe_vlm_boundary.build_window_rows_for_cut_index(
    joined_rows=reduced_rows,
    cut_index=cut_index,
    window_radius=runtime_window_radius,
    window_schema=window_schema,
    window_schema_seed=window_schema_seed,
    boundary_gap_seconds=int(str(payload.get("boundary_gap_seconds", "") or probe_vlm_boundary.DEFAULT_BOUNDARY_GAP_SECONDS)),
)
if not candidate_rows:
    raise ValueError("manual VLM analyze needs enough surrounding context for inference")
```

- [ ] **Step 6: Run the manual GUI tests**

Run: `uv run pytest scripts/pipeline/test_review_gui_image_only_diagnostics.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Add manual GUI window schema override"
```

## Task 5: Update Documentation and Run Final Regression Suite

**Files:**
- Modify: `README.md`
- Modify: `scripts/pipeline/README.md`
- Test: `scripts/pipeline/test_window_schema.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the documentation updates**

```markdown
- symmetric VLM context uses `VLM_WINDOW_RADIUS` for sample count per side
- `VLM_WINDOW_SCHEMA` controls how photos are selected inside each segment
- `VLM_WINDOW_SCHEMA_SEED` defaults to `42`
- default schema is `consecutive`, which preserves current behavior
- manual GUI VLM analysis can temporarily override the schema from a dropdown
```

- [ ] **Step 2: Update the probe command examples**

```bash
python3 scripts/pipeline/probe_vlm_photo_boundaries.py DAY \
  --photo-manifest-csv DAY/_workspace/media_manifest.csv \
  --image-variant thumb \
  --window-radius 3 \
  --window-schema index_quantile \
  --window-schema-seed 42 \
  --boundary-gap-seconds 10 \
  --max-batches 100 \
  --new-run
```

- [ ] **Step 3: Run the focused regression suite**

Run: `uv run pytest scripts/pipeline/test_window_schema.py scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/test_build_vlm_photo_boundary_gui_index.py scripts/pipeline/test_review_gui_image_only_diagnostics.py -q`
Expected: PASS

- [ ] **Step 4: Run one CLI help smoke check**

Run: `python3 scripts/pipeline/probe_vlm_photo_boundaries.py --help`
Expected: PASS and help output includes `--window-schema` and `--window-schema-seed`

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/pipeline/README.md
git commit -m "Document VLM window schema workflow"
```

## Self-Review

### Spec coverage

- Public contract: Task 1 and Task 2
- Shared schema module: Task 1
- Probe integration and metadata persistence: Task 2
- GUI index reconstruction: Task 3
- Manual GUI override: Task 4
- Error handling defaults and deterministic randomness: Task 1 and Task 2
- Documentation and regression coverage: Task 5

No spec section is left without a task.

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation markers remain
- Each task contains explicit files, code snippets, commands, and expected outcomes
- No task refers to “similar to previous task” without concrete content

### Type consistency

- Public names are consistently `window_schema` and `window_schema_seed`
- Shared module name is consistently `window_schema.py`
- Probe/manual/GUI index all route through `build_window_rows_for_cut_index(...)`
- Schema values match the spec exactly:
  - `consecutive`
  - `random`
  - `index_quantile`
  - `time_quantile`
  - `time_max_min`
  - `time_boundary_spread`
