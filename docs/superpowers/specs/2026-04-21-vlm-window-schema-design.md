# VLM Window Schema Design

## Context

The current VLM boundary workflow exposes one public sizing control:

- `.vocatio`: `VLM_WINDOW_RADIUS`
- CLI: `--window-radius`

That contract is already correct and should remain unchanged:

- `VLM_WINDOW_RADIUS` means how many photos are selected on the left side
- `VLM_WINDOW_RADIUS` means how many photos are selected on the right side

The limitation is not the radius itself. The limitation is that there is only one sampling policy for choosing those photos. In practice, many consecutive photos inside one segment are nearly identical, especially in burst sequences. A fixed "nearest consecutive photos around the gap" policy is sometimes too narrow and can hide useful segment-level context.

The goal is to keep `VLM_WINDOW_RADIUS` exactly as it is and add a second public axis that controls **how photos are chosen inside each segment**.

## Goals

- Keep the public meaning of `VLM_WINDOW_RADIUS` unchanged
- Add a new public parameter that controls the photo selection strategy inside each segment
- Keep current behavior as the default when the new parameter is absent
- Make the new contract available to:
  - auto VLM probe runs
  - GUI index reconstruction paths that depend on run metadata
  - manual GUI VLM analysis
- Keep manual GUI testing flexible by allowing temporary runtime override through a dropdown
- Make random selection deterministic
- Keep output photo order chronological in prompts and downstream artifacts

## Non-Goals

- Changing the meaning of `VLM_WINDOW_RADIUS`
- Changing current default behavior
- Adding hidden heuristic fallbacks between schemas
- Replacing the current candidate-gap detection rule
- Making ML artifact `window_radius` semantics more flexible

## Public Contract

Add two new public workflow settings:

- `VLM_WINDOW_SCHEMA`
- `VLM_WINDOW_SCHEMA_SEED`

Allowed `VLM_WINDOW_SCHEMA` values:

- `consecutive`
- `random`
- `index_quantile`
- `time_quantile`
- `time_max_min`
- `time_boundary_spread`

Defaults:

- `VLM_WINDOW_SCHEMA=consecutive` when absent
- `VLM_WINDOW_SCHEMA_SEED=42` when absent

Public semantics:

- `VLM_WINDOW_RADIUS` continues to mean the requested number of photos per side
- `VLM_WINDOW_SCHEMA` controls which photos are selected from the left and right segments
- `VLM_WINDOW_SCHEMA_SEED` affects only schemas that depend on randomness
- all schemas return photos in chronological order before prompt construction
- if a side has fewer than `VLM_WINDOW_RADIUS` photos available, select all available photos on that side without padding

Example `.vocatio`:

```text
WORKSPACE_DIR=/arch03/WORKSPACE/20260323DWC

VLM_NAME=qwen3.5:9b
VLM_WINDOW_RADIUS=3
VLM_WINDOW_SCHEMA=index_quantile
VLM_WINDOW_SCHEMA_SEED=42
VLM_BOUNDARY_GAP_SECONDS=15
```

## Segment Semantics

For every analyzed gap, define:

- left segment
- right segment

A segment is the maximal consecutive run of photos bounded by neighboring gaps whose size is at least `VLM_BOUNDARY_GAP_SECONDS`.

Each schema operates independently per side:

- choose up to `VLM_WINDOW_RADIUS` photos from the left segment
- choose up to `VLM_WINDOW_RADIUS` photos from the right segment

After both sides are selected:

- combine the left and right selections
- sort the final selected rows chronologically
- build the prompt from that ordered list

This keeps the prompt stable and preserves potentially useful "late in segment" information even when the internal selection logic is non-consecutive.

## Schema Definitions

### `consecutive`

Current behavior preserved exactly.

Selection policy:

- left segment: choose up to `N` photos nearest to the gap, biased toward the segment end
- right segment: choose up to `N` photos nearest to the gap, biased toward the segment start

Equivalent phrasing:

- left segment: last `min(N, available)` photos
- right segment: first `min(N, available)` photos

If `N` covers the whole segment, the whole segment is selected naturally.

### `random`

Selection policy:

- choose `min(N, available)` unique photos uniformly from the entire segment
- no duplicates
- deterministic for the same input and seed

Chronological ordering is applied after the selection step.

### `index_quantile`

Selection policy:

- choose `min(N, available)` photos approximately evenly spaced by index position inside the segment
- this is the index-based analogue of "beginning / middle / end"

Tie-break rule:

- when two candidates are equally valid for the target index position, choose the photo that is farther from the analyzed gap

### `time_quantile`

Selection policy:

- choose `min(N, available)` photos approximately evenly spaced across the time span of the segment using timestamps

Tie-break rule:

- when two candidates are equally valid for the target time position, choose the photo that is farther from the analyzed gap

### `time_max_min`

Selection policy:

- choose `min(N, available)` photos that maximize the minimum timestamp distance between already selected photos
- this is the spread-maximizing time-based strategy

Tie-break rule:

- when multiple candidates produce the same score, choose the photo that is farther from the analyzed gap

### `time_boundary_spread`

Selection policy:

- always include the photo nearest to the analyzed gap on that side
- choose the remaining `N - 1` photos to maximize timestamp spread

Tie-break rule:

- when multiple candidates produce the same score, choose the photo that is farther from the analyzed gap

## Deterministic Randomness

`random` must be deterministic for reproducibility.

Public seed:

- `VLM_WINDOW_SCHEMA_SEED`

Default:

- `42`

Expected behavior:

- the same segment input, schema, side, and seed must produce the same selected photos
- changing the seed may change the selection
- when `available <= N`, the schema degenerates to "select all available photos", so randomness is effectively irrelevant

The implementation may combine the public seed with stable segment-local identifiers if needed, but the user-visible seed contract remains a single integer parameter.

## Architecture

Introduce one shared selection layer rather than duplicating schema logic in each caller.

### New shared module

Create a dedicated library module, for example:

- `scripts/pipeline/lib/window_schema.py`

Responsibilities:

- define the allowed schema names
- resolve defaults and validate config values
- implement per-segment selection functions for each schema
- provide shared helpers for:
  - deterministic randomness
  - distance-from-gap tie-breaks
  - chronological normalization
  - timestamp validation for time-based schemas

### Probe integration

`probe_vlm_photo_boundaries.py` remains responsible for:

- reading workflow config
- detecting candidate gaps
- invoking VLM inference

It should delegate segment-side row selection to the shared schema layer.

Probe responsibilities after the change:

- resolve `window_radius`
- resolve `window_schema`
- resolve `window_schema_seed`
- detect left/right segments around each candidate gap
- ask the shared selector for left and right photo samples
- keep the final prompt rows chronologically ordered
- persist `window_schema` and `window_schema_seed` in run metadata

### GUI index integration

`build_vlm_photo_boundary_gui_index.py` must use the same run-time schema contract recorded by the VLM run.

Responsibilities:

- read `window_radius` from run metadata as today
- additionally read `window_schema`
- additionally read `window_schema_seed`
- reconstruct candidate rows using the same shared selector semantics
- fail clearly if required run metadata is missing or invalid

This avoids semantic drift between the original run and GUI-side reconstruction.

### Manual GUI integration

`review_performance_proxy_gui.py` must support two behaviors:

- default to the day-level `.vocatio` schema contract
- allow temporary manual override from the GUI

Required UX:

- add a dropdown with the public flat enum values
- initial dropdown value:
  - `.vocatio` `VLM_WINDOW_SCHEMA` if present
  - otherwise `consecutive`
- changing the dropdown affects only the current manual VLM analysis request
- the override does not rewrite `.vocatio`

Manual runtime should use the same shared selection module as probe and GUI index.

## Run Metadata Contract

VLM run metadata currently persists `window_radius` and other workflow settings. Extend that metadata to persist:

- `window_schema`
- `window_schema_seed`

These values are part of the effective runtime configuration and must be treated as authoritative by downstream consumers.

If a downstream flow reconstructs VLM windows from stored runs, it must use:

- stored `window_radius`
- stored `window_schema`
- stored `window_schema_seed`

## Error Handling

### Config validation

Raise explicit configuration errors for:

- unknown `VLM_WINDOW_SCHEMA`
- non-integer `VLM_WINDOW_SCHEMA_SEED`
- invalid `VLM_WINDOW_RADIUS`

Error messages should enumerate allowed schema values when relevant.

### Small segments

No error when a segment has fewer than `VLM_WINDOW_RADIUS` photos.

Rule:

- choose all available photos on that side
- never pad with duplicates
- never synthesize missing rows

This preserves current behavior.

### Time-based schema requirements

Time-based schemas require valid timestamps.

If timestamps needed by a selected time-based schema are unavailable or invalid:

- fail fast for that run or manual analysis
- do not silently degrade to another schema
- do not fall back to index-based behavior

### Metadata mismatch

Downstream reconstruction flows must fail explicitly if stored metadata is incomplete or inconsistent.

This includes:

- missing `window_schema`
- missing `window_schema_seed` when needed by reconstruction logic
- run metadata that cannot be validated against expected schema constraints

No silent compatibility bridge should be added.

## Testing Requirements

### Unit tests

Add focused tests for the shared selection layer:

- schema parser defaults:
  - missing schema => `consecutive`
  - missing seed => `42`
- schema enum validation
- `consecutive` preserves current behavior
- `random`:
  - deterministic under fixed seed
  - no duplicates
  - full-set degeneration when `available <= radius`
- `index_quantile`:
  - even index spread
  - tie-break favors the photo farther from gap
- `time_quantile`:
  - even time spread
  - tie-break favors the photo farther from gap
- `time_max_min`:
  - spread-maximizing time selection
  - tie-break favors the photo farther from gap
- `time_boundary_spread`:
  - always includes the photo nearest to gap
  - remaining picks maximize time spread

### Integration tests

Add integration coverage for:

- `.vocatio` defaults and explicit schema overrides in probe config resolution
- probe run metadata persistence of `window_schema` and `window_schema_seed`
- GUI index reading and reconstructing the schema contract from run metadata
- manual GUI runtime override through dropdown selection
- deterministic `random` behavior across repeated runs with the same seed

### Regression tests

Preserve current behavior with explicit tests that prove:

- absent `VLM_WINDOW_SCHEMA` behaves exactly like current `consecutive`
- current `window_radius` semantics remain unchanged
- small-segment behavior remains unchanged

## Acceptance Criteria

- `VLM_WINDOW_RADIUS` keeps its current meaning
- missing `VLM_WINDOW_SCHEMA` preserves current behavior
- the public schema enum is flat and exactly:
  - `consecutive`
  - `random`
  - `index_quantile`
  - `time_quantile`
  - `time_max_min`
  - `time_boundary_spread`
- `VLM_WINDOW_SCHEMA_SEED` defaults to `42`
- `random` is deterministic
- all schemas operate on the left and right segments defined by neighboring large gaps
- all outputs remain chronologically ordered before prompt construction
- manual GUI exposes the schema choices in a dropdown and can temporarily override the day default
- probe and downstream GUI reconstruction use one shared selection implementation
- time-based schemas fail explicitly when required timestamps are unavailable
