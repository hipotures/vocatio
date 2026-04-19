# Window Radius Contract Design

## Context

The current ML/VLM boundary flow is internally inconsistent:

- the VLM probe flow accepts external window sizing and can run on values other than `5`
- the ML flow is still hardcoded to a fixed 5-frame contract
- GUI precomputed ML hints and manual ML prediction inherit that fixed-width assumption

This creates invalid hybrid runs where:

- the VLM processes one window shape
- the ML hint path silently assumes another

That must be removed completely. The system must use one explicit external contract and fail fast on any mismatch.

## Goals

- Replace external `VLM_WINDOW_SIZE` and `VLM_OVERLAP` usage with a single public parameter: `VLM_WINDOW_RADIUS`
- Keep `window_size` only as an internal derived value:
  - `window_size = 2 * window_radius`
- Make VLM probe, ML dataset generation, ML training, ML inference, GUI precomputed ML hints, and GUI manual ML prediction all use the same radius contract
- Remove all fallback behavior that tries to bridge incompatible window shapes
- Fail fast whenever model artifacts and runtime configuration use different `window_radius`

## Non-Goals

- Backward compatibility with old external `VLM_WINDOW_SIZE`
- Backward compatibility with old external `VLM_OVERLAP`
- Silent translation from old artifacts to the new contract

## Public Contract

Externally, the system uses only:

- `.vocatio`: `VLM_WINDOW_RADIUS`
- CLI: `--window-radius`
- run metadata: `window_radius`
- training metadata: `window_radius`

Externally, the system does not use:

- `VLM_WINDOW_SIZE`
- `VLM_OVERLAP`
- `--window-size`
- `--overlap`

Internally only:

- `window_size = 2 * window_radius`

The public semantics are always symmetric around the main gap:

- `window_radius = 1` -> `window_size = 2`
- `window_radius = 2` -> `window_size = 4`
- `window_radius = 3` -> `window_size = 6`

The main boundary is always centered between the left and right anchor frames.

## Architecture Changes

### 1. Probe VLM

`probe_vlm_photo_boundaries.py` will:

- expose `--window-radius`
- read `VLM_WINDOW_RADIUS` from `.vocatio`
- derive internal `window_size = 2 * window_radius`
- remove external `overlap`
- center the main candidate gap symmetrically in the constructed window
- store only `window_radius` in run metadata and config hash inputs

Any logic that currently depends on `overlap` must be replaced with direct symmetric placement around the main candidate gap.

### 2. ML Dataset and Features

The ML dataset and feature pipeline must no longer assume a fixed 5-frame contract.

Dynamic schema generation must cover:

- candidate CSV headers
- frame path fields
- timestamp fields
- photo id fields
- image feature columns
- gap features
- heuristic pair features
- per-side descriptor features

Examples:

- `window_radius = 2` -> frames `01..04`, gaps `12..34`
- `window_radius = 3` -> frames `01..06`, gaps `12..56`

The implementation must generate these fields from the radius-derived frame count instead of using hardcoded frame positions.

### 3. ML Training and Inference

Training artifacts must record:

- `window_radius`

All ML consumers must validate:

- `runtime_window_radius == model_window_radius`

If the values differ:

- raise an explicit error
- do not adapt
- do not truncate
- do not pad
- do not recompute with a different shape

This applies to:

- probe-time ML hint generation
- GUI index ML hint precompute
- GUI manual ML prediction
- any evaluation path that loads trained artifacts

### 4. GUI Index Builder

`build_vlm_photo_boundary_gui_index.py` must build precomputed ML hints using the same radius contract as the run that produced the VLM output.

It must:

- read `window_radius` from run metadata
- derive the internal frame count
- build ML candidate windows using the same symmetric gap-centered contract
- require trained ML artifacts with matching `window_radius`

If there is a mismatch:

- write a clear `ml_hints_error`
- do not write fake or partial `ml_hint_pairs`

### 5. GUI Manual ML Prediction

`review_performance_proxy_gui.py` manual ML prediction must:

- read `VLM_WINDOW_RADIUS` from `.vocatio`
- derive internal `window_size`
- ignore removed `overlap`
- build symmetric context around the manually selected gap
- require trained ML artifacts with matching `window_radius`

It must not:

- accept old size/overlap config
- adapt one radius to another
- use hidden fallback logic

## Fail-Fast Rules

There are no compatibility fallbacks.

Errors must be explicit in these cases:

- runtime `window_radius` missing or invalid
- model metadata missing `window_radius`
- model `window_radius` differs from runtime `window_radius`
- run metadata missing `window_radius`
- old artifacts are used in new flow without regeneration

Error messages must name the actual conflicting values.

## Migration and Artifact Validity

After this refactor, old artifacts built under the previous mixed contract are not valid for the new ML/VLM handoff.

Artifacts that must be regenerated:

- ML candidate corpora
- trained ML models
- ML evaluation/training reports tied to old fixed-width schema
- GUI indexes with precomputed ML hints
- any downstream flow depending on those ML hint artifacts

Artifacts that may still be usable in isolation:

- raw VLM probe outputs, but only if treated strictly as VLM-only outputs and not mixed with incompatible ML hints

The practical migration order is:

1. regenerate ML corpus under the new radius contract
2. retrain ML models
3. rebuild VLM outputs / GUI index ML hints as needed
4. run GUI review on regenerated artifacts

## Testing Requirements

Coverage must include:

- radius-driven CLI/config parsing
- symmetric window construction for multiple radius values
- dynamic ML feature schema generation for multiple radius values
- training metadata persistence of `window_radius`
- mismatch detection between runtime and model metadata
- GUI index ML hint precompute for non-default radius values
- GUI manual ML prediction for non-default radius values
- removal of fallback behaviors that previously masked inconsistency

At minimum, tests should explicitly cover:

- `window_radius = 1`
- `window_radius = 2`
- `window_radius = 3`

## Acceptance Criteria

- No external flow accepts `VLM_WINDOW_SIZE` or `VLM_OVERLAP`
- All external configuration and metadata use only `window_radius`
- VLM and ML use the same symmetric gap-centered contract
- ML works correctly for non-default radius values
- Incompatible model/runtime radius combinations fail immediately with explicit errors
- GUI startup does not perform hidden ML recomputation
- Precomputed ML hints are either valid for the active radius or explicitly absent with a clear error
