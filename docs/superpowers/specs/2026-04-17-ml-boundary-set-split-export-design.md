## Summary

Fix ML boundary truth export so manual review splits created as set splits remain valid training data. The current exporter rejects a split such as `86 -> 86 + 87` because it interprets the right-side display name as a semantic segment label and only accepts `ceremony` or `warmup`. In review GUI semantics, that is wrong: a numeric `new_name` with `is_set_split=true` means a normal `performance -> performance` split and should produce a truth boundary for ML.

## Problem

Current review data contains split specs like:

- base set: `86@2026-03-23T14:58:53.378`
- split point: `start_filename = 20260323_150509_001_17403904.hif`
- `new_name = "87"`
- `is_set_split = true`

This is valid GUI behavior. The GUI explicitly enforces:

- when `is_set_split=true`, `new_name` must be digits only
- when `is_set_split=false`, `new_name` cannot be digits only

The ML exporter currently ignores `is_set_split` and treats the resulting display name as a semantic override. That leads to:

- `Unknown explicit split name for display set ...: 87`

This blocks ML truth export even though the user review state is valid.

## Goal

Keep the existing review-state schema and GUI behavior, but make ML truth export classify manual splits correctly:

- numeric set split => right segment type is `performance`
- semantic split to `ceremony` => right segment type is `ceremony`
- semantic split to `warmup` => right segment type is `warmup`

The exporter should derive truth rows from split boundaries and left/right segment types, not from display names alone.

## Non-Goals

- no review GUI changes
- no review-state schema migration
- no changes to `performance_proxy_index.json`
- no changes to candidate-gap ML dataset schema beyond already expected segment-type fields

## Recommended Approach

Introduce one small classifier in ML truth export that resolves the right-side segment type from split metadata, using both `new_name` and `is_set_split`.

Rules:

1. If `is_set_split=true`, classify the right segment as `performance`.
2. If `is_set_split=false` and `new_name=="ceremony"`, classify as `ceremony`.
3. If `is_set_split=false` and `new_name=="warmup"`, classify as `warmup`.
4. If `is_set_split=false` and `new_name` is anything else, raise a clear error.
5. If no explicit split applies, use the original/base segment type logic already used by the exporter.

This keeps the existing user workflow intact and only fixes the exporter assumption.

## Data Semantics

For ML, the numeric name itself is not the target. It is only a display/review label.

Example:

- original reviewed set: `86`
- manual split: `86 -> 86 + 87`

Expected exported semantics:

- there is a true `boundary` at the split point
- left segment type = `performance`
- right segment type = `performance`
- canonical `segment_type` for the right segment = `performance`

Example:

- original reviewed set: `86`
- manual split: `86 -> warmup`

Expected exported semantics:

- there is a true `boundary` at the split point
- left segment type = `performance`
- right segment type = `warmup`

## Implementation Scope

Only the ML-boundary worktree code changes:

- `scripts/pipeline/lib/ml_boundary_review_truth_export.py`
- `scripts/pipeline/export_ml_boundary_reviewed_truth.py` only if needed for messaging or plumbing
- exporter tests in the ML-boundary worktree

No other pipeline stages should change.

## Proposed Internal Shape

Add a helper with narrow responsibility, conceptually:

- input: one split spec plus original/base performance context
- output: right-side canonical segment type

This helper should be the only place that interprets:

- `is_set_split`
- `new_name`

That avoids having this logic spread across row-building code.

## Error Handling

Keep exporter strict, but make the error match the actual contract:

- valid numeric set split: accept
- valid semantic split: accept
- invalid semantic split name: fail with a message that references `is_set_split=false`
- malformed split spec: fail with the existing clear validation style

The old error text that implies only `ceremony` or `warmup` are valid explicit split names is no longer correct for `is_set_split=true`.

## Tests

Required tests:

1. Numeric set split is accepted
   - `new_name="87"`
   - `is_set_split=true`
   - exported right-side type is `performance`

2. Semantic ceremony split is accepted
   - `new_name="ceremony"`
   - `is_set_split=false`
   - exported right-side type is `ceremony`

3. Semantic warmup split is accepted
   - `new_name="warmup"`
   - `is_set_split=false`
   - exported right-side type is `warmup`

4. Invalid semantic split name still fails
   - `new_name="foo"`
   - `is_set_split=false`

5. Existing no-split/performance cases still export unchanged

## Risks

- If there are legacy review-state files missing `is_set_split`, exporter behavior must remain deterministic. Prefer treating missing `is_set_split` as the current explicit default used in the stored data, not guessing silently from the name unless existing code already defines that behavior.
- Existing tests may currently encode the old wrong assumption; update them to match GUI semantics.

## Success Criteria

The failing real-data case exports successfully:

- base set `86`
- split point at `20260323_150509_001_17403904.hif`
- `new_name="87"`
- `is_set_split=true`

and produces ML truth rows with a valid `performance -> performance` boundary instead of raising `Unknown explicit split name`.
