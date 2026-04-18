# Review GUI Segment Type Override Design

## Goal

Add a lightweight manual segment-type override to `review_performance_proxy_gui.py` so the reviewer can change the current set type from the keyboard without opening a dialog.

## Scope

This change is limited to the review GUI state model and presentation. It does not change ML training, VLM probe output, or review-index generation.

## User Interaction

- Keyboard shortcut: `Y`
- Target: current top-level set in the tree
- Behavior: cycle through these visible type codes:
  - `?`
  - `D`
  - `C`
  - `A`
  - `R`
  - `O`
  - back to `?`

Visible meaning:

- `?` = no manual override
- `D` = dance
- `C` = ceremony
- `A` = audience
- `R` = rehearsal
- `O` = other

`?` is a reset state, not a literal saved segment label.

## State Model

Manual override is stored in:

- `review_state["performances"][set_id]["segment_type_override"]`

Rules:

- missing / empty value means no override
- saved values are canonical lowercase labels:
  - `dance`
  - `ceremony`
  - `audience`
  - `rehearsal`
  - `other`
- resetting back to `?` removes the override by storing an empty value

Existing review-state fields remain unchanged.

## Resolution Rules

For each display set and its child photo rows:

1. Start from the type derived from the index payload:
   - `segment_type`
   - `type_code`
2. If a non-empty `segment_type_override` exists in review state, it becomes the effective type.
3. The displayed `Type` column and info panel use the effective type, not the base index type.

## GUI Presentation

### Tree

- existing `Type` column remains
- after a manual override, the row shows the override code instead of the base code
- child rows inherit the same effective type code

### Styling

- manually overridden sets are visually distinguished with `italic`
- do not use bold for this feature
- existing `no_photos_confirmed` styling should continue to work
- if both states apply, the row may be both greyed and italicized

### Info Panel

For set view:

- keep `Type: <effective code>`
- add:
  - `Type override: yes` when override exists
  - `Type override: no` otherwise

For photo view:

- keep `Type: <effective code>`
- add the same override indicator

## Status Feedback

After pressing `Y`, show a status-bar message:

- when setting an override:
  - `Type set to D for set VLM0007`
- when clearing the override:
  - `Type reset for set VLM0007`

If saving fails, keep the existing GUI pattern:

- report that the change exists in memory but save failed

## Non-Goals

- no modal chooser
- no mouse-only control
- no editing of source index JSON
- no ML retraining integration in this step
- no additional type values beyond the five VLM classes plus reset

## Files Affected

- `scripts/pipeline/review_performance_proxy_gui.py`
- tests in:
  - `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - and/or a new focused GUI-state test if needed

## Acceptance Criteria

- pressing `Y` on a set cycles through `? -> D -> C -> A -> R -> O -> ?`
- override persists in the dedicated `review_state` file
- rebuilding the tree keeps the override
- `Type` column updates immediately
- info panel reflects effective type and override status
- overridden rows are italic, not bold because of this feature
