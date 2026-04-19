# Manual Background Workers And Selection Summary Design

## Goal

Make both manual runtime actions in the review GUI non-blocking:

- `Manual ML prediction`
- `Manual VLM analyze`

and simplify the image-only `Selection summary` so it shows only the selected photo count plus one line per selected photo with its current set name.

## Motivation

The current manual actions execute synchronously on the GUI thread. That has three visible problems:

- the window freezes until the action finishes
- the button label and section state do not visibly update while the action is running
- keyboard and mouse interaction feel dead during inference

The current `Selection summary` is also too verbose. It duplicates timing information and shows `status` / `reason` fields that do not help with manual boundary review.

The desired outcome is:

- manual actions run in the background and the GUI stays responsive
- running state is visible immediately in the section UI
- `Selection summary` becomes shorter and more useful for set inspection

## Existing Surface To Modify

This change does not create a new info-panel subsystem. It modifies the existing image-only multi-photo info flow in:

- `scripts/pipeline/review_performance_proxy_gui.py`

Specifically:

- the existing `Selection summary` section is trimmed, not replaced
- the existing `Manual ML prediction` section stays in place and becomes asynchronous
- the existing `Manual VLM analyze` section stays in place and becomes asynchronous

No batch pipeline behavior changes. This is GUI-only runtime behavior.

## Selection Summary Contract

For image-only multi-photo selection, the `Selection summary` section becomes:

- `Selected photos: N`
- one line per selected photo in chronological order

Each selected-photo line shows:

- timestamp
- relative path
- current displayed set name

Example:

```text
Selected photos: 2
2026-03-23T08:51:28.250 | p-a7r5/20260323_085128_010_11935744.hif | set=VLM0123
2026-03-23T08:51:44.468 | p-a7r5/20260323_085144_001_18194432.hif | set=VLM0124
```

The section removes:

- `First time`
- `Last time`
- the `Selected photo rows` heading
- `status`
- `reason`

The set label must use the current displayed set identity, not stale raw payload values. If the photo belongs to a split or merged display set, the section shows that current display name.

## Background Execution Model

The GUI must use a concrete background-worker model instead of synchronous execution.

Implementation model:

- reuse `QThreadPool.globalInstance()`
- create one lightweight `QRunnable` worker type for manual actions
- create one small `QObject` signal bridge for worker completion/error callbacks

The worker receives:

- action type: `manual_ml` or `manual_vlm`
- the exact runtime inputs needed by the current manual action
- enough GUI-side snapshot data to resolve the rest of the work off the GUI thread

The worker returns:

- success payload as a plain mapping
- or error payload as a plain mapping

All GUI state mutation remains on the main thread. The worker only performs computation and emits results.

This design matches the existing file’s current use of `QThreadPool` for preview-related background work and avoids inventing a second concurrency mechanism.

## Manual Action Lifecycle

Both manual actions follow the same lifecycle:

1. User clicks the section action button.
2. GUI immediately updates the section state to `running`.
3. GUI disables both manual action buttons.
4. GUI submits a worker to the thread pool.
5. The worker performs all expensive work off the GUI thread, including runtime resolution, module reload, data loading, and inference/analysis.
6. Worker emits success or failure back to the main thread.
7. GUI updates the section state to `result` or `error`.
8. GUI re-enables both manual action buttons.

The GUI must remain responsive for:

- repaint
- scrolling
- selecting another photo
- using keyboard navigation
- opening and closing docks

## Button State Contract

Idle labels remain short:

- `Manual ML prediction` -> `Run`
- `Manual VLM analyze` -> `Analyze`

Running state must not use long text that gets clipped.

Running-state behavior:

- keep the short base label
- show a small inline spinner indicator next to the label
- disable both manual action buttons while either worker is active
- only the active section shows the spinner

Examples:

- `Run` + spinner
- `Analyze` + spinner

There is no long running label like `Running ML...` or `Analyzing VLM...`.

## Result Ownership And Selection Changes

Completed results are not discarded because the user changed selection while the worker was running.

Rules:

- do not try to cancel, discard, or re-key results when selection changes
- do not compare the finishing result against the current selection before applying it
- the latest completed run for a section simply replaces the previous runtime result for that section
- this applies even if the current selection is now a different pair or is no longer exactly two photos
- if a job finishes after the selection stops qualifying for the manual sections, the finished result must remain visible in the info panel instead of being silently hidden

This is intentionally permissive. The user explicitly prefers to keep the finished result rather than have the GUI silently throw it away.

To make provenance explicit, `Manual VLM analyze` must always render:

```text
Anchors:
  <left_relative_path> -> <right_relative_path>
```

This anchor block is required in the finished VLM result so a stale-but-intentional result can still be interpreted correctly after selection changes.

## Concurrency Rules

The asynchronous design removes the old implicit serialization from synchronous execution. The GUI needs one explicit rule:

- at most one in-flight manual job globally

That means:

- if `Manual ML prediction` is running, `Manual VLM analyze` cannot start
- if `Manual VLM analyze` is running, `Manual ML prediction` cannot start
- both buttons remain disabled until the running job finishes with either `result` or `error`

This global serialization is required because the current manual actions rely on a shared module reload seam for `probe_vlm_photo_boundaries`, and concurrent reload/use would create undefined races.

## Status Feedback Rules

Status feedback stays local to each section.

The design does not add new status-bar messages for:

- worker start
- worker completion
- worker failure

Reason:

- the status bar is already used for unrelated GUI feedback
- section-local state is enough for these manual tools
- avoiding new status-bar traffic prevents message races

The section body remains the single source of truth:

- `Status: idle`
- `Status: running`
- `Status: result`
- `Status: error`

## Manual ML Contract

The asynchronous conversion must preserve the current manual ML logic:

- same predictor execution semantics
- same reload seam for `probe_vlm_photo_boundaries`
- same rendered fields

Only execution mode changes:

- synchronous on GUI thread -> background worker

The expensive runtime resolution for manual ML also moves into the worker. The click handler does not perform heavy joined-row loading or compute preparation before setting `running`.

No change to ML semantics is part of this design.

## Manual VLM Contract

The asynchronous conversion must preserve the current manual VLM logic:

- same manual window builder
- same prompt builder
- same provider request flow
- same debug dump behavior
- same rendered output structure

Only execution mode changes:

- synchronous on GUI thread -> background worker

The expensive runtime resolution for manual VLM also moves into the worker. The click handler does not perform heavy joined-row loading, runtime argument resolution, or prompt preparation before setting `running`.

No change to prompt semantics is part of this design.

## Error Handling

Worker failures must surface as normal section errors:

- `Status: error`
- concise error text in the section body

On failure:

- stale result text must not survive
- stale resolution-error text must not survive
- stale debug file paths must not survive unless the failing run itself produced new debug artifacts

The GUI must not crash if:

- ML artifacts are missing
- VLM provider request fails
- module reload fails
- the worker raises unexpectedly

## Files In Scope

Expected implementation scope:

- `scripts/pipeline/review_performance_proxy_gui.py`
- `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

No other file is required for this design.

## Testing Expectations

Tests must prove:

- `Selection summary` no longer renders first/last time, the extra heading, or `status` / `reason`
- selected-photo lines include `set=<display_name>`
- both manual sections enter `running` without blocking the rest of the GUI update path
- both manual sections use background workers instead of direct synchronous compute from the click handler
- starting either action disables both manual action buttons
- the idle button labels remain `Run` and `Analyze`
- only the active action shows the spinner while running
- the active button keeps the same short base label while the spinner is shown
- success updates section state back on the GUI thread
- failure updates section state back on the GUI thread
- ML and VLM cannot both be in `running` state at the same time
- changing selection during a running task does not discard the finished result
- finished VLM results always render the multi-line `Anchors:` block

## Out Of Scope

This design does not include:

- changing ML or VLM decision semantics
- adding new status-bar messaging
- adding job cancellation
- adding persistent history of manual runs
- changing batch probe behavior
