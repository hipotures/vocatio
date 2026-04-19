# Review GUI Manual ML Prediction Design

## Goal

Add a lightweight manual ML prediction flow to the review GUI info panel so a reviewer can:

- inspect current diagnostics in smaller copyable sections,
- trigger an on-demand ML prediction for exactly two selected photos,
- see the result inline without modifying the existing index or review-state formats.

The change must stay local to the GUI. It must not rebuild the broader probe/index flow and must not introduce persisted history for manual predictions.

## Scope

In scope:

- replace the single large info text block with sectioned panels inside the existing `I` panel,
- make each section header clickable and copy that section's text to the clipboard,
- show a status-bar confirmation after copying,
- show a `Manual ML Prediction` section only when exactly two photos are selected,
- run an on-demand ML prediction when the user clicks the section button,
- show the manual prediction result inline in the info panel,
- keep the result in memory only for the current GUI session and current selection context.

Out of scope:

- storing manual prediction history in `review_state.json`,
- extending GUI index JSON payload structure,
- rebuilding probe/index pipelines,
- changing the trained ML feature contract,
- adding a separate dialog, report, or persistent artifact for the manual prediction.

## Constraints

- Use only existing sources and runtime defaults.
- If `.vocatio` defines `VLM_WINDOW_SIZE` and `VLM_OVERLAP`, use those.
- If those keys are missing, fall back to the same defaults already used by the existing code.
- Do not add new config fields to the GUI index payload for this feature.
- The action is only meaningful for exactly two selected photos.

## User Experience

### Info Panel Layout

The current info panel content will be split into small sections rendered inside the existing dock.

Each section contains:

- a small clickable title bar,
- a short description,
- a text body.

Clicking a section title copies only that section body to the clipboard and shows a status-bar message such as:

- `Copied ML hints`
- `Copied Boundary diagnostics`

The titles behave like lightweight copy actions, not collapsible accordions.

### Section Set

For a selected set:

- `Set summary`
- `Boundary diagnostics`
- `ML hints`
- `Manual ML Prediction` only when exactly two photos are selected

For a selected single photo:

- `Photo summary`
- `Boundary diagnostics`
- `ML hints`

For multiple selected photos:

- `Selection summary`
- `Manual ML Prediction` only when exactly two photos are selected

### Manual ML Prediction Trigger

The `Manual ML Prediction` section appears only when:

- the current GUI source mode is image-only review,
- exactly two photos are selected.

The section includes:

- a short description,
- a button labeled `ML Prediction`.

When pressed:

- the button becomes disabled,
- the button shows a busy state with a spinner-like indicator and `Computing...`,
- the GUI may block while inference runs,
- when finished, the button returns to the normal state,
- the result is rendered in the same section.

The result is not stored in review state and does not persist across restarts.

## Manual Prediction Semantics

### Photo Selection

The feature accepts exactly two selected photos from the same day.

The photos do not need to be consecutive in the day ordering.

The GUI sorts the two photos by day order and treats them as:

- `left_anchor`
- `right_anchor`

### Gap Definition

The manual prediction evaluates the gap directly between the two selected anchor photos.

Important rule:

- any photos located between the selected anchors are ignored as the interior of that gap.

This supports workflows where one or more intermediate frames are bad, blurred, or otherwise unsuitable, and the reviewer wants to test the gap using cleaner anchor frames.

### Window Construction

The manual prediction uses the same window contract as the existing VLM/ML flow:

- `VLM_WINDOW_SIZE`
- `VLM_OVERLAP`

Source of these values:

1. `.vocatio`
2. existing code defaults if `.vocatio` does not define them

No new payload fields are introduced for this.

The two selected anchors define the manual gap. The prediction path reconstructs an ML input window around that manual gap using the existing parameter logic and current day ordering, while ignoring photos inside the anchor gap itself.

### Output

The manual prediction displays:

- boundary prediction,
- boundary confidence,
- segment type prediction,
- segment type confidence,
- gap seconds,
- the chosen anchor identities.

If inference fails, the section shows a concise error message instead of a result.

## Data Flow

### Existing Diagnostics

The GUI continues to use the existing startup-loaded data sources:

- index payload,
- image-only diagnostics from workspace files,
- ML hint data already available through current GUI startup loading.

No behavior changes are required there beyond presenting the information in sectioned form instead of a single large text block.

### Manual Prediction

The manual prediction is computed on click, inside the GUI process, using existing ML inference helpers and existing configuration resolution.

It is an ephemeral runtime action:

- no history,
- no review-state writes,
- no index rebuild,
- no extra CSV/JSON artifacts.

## UI Component Strategy

The info dock should stop using a single `QLabel` for its content.

Instead it should use a scrollable section-based widget composed from small reusable parts:

- clickable header,
- description label,
- plain text body,
- optional action button for the manual prediction section.

This solves both needs:

- better structure and scanability,
- copyable section content without requiring text selection.

The section body can still be plain text. No rich HTML editor behavior is required.

## Error Handling

If the manual prediction cannot run:

- missing ML model context,
- invalid selected photos,
- unresolved config,
- inference error,

the section should render a concise failure message and re-enable the button.

The status bar should remain reserved for short action feedback, not stack traces.

## Testing

Required coverage:

- section rendering for set/photo/multi-photo cases,
- copy-on-title behavior for individual sections,
- status-bar feedback after section copy,
- conditional appearance of `Manual ML Prediction` only for exactly two selected photos,
- busy-state button behavior,
- manual prediction result rendering,
- error rendering for failed manual prediction,
- parameter resolution from `.vocatio` with fallback to existing defaults,
- non-consecutive selected photos producing a manual gap while ignoring interior photos.

## Rollout

This feature is GUI-only and additive.

It should be implemented without changing:

- review-state format,
- index payload format,
- probe output format,
- ML training artifacts.

That keeps the change isolated and reversible.
