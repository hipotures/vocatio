# VLM Group Compare Prompt Design

## Goal

Replace the current boundary prompt contract with a group-comparison prompt family that matches the `VLM_WINDOW_SCHEMA` architecture.

The current prompt is semantically wrong for non-consecutive window schemas because it still asks the model to reason about:

- consecutive photos
- arbitrary `cut_after_N` positions inside the window

The new design treats each probe window as two sampled groups around one already known suspected boundary:

- `group_a` = photos sampled from the left side of the suspected boundary
- `group_b` = photos sampled from the right side of the same suspected boundary

The model decides only whether the two groups belong to the same segment or to different segments.

## Public Configuration

### New Parameters

- `VLM_PROMPT_TEMPLATE_ID`
  - public template identifier
  - default: `group_compare_long`
- `VLM_PROMPT_TEMPLATE_FILE`
  - optional explicit template file override
  - if set, overrides `VLM_PROMPT_TEMPLATE_ID`

### CLI

`probe_vlm_photo_boundaries.py` will accept:

- `--prompt-template-id`
- `--prompt-template-file`

Resolution priority:

1. CLI
2. `.vocatio`
3. default `group_compare_long`

### `.vocatio`

Supported keys:

- `VLM_PROMPT_TEMPLATE_ID`
- `VLM_PROMPT_TEMPLATE_FILE`

## Prompt Template Registry

### Registry File

Add a dedicated registry file:

- `conf/vlm_prompt_templates.yaml`

The registry is the source of truth for built-in prompt templates exposed in GUI and CLI validation.

Each entry defines:

- `ID`
- `LABEL`
- `FILE`
- `DESCRIPTION`

Initial built-in entries:

- `group_compare_long`
- `group_compare_short`

### Template Files

Prompt bodies live as plain text files in `conf/`:

- `conf/vlm_boundary_prompt.group_compare_long.txt`
- `conf/vlm_boundary_prompt.group_compare_short.txt`

These files are intended to be easy to edit and extend without touching Python code.

### Loader Module

Add a shared loader module:

- `scripts/pipeline/lib/vlm_prompt_templates.py`

Responsibilities:

- load and validate the registry YAML
- resolve `prompt_template_id -> template file`
- resolve explicit template file override
- load template text
- render placeholders for arbitrary `VLM_WINDOW_RADIUS`
- expose built-in template metadata for GUI dropdowns

## Prompt Semantics

The new prompt family is group-based, not boundary-position-based.

The model is told that:

- it receives `2 * VLM_WINDOW_RADIUS` photos divided into two groups
- `group_a` and `group_b` come from opposite sides of one suspected boundary
- photos inside each group may be non-consecutive
- photos may be selected by different sampling strategies
- the two groups may still belong to the same segment
- the model must not assume that a real boundary exists

The prompt must not mention:

- consecutive photos as a general rule
- `cut_after_N`
- `boundary_after_frame`
- searching for another boundary inside the window

The task is binary:

- `same_segment`
- `different_segments`

## Prompt Placeholders

Template rendering uses simple placeholder replacement, not a general-purpose templating engine.

Supported placeholders:

- `{{WINDOW_SIZE}}`
- `{{WINDOW_RADIUS}}`
- `{{GROUP_A_COUNT}}`
- `{{GROUP_B_COUNT}}`
- `{{GROUP_MAPPING}}`
- `{{ML_HINTS_BLOCK}}`
- `{{FRAME_NOTES_JSON_EXAMPLE}}`
- `{{SEGMENT_TYPES_INLINE}}`

### Generated Group Mapping

For `VLM_WINDOW_RADIUS = 3`, render:

- `a_01 = attached image 1`
- `a_02 = attached image 2`
- `a_03 = attached image 3`
- `b_01 = attached image 4`
- `b_02 = attached image 5`
- `b_03 = attached image 6`

This must scale to arbitrary positive radius.

### Generated Frame Notes Example

The prompt example JSON must contain exactly:

- `a_01..a_N`
- `b_01..b_N`

where `N = VLM_WINDOW_RADIUS`.

## New Response Contract

The new group-compare prompt family uses one response format only.

There is no legacy fallback and no mixed parser mode.

### Required Response JSON

```json
{
  "decision": "<same_segment|different_segments>",
  "group_a_segment_type": "<dance|ceremony|audience|rehearsal|other>",
  "group_b_segment_type": "<dance|ceremony|audience|rehearsal|other>",
  "frame_notes": [
    {"frame_id": "a_01", "group": "group_a", "note": "<short note>"},
    {"frame_id": "a_02", "group": "group_a", "note": "<short note>"},
    {"frame_id": "b_01", "group": "group_b", "note": "<short note>"},
    {"frame_id": "b_02", "group": "group_b", "note": "<short note>"}
  ],
  "primary_evidence": [
    "<short evidence item>",
    "<short evidence item>"
  ],
  "summary": "<one short sentence>"
}
```

For larger radius values the `frame_notes` list grows accordingly.

### Validation Rules

The parser must reject the response unless all of the following are true:

- `decision` is exactly `same_segment` or `different_segments`
- `group_a_segment_type` is one of:
  - `dance`
  - `ceremony`
  - `audience`
  - `rehearsal`
  - `other`
- `group_b_segment_type` is one of the same values
- `frame_notes` is a list with exact length `2 * VLM_WINDOW_RADIUS`
- every entry contains:
  - `frame_id`
  - `group`
  - `note`
- allowed `frame_id` values are exactly:
  - `a_01..a_N`
  - `b_01..b_N`
- every expected `frame_id` appears exactly once
- no extra `frame_id` values are present
- `group` must match the `frame_id` prefix:
  - `a_* -> group_a`
  - `b_* -> group_b`
- `note` must be non-empty
- `primary_evidence` must contain at least one non-empty item
- `summary` must be non-empty

If any rule fails, the batch result is:

- `response_status = invalid_response`

and debug dumping must include `*_error.txt`.

## Parser and Pipeline Mapping

The model no longer decides boundary position.

The suspected boundary is already known by construction of the two groups.

### Internal Mapping

The parser maps:

- `same_segment -> no_cut`
- `different_segments -> cut_after_<radius>`

where `<radius>` is the deterministic central split between:

- the last frame of `group_a`
- the first frame of `group_b`

For `VLM_WINDOW_RADIUS = 3`, this means:

- `different_segments -> cut_after_3`

This mapping exists only to keep current downstream pipeline fields usable.

Semantically, the system is no longer asking the model for `cut_after_N`.

## Run Metadata

Run metadata must store:

- `prompt_template_id`
- `prompt_template_file` if used
- rendered `user_prompt_template`

This ensures that auto-probe runs remain auditable and GUI index rebuild can reflect the actual prompt configuration used during the run.

## GUI Integration

### Manual GUI Prompt Template Selection

Manual GUI must support prompt templates with the same UX pattern already used for manual VLM models:

- list of available templates loaded from configuration
- dropdown selection
- reload support
- error state if template registry loading fails

The GUI must not use a hardcoded list of built-in template IDs.

The dropdown options come from:

- `conf/vlm_prompt_templates.yaml`

### Manual Runtime Override

Manual GUI selection overrides only the current manual analysis runtime.

It does not rewrite `.vocatio`.

### GUI Result and Error Panels

Manual `VLM analyze` result/error output must include:

- `VLM_PROMPT_TEMPLATE_ID`
- `VLM_PROMPT_TEMPLATE_FILE` when active

This is analogous to the existing model/window config diagnostics.

## Error Handling

### Configuration Errors

Fail fast for:

- unknown `VLM_PROMPT_TEMPLATE_ID`
- missing `VLM_PROMPT_TEMPLATE_FILE`
- malformed `vlm_prompt_templates.yaml`
- template file path that does not exist
- template file missing required placeholders

### Response Errors

Fail as `invalid_response` for:

- invalid `decision`
- invalid segment type values
- missing or duplicate `frame_id`
- invalid `group` value
- empty note
- empty `primary_evidence`
- empty `summary`
- any old legacy `cut_after_N` response format

There is no compatibility fallback.

## Testing

Minimum required coverage:

- registry YAML loader
- template ID resolution
- template file override resolution
- required placeholder validation
- prompt rendering for:
  - `VLM_WINDOW_RADIUS = 2`
  - `VLM_WINDOW_RADIUS = 3`
  - larger radius
- parser acceptance:
  - valid `same_segment`
  - valid `different_segments`
- parser rejection:
  - duplicate `frame_id`
  - missing `frame_id`
  - wrong `group`
  - wrong segment type
  - old legacy response format
- mapping:
  - `same_segment -> no_cut`
  - `different_segments -> cut_after_<radius>`
- run metadata includes prompt template fields
- manual GUI template list loading and reload
- manual GUI result/error panel shows active prompt template

## Recommended Implementation Scope

Recommended rollout:

1. add prompt template registry and template files
2. replace prompt generation with rendered group-compare templates
3. replace parser contract with the new group JSON
4. keep downstream pipeline fields via deterministic mapping only
5. add GUI template loader, dropdown, reload, and diagnostics

This yields a prompt system that matches `VLM_WINDOW_SCHEMA`, removes incorrect `cut_after_N` prompting, and keeps the rest of the pipeline stable enough to integrate incrementally.
