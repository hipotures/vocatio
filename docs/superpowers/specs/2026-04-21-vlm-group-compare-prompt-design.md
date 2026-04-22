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

- `id`
- `label`
- `file`
- `description`

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

- it receives two groups of sampled photos
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
- `{{GROUP_A_IDS}}`
- `{{GROUP_B_IDS}}`
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

This must scale to arbitrary positive group sizes.

### Generated Frame Notes Example

The prompt example JSON must contain exactly:

- the runtime `group_a_ids`
- the runtime `group_b_ids`

For the current rollout, built-in samplers still produce symmetric groups of size `VLM_WINDOW_RADIUS`, but renderer and parser logic are defined over actual `group_a_ids` and `group_b_ids`, not over radius-derived names alone.

## Response Contract Identity

The new prompt family must carry an explicit response contract identifier.

Initial value:

- `response_contract_id = grouped_v1`

This identifier is stored in run metadata and used explicitly by parser and GUI/index diagnostics.

Parser selection must not depend on inferring semantics from `prompt_template_id` alone.

## ML Hint Label Mapping

The prompt label space differs from the current ML boundary verifier label space.

Runtime ML hints must use an explicit semantic mapping:

- `performance -> dance`
- `warmup -> rehearsal`
- `ceremony -> ceremony`

There is no direct ML class today for:

- `audience`
- `other`

When the ML stack cannot produce a direct mapped type hint for one of these categories, the type hint must be rendered as unavailable rather than guessed.

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

For larger group sizes the `frame_notes` list grows accordingly.

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
- `frame_notes` is a list with exact length `len(group_a_ids) + len(group_b_ids)`
- every entry contains:
  - `frame_id`
  - `group`
  - `note`
- allowed `frame_id` values are exactly the runtime:
  - `group_a_ids`
  - `group_b_ids`
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

Before finalizing `invalid_response`, the runtime performs one structural repair retry for structural contract failures such as:

- missing `frame_id`
- duplicate `frame_id`
- malformed `frame_notes` shape

There is no semantic fallback and no legacy-format salvage path.

## Parser and Pipeline Mapping

The model no longer decides boundary position.

The suspected boundary is already known by construction of the two groups.

### Semantic Fields and Internal Mapping

The parser must expose explicit semantic fields alongside compatibility projection fields:

- `semantic_decision`
- `semantic_group_a_segment_type`
- `semantic_group_b_segment_type`
- `response_contract_id`

The parser maps:

- `same_segment -> no_cut`
- `different_segments -> cut_after_<group_a_count>`

where `<group_a_count>` is the deterministic central split between:

- the last frame of `group_a`
- the first frame of `group_b`

For `VLM_WINDOW_RADIUS = 3`, this means:

- `different_segments -> cut_after_3`

This mapping exists only to keep current downstream pipeline fields usable.

The old fields:

- `decision`
- `cut_after_local_index`
- `cut_after_global_row`
- `cut_left_relative_path`
- `cut_right_relative_path`

are compatibility projection fields only. They must not be treated as the primary semantics of the model response.

Semantically, the system is no longer asking the model for `cut_after_N`.

## Run Metadata

Run metadata must store:

- `prompt_template_id`
- `prompt_template_file` if used
- `response_contract_id`
- rendered `rendered_user_prompt`

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

There is no compatibility fallback and no legacy salvage path.

## Testing

Minimum required coverage:

- registry YAML loader
- template ID resolution
- template file override resolution
- required placeholder validation
- explicit `response_contract_id` persistence
- ML label-space mapping
- prompt rendering for:
  - `VLM_WINDOW_RADIUS = 2`
  - `VLM_WINDOW_RADIUS = 3`
  - larger radius
  - empty `ML_HINTS_BLOCK`
- parser acceptance:
  - valid `same_segment`
  - valid `different_segments`
- parser rejection:
  - duplicate `frame_id`
  - missing `frame_id`
  - wrong `group`
  - wrong segment type
  - old legacy response format
- structural repair retry for malformed frame notes
- mapping:
  - `same_segment -> no_cut`
  - `different_segments -> cut_after_<group_a_count>`
- run metadata includes prompt template fields
- run metadata includes `response_contract_id`
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
