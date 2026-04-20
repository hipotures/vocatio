# Manual VLM Description Tooltip Design

## Goal

Add optional descriptive metadata for manual VLM presets so the GUI can show short preset names in the dropdown while keeping a longer human-readable description available in tooltips and result output.

This change applies only to the manual VLM preset flow in:

- `conf/manual_vlm_models.yaml`
- `scripts/pipeline/lib/manual_vlm_models.py`
- `scripts/pipeline/review_performance_proxy_gui.py`

It does not change:

- manual ML behavior
- batch `probe_vlm_photo_boundaries.py`
- `.vocatio` handling
- preset MD5 reload behavior

## Problem

`VLM_NAME` currently serves two roles at once:

- the visible dropdown label
- the full descriptive identifier for a preset

That makes the `Manual VLM analyze` combobox too wide when the user wants long preset names such as:

- `llama.cpp unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL`

The GUI needs a way to keep the dropdown compact without losing the longer explanation.

## Configuration File Change

### New Optional Field

Add a new optional field to every preset:

- `VLM_DESCRIPTION`

Example:

```yaml
models:
  - VLM_NAME: "llama.cpp gemma-4-E4B"
    VLM_DESCRIPTION: "llama.cpp unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL, temp 0, strict JSON, localhost:8002"
    VLM_PROVIDER: "llamacpp"
    VLM_BASE_URL: "http://127.0.0.1:8002"
    VLM_MODEL: "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 300
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "false"
    VLM_RESPONSE_SCHEMA_MODE: "off"
    VLM_JSON_VALIDATION_MODE: "strict"
```

### Semantics

- `VLM_NAME` is the short visible label used in the dropdown and in the top-level result summary.
- `VLM_DESCRIPTION` is the longer explanatory text used in tooltips and preserved in result metadata.

### Validation

`VLM_DESCRIPTION` is optional.

If present:

- it must be a scalar string after normalization
- an empty string is allowed but treated the same as missing

If absent:

- the preset remains valid
- GUI falls back to `VLM_NAME` anywhere a description is needed

## GUI Behavior

### Dropdown Display

The `Manual VLM analyze` combobox continues to render only:

- `VLM_NAME`

This keeps the control narrow enough for the current layout.

### Tooltip Behavior

Tooltip behavior is:

- the combobox itself shows the description for the currently selected preset
- each item in the expanded dropdown list shows its own description as a tooltip

Tooltip text source:

- use `VLM_DESCRIPTION` if present and non-empty
- otherwise fall back to `VLM_NAME`

This uses normal Qt tooltip behavior.

There is no custom `3s` hover timer.

## Result Rendering

The `Manual VLM analyze` result continues to show:

- `Model: <VLM_NAME>`

The `Model config:` block must then render all preset fields in the same order they appear in the YAML entry.

Important:

- do not sort keys alphabetically
- do not special-case `VLM_DESCRIPTION`
- preserve YAML order exactly

That means if `VLM_DESCRIPTION` appears after `VLM_NAME` in YAML, it appears in that exact place in the rendered `Model config:` block.

## Ordering Rule

The YAML field order becomes authoritative for rendering preset metadata in the GUI.

This applies to:

- `Model config:` in the manual VLM result

It does not require preserving order in any unrelated internal structures unless those structures are later used for user-visible rendering.

## Compatibility

Existing preset files without `VLM_DESCRIPTION` continue to work.

Fallback behavior:

- dropdown text still uses `VLM_NAME`
- tooltip uses `VLM_NAME`
- `Model config:` simply has no `VLM_DESCRIPTION` line unless the YAML entry defines it

## Scope of Code Changes

At minimum, this requires updates to:

- `conf/manual_vlm_models.yaml.example`
- `scripts/pipeline/lib/manual_vlm_models.py`
- `scripts/pipeline/review_performance_proxy_gui.py`
- `scripts/pipeline/test_manual_vlm_models.py`
- `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

## Verification

Verification must cover:

- preset files with and without `VLM_DESCRIPTION`
- tooltip fallback to `VLM_NAME`
- tooltip population for the currently selected combobox value
- tooltip population for dropdown items
- rendering of `Model config:` in YAML key order
- no regression in existing manual VLM preset loading and reload behavior
