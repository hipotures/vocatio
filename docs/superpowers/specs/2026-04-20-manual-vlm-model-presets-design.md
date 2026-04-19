# Manual VLM Model Presets Design

## Goal

Add explicit model preset selection for `Manual VLM analyze` in `review_performance_proxy_gui.py`.

The GUI must stop using `.vocatio` as the source of manual VLM model/request parameters. Instead, it must load model presets from a dedicated YAML file in the repository:

- `conf/manual_vlm_models.yaml`

This change applies only to the manual VLM action in the GUI. It does not change:

- `Manual ML prediction`
- batch `probe_vlm_photo_boundaries.py`
- `.vocatio` keys such as `VLM_WINDOW_RADIUS`
- day/workspace resolution

## Scope

This feature covers:

1. A new YAML file format for manual VLM model presets.
2. GUI loading of that file at startup to populate a dropdown.
3. Runtime detection of YAML changes via file content MD5.
4. Runtime reload of presets when the file changes.
5. Manual VLM execution using the selected preset instead of `.vocatio` model settings.
6. Result rendering that shows the selected preset name and all preset parameters.
7. Retry behavior for manual VLM failures.

It does not cover:

- changing batch probe configuration flow
- changing manual ML configuration flow
- moving all VLM config in the repo away from `.vocatio`

## Configuration File

### Path

The preset file path is fixed:

- `conf/manual_vlm_models.yaml`

There is no CLI override and no `.vocatio` override for this file.

### Top-Level Structure

The YAML structure is:

```yaml
models:
  - VLM_NAME: "Qwen2.5-VL 7B temp 0"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "qwen2.5vl:7b"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "30m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "low"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"

  - VLM_NAME: "Qwen2.5-VL GGUF think off"
    VLM_PROVIDER: "llamacpp"
    VLM_BASE_URL: "http://127.0.0.1:8080"
    VLM_MODEL: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "off"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"

  - VLM_NAME: "InternVL3 8B temp 0.2"
    VLM_PROVIDER: "vllm"
    VLM_BASE_URL: "http://127.0.0.1:8000"
    VLM_MODEL: "OpenGVLab/InternVL3-8B"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.2
    VLM_REASONING_LEVEL: "inherit"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
```

### Supported Providers

The file must support presets for all currently supported VLM backends:

- `ollama`
- `llamacpp`
- `vllm`

### Required Fields

Every preset entry must define all of these fields:

- `VLM_NAME`
- `VLM_PROVIDER`
- `VLM_BASE_URL`
- `VLM_MODEL`
- `VLM_CONTEXT_TOKENS`
- `VLM_MAX_OUTPUT_TOKENS`
- `VLM_KEEP_ALIVE`
- `VLM_TIMEOUT_SECONDS`
- `VLM_TEMPERATURE`
- `VLM_REASONING_LEVEL`
- `VLM_RESPONSE_SCHEMA_MODE`
- `VLM_JSON_VALIDATION_MODE`

There are no missing-field fallbacks to `.vocatio`.

### Validation Rules

The loader must reject the file if:

- the file does not exist
- YAML parsing fails
- `models` is missing or is not a list
- the list is empty
- any preset is not a mapping
- any required field is missing
- `VLM_NAME` is empty
- `VLM_NAME` values are duplicated

The loader may also validate obvious type/value issues, for example:

- `VLM_PROVIDER` not in the supported provider set
- `VLM_TIMEOUT_SECONDS <= 0`
- invalid numeric values for token counts or temperature

## Parameter Ownership

### Taken From YAML

Manual VLM model/request parameters must come only from the selected preset:

- `VLM_PROVIDER`
- `VLM_BASE_URL`
- `VLM_MODEL`
- `VLM_CONTEXT_TOKENS`
- `VLM_MAX_OUTPUT_TOKENS`
- `VLM_KEEP_ALIVE`
- `VLM_TIMEOUT_SECONDS`
- `VLM_TEMPERATURE`
- `VLM_REASONING_LEVEL`
- `VLM_RESPONSE_SCHEMA_MODE`
- `VLM_JSON_VALIDATION_MODE`

### Not Taken From YAML

These remain outside the preset system:

- `VLM_WINDOW_RADIUS`
- `VLM_BOUNDARY_GAP_SECONDS`
- day/workspace resolution
- selected anchors
- prompt content/rules
- debug file destination rules

For manual VLM, those still come from the existing runtime flow and day context.

## GUI Behavior

### Initial Load

At GUI startup:

1. Read `conf/manual_vlm_models.yaml`.
2. Validate it.
3. Populate a dropdown in the `Manual VLM analyze` section using `VLM_NAME`.
4. Select the first preset by default.
5. Compute and store the MD5 of the file contents.

If loading fails:

- the dropdown is empty
- `Analyze` is disabled
- the section subtitle area shows the configuration error

### Placement

The preset selector belongs to the `Manual VLM analyze` section.

It should appear in the section controls area associated with the analyze action rather than as a separate large block. The existing static gray descriptive text under the section title becomes a short runtime message area for:

- model reload notices
- configuration errors

### Runtime MD5 Reload

When the user clicks `Analyze`:

1. Compute the current MD5 of `conf/manual_vlm_models.yaml`.
2. Compare it to the stored MD5 from the last successful load.

If the MD5 is unchanged:

- continue using the currently loaded preset list and current dropdown selection

If the MD5 changed:

1. Reload the YAML file.
2. Revalidate it.
3. Refresh the dropdown.
4. Replace the stored MD5 with the new value.
5. Show a short message in the section subtitle area:
   - `Models reloaded from config.`

Selection retention rule after reload:

- if the previously selected `VLM_NAME` still exists, keep it selected
- otherwise select the first preset in the reloaded list

If reload fails:

- show the configuration error in the subtitle area
- leave `Analyze` disabled

## Manual VLM Execution

### Window Construction

The window semantics do not change.

Manual VLM analyze still:

- requires exactly 2 selected photos
- uses those as `left_anchor | right_anchor`
- uses `VLM_WINDOW_RADIUS` from the existing day/runtime config
- builds a window of size `2 * VLM_WINDOW_RADIUS`
- ignores photos between the two anchors

### Runtime Source Split

For a manual VLM request:

- model/request fields come from the selected YAML preset
- day/runtime fields still come from the existing GUI/day context

### Retry Policy

Manual VLM analyze must retry failed model calls:

- maximum attempts: `3`
- retry delay: `5` seconds between attempts

Retry applies to:

- transport failures
- timeout failures
- provider-level request errors
- empty/invalid model responses
- JSON/schema validation failures

Retry does not apply to:

- missing YAML file
- invalid YAML structure
- missing preset fields
- invalid local input
- insufficient photo context
- other local preflight failures

If all attempts fail:

- show `Status: error`
- show the final error
- keep any debug files that were produced

## Result Rendering

The `Manual VLM analyze` result text must include the selected preset and all preset parameters.

The result layout is:

```text
Status: result
Decision: cut_after_3
Segments: dance -> rehearsal
Summary: ...

Anchors:
  left_path -> right_path

Model: Qwen2.5-VL 7B temp 0
Model config:
  VLM_PROVIDER: ollama
  VLM_BASE_URL: http://127.0.0.1:11434
  VLM_MODEL: qwen2.5vl:7b
  VLM_CONTEXT_TOKENS: 16384
  VLM_MAX_OUTPUT_TOKENS: 512
  VLM_KEEP_ALIVE: 30m
  VLM_TIMEOUT_SECONDS: 180
  VLM_TEMPERATURE: 0.0
  VLM_REASONING_LEVEL: low
  VLM_RESPONSE_SCHEMA_MODE: on
  VLM_JSON_VALIDATION_MODE: strict

Attempts: 2
Succeeded on attempt: 2

Debug files:
  /tmp/...
  /tmp/...
  /tmp/...
```

Rules:

- `Model:` shows `VLM_NAME`
- `Model config:` shows all fields from the preset in the fixed configured order, not alphabetically
- `Attempts:` is always shown
- `Succeeded on attempt:` is shown only when success happened after attempt 1
- `Anchors:` stays multiline:
  - line 1: `Anchors:`
  - line 2: `  left -> right`
- there is a blank line between:
  - semantic result and `Anchors`
  - `Anchors` and `Model`
  - retry info and `Debug files`

For error output:

- `Status: error` replaces success-only fields such as final decision summary
- `Model:`
- `Model config:`
- `Attempts:`
- and `Debug files:`
  are still shown whenever available

## Interaction With Existing Async Manual Actions

This feature must fit the current async manual action model:

- only one manual action can run at a time
- if manual VLM is running:
  - `Analyze` is disabled and shows spinner
  - `Run` for manual ML is disabled without spinner
- after completion:
  - both buttons are re-enabled

This feature does not change manual ML behavior.

## Error Messaging

The short subtitle/status text under `Manual VLM analyze` must be used for:

- config load errors
- config reload errors
- successful reload notice

Examples:

- `Models reloaded from config.`
- `Model config error: missing VLM_MODEL in preset "..."` 
- `Model config error: duplicate VLM_NAME "..."` 

This replaces the idea of adding a dedicated reload button.

## Testing Expectations

Add coverage for:

1. successful YAML load at startup
2. dropdown populated with `VLM_NAME` values
3. default selection is the first preset
4. invalid YAML disables `Analyze` and shows error text
5. duplicate `VLM_NAME` is rejected
6. changed file MD5 triggers reload on `Analyze`
7. unchanged MD5 does not trigger reload
8. selection is preserved across reload when `VLM_NAME` still exists
9. selection falls back to first preset if the old name disappears
10. manual VLM request uses preset fields instead of `.vocatio` model fields
11. retry logic:
    - success on first attempt
    - success on later attempt
    - failure after 3 attempts
12. result text includes:
    - `Model`
    - all config fields
    - `Attempts`
    - `Succeeded on attempt` when applicable
13. subtitle area shows reload notice and config errors

