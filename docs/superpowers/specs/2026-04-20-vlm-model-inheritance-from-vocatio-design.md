# VLM Model Inheritance From `.vocatio`

## Goal

Unify model configuration so that:

- `conf/vlm_models.yaml` is the only source of model definitions
- `.vocatio` chooses a model preset by name and may override only allowed model parameters
- this applies to both `VLM_*` and `PREMODEL_*` model parameters
- workflow parameters remain outside this inheritance system

This change also renames the preset file used by the code:

- from `conf/manual_vlm_models.yaml`
- to `conf/vlm_models.yaml`

The code must use only `conf/vlm_models.yaml`.

## Scope

This design covers:

- VLM model config resolution
- PREMODEL model config resolution
- `.vocatio` inheritance semantics
- validation and error behavior
- migration from `manual_vlm_models.yaml` references to `vlm_models.yaml`

This design does not change:

- `VLM_WINDOW_RADIUS`
- `VLM_BOUNDARY_GAP_SECONDS`
- `PREMODEL_IMAGE_COLUMN`
- `PREMODEL_PHOTO_INDEX`
- `PREMODEL_OUTPUT_DIR`
- `PREMODEL_WORKERS`
- other workflow, batch, manifest, or data-path settings

## Configuration Sources

There are exactly two sources involved in model inheritance:

1. `conf/vlm_models.yaml`
2. `day_dir/.vocatio`

Rules:

- `conf/vlm_models.yaml` is the only source of model definitions
- `.vocatio` does not define a model from scratch
- `.vocatio` may only:
  - select a preset by `VLM_NAME` or `PREMODEL_NAME`
  - override allowed model parameters for that selected preset

There is no fallback that builds a model purely from local `.vocatio` fields.

## Exact Preset Selection

Preset lookup is strict:

- `VLM_NAME` matches a VLM preset by exact `1:1` identifier
- `PREMODEL_NAME` matches a PREMODEL preset by exact `1:1` identifier

If `.vocatio` contains:

```text
VLM_NAME=qwen3.5:9b
```

then only a preset with exactly:

```text
VLM_NAME=qwen3.5:9b
```

is valid.

These are all errors if no exact preset exists:

- `qwen3.5:9`
- `qwen3.5:19b`
- `qwen3.6:9b`

## Allowed Inherited Fields

### VLM model fields

Allowed model fields for VLM inheritance:

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

### PREMODEL model fields

Allowed model fields for PREMODEL inheritance:

- `PREMODEL_NAME`
- `PREMODEL_PROVIDER`
- `PREMODEL_BASE_URL`
- `PREMODEL_MODEL`
- `PREMODEL_MAX_OUTPUT_TOKENS`
- `PREMODEL_TEMPERATURE`
- `PREMODEL_TIMEOUT_SECONDS`

### Explicitly excluded

The following do not participate in model inheritance:

- `VLM_WINDOW_RADIUS`
- `VLM_BOUNDARY_GAP_SECONDS`
- `VLM_MAX_BATCHES`
- `VLM_DUMP_DEBUG_DIR`
- `VLM_PHOTO_MANIFEST_CSV`
- `VLM_EMBEDDED_MANIFEST_CSV`
- `VLM_IMAGE_VARIANT`
- `VLM_PHOTO_PRE_MODEL_DIR`
- `VLM_ML_MODEL_RUN_ID`
- `PREMODEL_PHOTO_INDEX`
- `PREMODEL_IMAGE_COLUMN`
- `PREMODEL_OUTPUT_DIR`
- `PREMODEL_WORKERS`

These remain ordinary workflow settings with their existing behavior.

## Resolution Rules

### VLM resolution

VLM config resolution is:

1. read `.vocatio`
2. require `VLM_NAME`
3. load the matching VLM preset from `conf/vlm_models.yaml`
4. apply local overrides from `.vocatio` only for allowed `VLM_*` model fields
5. validate the final resolved config against the chosen provider

### PREMODEL resolution

PREMODEL config resolution is:

1. read `.vocatio`
2. require `PREMODEL_NAME`
3. load the matching PREMODEL preset from `conf/vlm_models.yaml`
4. apply local overrides from `.vocatio` only for allowed `PREMODEL_*` model fields
5. validate the final resolved config against the chosen provider

These are two separate resolution paths:

- VLM resolution ignores `PREMODEL_*`
- PREMODEL resolution ignores `VLM_*`

## Example `.vocatio`

### Preset only

```text
WORKSPACE_DIR=/arch03/WORKSPACE/20260323DWC

VLM_NAME=qwen3.5:9b
PREMODEL_NAME=qwen3.5-4b-pre

VLM_WINDOW_RADIUS=3
VLM_BOUNDARY_GAP_SECONDS=15
PREMODEL_IMAGE_COLUMN=thumb_path
```

### Preset plus local override

```text
WORKSPACE_DIR=/arch03/WORKSPACE/20260323DWC

VLM_NAME=qwen3.5:9b
VLM_TEMPERATURE=1.0

PREMODEL_NAME=qwen3.5-4b-pre

VLM_WINDOW_RADIUS=3
VLM_BOUNDARY_GAP_SECONDS=15
PREMODEL_IMAGE_COLUMN=thumb_path
```

In this case:

- the base VLM config comes from the `qwen3.5:9b` preset in `conf/vlm_models.yaml`
- `VLM_TEMPERATURE=1.0` overrides the preset value locally

### Error: missing preset name

```text
VLM_TEMPERATURE=1.0
```

This is an error because `.vocatio` may not define a VLM model from scratch.

### Error: unknown preset identifier

```text
VLM_NAME=qwen3.5:19b
```

This is an error if no exact matching preset exists in `conf/vlm_models.yaml`.

## Provider Compatibility

After preset inheritance and local overrides are merged, the final config must be validated against the selected provider.

Rules:

- if a provider does not support a configured parameter, that is an error
- no configured field is silently dropped
- no configured field is silently ignored
- no automatic “best effort” coercion is attempted

Responsibility for choosing a coherent provider + parameter set belongs to the user.

## Error Behavior

The system must fail with a clear configuration error when:

- `conf/vlm_models.yaml` does not exist
- required preset name is missing:
  - `VLM_NAME`
  - `PREMODEL_NAME`
- a preset identifier does not exist in `conf/vlm_models.yaml`
- a resolved provider does not support one of the configured parameters
- a preset definition itself is invalid

These are configuration errors, not soft warnings.

## File Rename Migration

All code, tests, docs, and user-facing messages must switch to:

- `conf/vlm_models.yaml`

They must stop referring to:

- `conf/manual_vlm_models.yaml`

The temporary symlink may exist on disk during migration, but the code must not depend on it and must not mention the old filename anymore.

## Implementation Notes

Recommended internal shape:

- keep the preset loader responsible for reading `conf/vlm_models.yaml`
- add explicit helpers for:
  - `resolve_vlm_model_config(...)`
  - `resolve_premodel_model_config(...)`
- keep provider compatibility validation after inheritance is applied

This keeps the shared preset file manageable while preserving separate VLM and PREMODEL resolution paths.
