# VLM Transport Design

Date: 2026-04-15

## Goal

Introduce one narrow, provider-neutral multimodal transport layer for the two existing VLM workflows:

- `scripts/pipeline/build_photo_pre_model_annotations.py`
- `scripts/pipeline/probe_vlm_photo_boundaries.py`

The transport layer must support:

- `ollama`
- `llamacpp`
- `vllm`

The scope is intentionally narrow:

- one neutral request/response contract
- no task-specific prompt building inside the transport
- no attempt to generalize all future multimodal workflows in the repository

The domain logic stays in the calling scripts.

## Non-Goals

This design does not include:

- redesign of prompt contents
- redesign of output CSV or JSON artifacts
- migration of GUI tooling
- unification of all model benchmark scripts
- provider-specific optimization beyond what is needed to preserve current behavior

## Current Problems

Today the two VLM entrypoints already expose provider-oriented configuration through CLI and `.vocatio`, but execution is still effectively hard-wired per script:

- `probe_vlm_photo_boundaries.py` only executes against `ollama`
- `build_photo_pre_model_annotations.py` only executes against `llamacpp`

This causes three concrete problems:

1. configuration claims are broader than runtime support
2. request/response logic is duplicated and diverging
3. adding `vllm` would require another one-off implementation path

## Recommended Approach

Use one thin transport adapter at the request/response layer.

The transport layer is responsible for:

- validating the selected provider
- mapping a neutral request into provider-specific HTTP payloads
- sending the request
- normalizing the response
- normalizing provider metrics when available
- surfacing transport/configuration errors in a consistent way

The transport layer is not responsible for:

- building prompts
- interpreting domain meaning of model output
- deciding how a script stores its results
- defining task-specific schemas

## Architecture

Add a new module:

- `scripts/pipeline/lib/vlm_transport.py`

Optional small companion module if needed after implementation starts:

- `scripts/pipeline/lib/vlm_capabilities.py`

The transport module should expose:

- one neutral request type
- one neutral response type
- one provider capability descriptor
- one common execution entrypoint

Recommended public surface:

- `VlmRequest`
- `VlmResponse`
- `VlmCapabilities`
- `VlmTransportError`
- `get_vlm_capabilities(provider: str) -> VlmCapabilities`
- `run_vlm_request(request: VlmRequest) -> VlmResponse`

## Neutral Request Contract

`VlmRequest` should contain only fields already needed by the current two workflows.

Required fields:

- `provider`
- `base_url`
- `model`
- `messages`
- `image_paths`
- `timeout_seconds`

Optional fields:

- `response_format`
- `temperature`
- `context_tokens`
- `max_output_tokens`
- `reasoning_level`
- `keep_alive`

Field semantics:

- `provider`
  - one of `ollama`, `llamacpp`, `vllm`
- `messages`
  - already task-built chat message list
- `image_paths`
  - local image paths to be attached to the request
- `response_format`
  - neutral structured-output request, if the task wants JSON shaping
- `temperature`
  - provider-neutral sampling knob
- `context_tokens`
  - neutral name replacing provider-specific `num_ctx`
- `max_output_tokens`
  - neutral name replacing provider-specific `num_predict` or `max_tokens`
- `reasoning_level`
  - neutral name for current think/reasoning controls
- `keep_alive`
  - lifetime hint only for providers that support it

## Neutral Response Contract

`VlmResponse` should expose:

- `provider`
- `model`
- `text`
- `json_payload`
- `finish_reason`
- `metrics`
- `raw_response`

Field semantics:

- `text`
  - normalized final text content, if present
- `json_payload`
  - parsed JSON object if the returned text is valid JSON
- `finish_reason`
  - normalized best-effort terminal status such as `stop`, `length`, or provider-specific equivalent
- `metrics`
  - best-effort normalized metrics only when returned by the backend
- `raw_response`
  - full backend payload for debugging and artifact dumps

## Metrics Contract

Metrics should only be normalized where the backend already provides something equivalent.

Recommended normalized metric keys:

- `prompt_tokens`
- `completion_tokens`
- `total_duration_seconds`
- `eval_duration_seconds`
- `tokens_per_second`

If a provider does not return one of these, the field stays absent.

The transport must not guess missing values.

## Capability Model

Use a small capability object, not a full framework.

`VlmCapabilities` should contain:

- `supports_json_schema`
- `supports_json_object`
- `supports_reasoning_control`
- `supports_keep_alive`
- `supports_multi_image`

Purpose:

- let calling scripts reject incompatible configurations early
- avoid silent provider degradation when a task asks for unsupported behavior

This is not intended to become a dynamic plugin system.

## Provider Mapping

### Ollama

Transport style:

- native Ollama API
- `/api/chat`

Mapping:

- `context_tokens` -> `options.num_ctx`
- `max_output_tokens` -> `options.num_predict`
- `temperature` -> `options.temperature`
- `reasoning_level` -> current think/reasoning mapping
- `keep_alive` -> `keep_alive`

Structured output:

- preserve current behavior already used by `probe_vlm_photo_boundaries.py`

### llama.cpp

Transport style:

- OpenAI-compatible chat endpoint

Mapping:

- `max_output_tokens` -> `max_tokens`
- `temperature` -> `temperature`
- `response_format` -> OpenAI-compatible structured-output field

Notes:

- use the request style already exercised by the existing llama.cpp benchmark scripts
- keep provider-specific raw payloads available because llama.cpp returns useful metrics

### vLLM

Transport style:

- OpenAI-compatible chat endpoint

Mapping:

- `max_output_tokens` -> `max_tokens`
- `temperature` -> `temperature`
- `response_format` -> OpenAI-compatible structured-output field

Notes:

- even though vLLM and llama.cpp are both OpenAI-compatible, keep them as separate providers inside the transport
- this avoids conflating metric extraction and capability differences

## Configuration Model

The existing `.vocatio` and CLI surface stays conceptually intact.

Current names already in use should remain valid:

- `PREMODEL_PROVIDER`
- `PREMODEL_BASE_URL`
- `PREMODEL_MODEL`
- `VLM_PROVIDER`
- `VLM_BASE_URL`
- `VLM_MODEL`

The scripts should continue to resolve config with the same precedence:

1. CLI
2. `.vocatio`
3. hardcoded defaults

The transport sits below that layer and receives already-resolved values.

## Integration Plan

Integration should be staged.

### Stage 1: Add transport module and tests

Deliver:

- `vlm_transport.py`
- provider mapping tests
- fixture-style response normalization tests

No behavior changes in the main scripts yet.

### Stage 2: Migrate pre-model annotations

Target:

- `build_photo_pre_model_annotations.py`

Why first:

- simpler request shape
- one image per request
- existing llama.cpp benchmarks already cover the real backend shape

Result:

- script still owns prompt/schema logic
- transport owns provider execution

### Stage 3: Migrate boundary probing

Target:

- `probe_vlm_photo_boundaries.py`

Why last:

- richer control surface
- structured output
- resume and CSV logging
- more domain-specific parsing

Result:

- remove direct dependency on `ollama_post_json()`
- convert provider-specific argument usage to neutral transport inputs

## Pre-Model Script Expectations

After migration, `build_photo_pre_model_annotations.py` should:

- continue saving one JSON file per image
- continue supporting resume and overwrite behavior
- continue supporting `.vocatio` defaults
- support `llamacpp` and future `vllm`/`ollama` execution through the same transport path

The script should not gain task-specific provider branches after migration.

## Boundary Probe Script Expectations

After migration, `probe_vlm_photo_boundaries.py` should:

- continue producing `vlm_boundary_results.csv`
- continue producing run metadata under `vlm_runs/`
- continue supporting `.vocatio`
- continue supporting `max_batches`, resume, debug dumps, and response parsing
- gain real multi-provider execution behind the existing `provider` field

The parser and downstream CSV semantics should remain unchanged.

## Error Handling

Add one common transport error type:

- `VlmTransportError`

Recommended categories:

- `connection`
- `http`
- `timeout`
- `invalid_response`
- `unsupported_configuration`

The calling script decides whether to fail the run, mark a row invalid, or continue.

The transport layer should not silently swallow backend failures.

## Testing Strategy

Testing should be split into three layers.

### Transport Contract Tests

Add direct tests for:

- request mapping for `ollama`
- request mapping for `llamacpp`
- request mapping for `vllm`
- response normalization for representative payload fixtures
- capability checks
- error categorization

These tests should not require live model servers.

### Script Regression Tests

Preserve and extend existing script tests for:

- `.vocatio` defaults
- argument precedence
- structured output handling
- output file behavior
- CSV/run artifact behavior

The goal is to prove that migration does not change task semantics.

### Live Benchmark Scripts

Keep the current benchmark/test scripts as manual runtime validation tools.

They are useful for:

- verifying provider configuration
- checking structured output behavior
- comparing throughput and output quality

They should remain operational references, not become core unit-test fixtures.

## Trade-Offs

This design intentionally keeps abstraction shallow.

Pros:

- minimal migration risk
- real provider support where config already implies it
- shared metrics/raw response handling
- easy incremental rollout

Cons:

- some provider-specific naming will still exist at the script config layer until a later cleanup
- the transport will still need some branching because the providers are not truly identical
- capability handling is intentionally small and will not cover every future backend difference

## Success Criteria

The design is successful when:

- `build_photo_pre_model_annotations.py` can switch between declared providers through one execution path
- `probe_vlm_photo_boundaries.py` can switch between declared providers through one execution path
- structured output still works where already supported
- `.vocatio` configuration remains stable for current users
- benchmark scripts remain usable for live sanity checks
- no downstream artifact format changes are required

## Recommended Next Step

Write an implementation plan that:

- introduces `vlm_transport.py`
- defines transport contract tests first
- migrates `build_photo_pre_model_annotations.py`
- migrates `probe_vlm_photo_boundaries.py`
- defers broader config cleanup until after runtime parity is confirmed
