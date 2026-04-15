# VLM Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one narrow multimodal transport layer that lets `build_photo_pre_model_annotations.py` and `probe_vlm_photo_boundaries.py` execute against `ollama`, `llamacpp`, and `vllm` through one neutral request/response API.

**Architecture:** Introduce `scripts/pipeline/lib/vlm_transport.py` with a small typed contract for request, response, capabilities, and transport errors. Migrate the simpler pre-model script first, then migrate the boundary probe script, while keeping prompt construction, schema logic, CSV artifacts, and resume behavior in the calling scripts.

**Tech Stack:** Python 3, `urllib.request`, `json`, `dataclasses`, `pathlib`, `unittest`, existing `.vocatio` config loading and pipeline test scripts.

---

## File Structure

### New files

- `scripts/pipeline/lib/vlm_transport.py`
  - Neutral transport contract and provider adapters.
- `scripts/pipeline/test_vlm_transport.py`
  - Contract tests for request mapping, response normalization, capabilities, and error handling.

### Modified files

- `scripts/pipeline/build_photo_pre_model_annotations.py`
  - Replace direct llama.cpp request path with transport calls.
- `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - Replace direct Ollama request path with transport calls and neutral request assembly.
- `scripts/pipeline/test_build_photo_pre_model_annotations.py`
  - Preserve regression coverage while asserting provider-neutral transport integration.
- `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
  - Preserve regression coverage while asserting provider-neutral transport integration.
- `README.md`
  - Document real provider support for pre-model and boundary probe after migration.

### Existing files to inspect while implementing

- `scripts/pipeline/benchmark_caption_llamacpp_qwen4b.py`
- `scripts/pipeline/benchmark_schema_llamacpp_qwen4b.py`
- `scripts/pipeline/lib/workspace_dir.py`
- `scripts/pipeline/lib/review_index_loader.py`

## Task 1: Define the transport contract and failing tests

**Files:**
- Create: `scripts/pipeline/lib/vlm_transport.py`
- Create: `scripts/pipeline/test_vlm_transport.py`

- [ ] **Step 1: Write failing contract tests for capabilities and request validation**

```python
import unittest

from scripts.pipeline.lib import vlm_transport


class VlmTransportContractTests(unittest.TestCase):
    def test_get_vlm_capabilities_returns_expected_flags_for_ollama(self):
        capabilities = vlm_transport.get_vlm_capabilities("ollama")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertTrue(capabilities.supports_reasoning_control)
        self.assertTrue(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_get_vlm_capabilities_returns_expected_flags_for_llamacpp(self):
        capabilities = vlm_transport.get_vlm_capabilities("llamacpp")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertFalse(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_get_vlm_capabilities_returns_expected_flags_for_vllm(self):
        capabilities = vlm_transport.get_vlm_capabilities("vllm")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertFalse(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_validate_request_rejects_unknown_provider(self):
        request = vlm_transport.VlmRequest(
            provider="bad-provider",
            base_url="http://127.0.0.1:11434",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=30.0,
        )

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: FAIL with import or attribute errors for missing `VlmRequest`, `get_vlm_capabilities`, or `validate_vlm_request`.

- [ ] **Step 3: Add the minimal transport contract implementation**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VlmCapabilities:
    supports_json_schema: bool
    supports_json_object: bool
    supports_reasoning_control: bool
    supports_keep_alive: bool
    supports_multi_image: bool


@dataclass(frozen=True)
class VlmRequest:
    provider: str
    base_url: str
    model: str
    messages: List[Dict[str, Any]]
    image_paths: List[Path]
    timeout_seconds: float
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    reasoning_level: Optional[str] = None
    keep_alive: Optional[str] = None


@dataclass(frozen=True)
class VlmTransportError(Exception):
    category: str
    message: str

    def __str__(self) -> str:
        return self.message


def get_vlm_capabilities(provider: str) -> VlmCapabilities:
    if provider == "ollama":
        return VlmCapabilities(True, True, True, True, True)
    if provider == "llamacpp":
        return VlmCapabilities(True, True, False, False, True)
    if provider == "vllm":
        return VlmCapabilities(True, True, False, False, True)
    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {provider}")


def validate_vlm_request(request: VlmRequest) -> None:
    get_vlm_capabilities(request.provider)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: PASS for the new contract tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/vlm_transport.py scripts/pipeline/test_vlm_transport.py
git commit -m "Add VLM transport contract"
```

## Task 2: Add provider payload builders and response normalization

**Files:**
- Modify: `scripts/pipeline/lib/vlm_transport.py`
- Modify: `scripts/pipeline/test_vlm_transport.py`

- [ ] **Step 1: Write failing tests for provider payload mapping and normalized responses**

```python
from pathlib import Path
from unittest import mock


class VlmTransportPayloadTests(unittest.TestCase):
    def test_build_ollama_request_payload_maps_neutral_fields(self):
        request = vlm_transport.VlmRequest(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": "Describe image."}],
            image_paths=[Path("/tmp/example.jpg")],
            timeout_seconds=60.0,
            temperature=0.0,
            context_tokens=16384,
            max_output_tokens=256,
            reasoning_level="false",
            keep_alive="15m",
            response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        )

        payload = vlm_transport.build_provider_request_payload(request)

        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertEqual(payload["keep_alive"], "15m")
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertIn("format", payload)

    def test_build_llamacpp_request_payload_maps_neutral_fields(self):
        request = vlm_transport.VlmRequest(
            provider="llamacpp",
            base_url="http://127.0.0.1:8002",
            model="unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
            messages=[{"role": "user", "content": "Describe image."}],
            image_paths=[Path("/tmp/example.jpg")],
            timeout_seconds=60.0,
            temperature=0.0,
            max_output_tokens=512,
            response_format={"type": "json_object"},
        )

        payload = vlm_transport.build_provider_request_payload(request)

        self.assertEqual(payload["model"], "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["response_format"]["type"], "json_object")

    def test_normalize_ollama_response_extracts_text_and_metrics(self):
        payload = {
            "message": {"content": "{\"answer\":\"ok\"}"},
            "done_reason": "stop",
            "prompt_eval_count": 123,
            "eval_count": 45,
            "total_duration": 1_500_000_000,
            "eval_duration": 300_000_000,
        }

        response = vlm_transport.normalize_provider_response("ollama", "demo", payload)

        self.assertEqual(response.text, "{\"answer\":\"ok\"}")
        self.assertEqual(response.json_payload, {"answer": "ok"})
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.metrics["prompt_tokens"], 123)
        self.assertEqual(response.metrics["completion_tokens"], 45)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: FAIL with missing `build_provider_request_payload` or `normalize_provider_response`.

- [ ] **Step 3: Implement provider mapping and response normalization**

```python
import base64
import json


@dataclass(frozen=True)
class VlmResponse:
    provider: str
    model: str
    text: str
    json_payload: Optional[Dict[str, Any]]
    finish_reason: Optional[str]
    metrics: Dict[str, Any]
    raw_response: Dict[str, Any]


def build_provider_request_payload(request: VlmRequest) -> Dict[str, Any]:
    if request.provider == "ollama":
        options: Dict[str, Any] = {}
        if request.context_tokens is not None:
            options["num_ctx"] = request.context_tokens
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if request.keep_alive:
            payload["keep_alive"] = request.keep_alive
        if request.response_format is not None:
            payload["format"] = request.response_format
        return payload

    if request.provider in {"llamacpp", "vllm"}:
        payload = {
            "model": request.model,
            "messages": request.messages,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        return payload

    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {request.provider}")


def normalize_provider_response(provider: str, model: str, payload: Dict[str, Any]) -> VlmResponse:
    if provider == "ollama":
        text = str(payload.get("message", {}).get("content", "") or "")
        json_payload = None
        if text:
            try:
                json_payload = json.loads(text)
            except json.JSONDecodeError:
                json_payload = None
        metrics: Dict[str, Any] = {}
        if "prompt_eval_count" in payload:
            metrics["prompt_tokens"] = payload["prompt_eval_count"]
        if "eval_count" in payload:
            metrics["completion_tokens"] = payload["eval_count"]
        if "total_duration" in payload:
            metrics["total_duration_seconds"] = payload["total_duration"] / 1_000_000_000
        if "eval_duration" in payload:
            metrics["eval_duration_seconds"] = payload["eval_duration"] / 1_000_000_000
        return VlmResponse(
            provider=provider,
            model=model,
            text=text,
            json_payload=json_payload,
            finish_reason=str(payload.get("done_reason", "") or "") or None,
            metrics=metrics,
            raw_response=payload,
        )

    message = payload.get("choices", [{}])[0].get("message", {})
    text = str(message.get("content", "") or "")
    json_payload = None
    if text:
        try:
            json_payload = json.loads(text)
        except json.JSONDecodeError:
            json_payload = None
    usage = payload.get("usage", {}) or {}
    metrics = {}
    if "prompt_tokens" in usage:
        metrics["prompt_tokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        metrics["completion_tokens"] = usage["completion_tokens"]
    return VlmResponse(
        provider=provider,
        model=model,
        text=text,
        json_payload=json_payload,
        finish_reason=str(payload.get("choices", [{}])[0].get("finish_reason", "") or "") or None,
        metrics=metrics,
        raw_response=payload,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: PASS for request mapping and normalization tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/vlm_transport.py scripts/pipeline/test_vlm_transport.py
git commit -m "Add VLM provider payload mapping"
```

## Task 3: Add HTTP execution and transport error handling

**Files:**
- Modify: `scripts/pipeline/lib/vlm_transport.py`
- Modify: `scripts/pipeline/test_vlm_transport.py`

- [ ] **Step 1: Write failing tests for HTTP path selection and error categories**

```python
from urllib.error import HTTPError, URLError


class VlmTransportExecutionTests(unittest.TestCase):
    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_uses_ollama_chat_endpoint(self, post_json_mock):
        post_json_mock.return_value = {"message": {"content": "ok"}, "done_reason": "stop"}
        request = vlm_transport.VlmRequest(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        response = vlm_transport.run_vlm_request(request)

        self.assertEqual(response.text, "ok")
        post_json_mock.assert_called_once()
        self.assertEqual(post_json_mock.call_args.args[1], "/api/chat")

    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_uses_openai_chat_endpoint_for_llamacpp(self, post_json_mock):
        post_json_mock.return_value = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        request = vlm_transport.VlmRequest(
            provider="llamacpp",
            base_url="http://127.0.0.1:8002",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        vlm_transport.run_vlm_request(request)

        self.assertEqual(post_json_mock.call_args.args[1], "/v1/chat/completions")

    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_wraps_url_errors(self, post_json_mock):
        post_json_mock.side_effect = URLError("connection refused")
        request = vlm_transport.VlmRequest(
            provider="vllm",
            base_url="http://127.0.0.1:8000",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.run_vlm_request(request)

        self.assertEqual(error.exception.category, "connection")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: FAIL with missing `run_vlm_request` or `post_json`.

- [ ] **Step 3: Implement HTTP execution and error wrapping**

```python
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def run_vlm_request(request: VlmRequest) -> VlmResponse:
    validate_vlm_request(request)
    payload = build_provider_request_payload(request)
    path = "/api/chat" if request.provider == "ollama" else "/v1/chat/completions"
    try:
        response_payload = post_json(request.base_url, path, payload, request.timeout_seconds)
    except HTTPError as exc:
        raise VlmTransportError("http", f"HTTP error from {request.provider}: {exc}") from exc
    except TimeoutError as exc:
        raise VlmTransportError("timeout", f"Timeout from {request.provider}: {exc}") from exc
    except URLError as exc:
        raise VlmTransportError("connection", f"Connection error from {request.provider}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VlmTransportError("invalid_response", f"Invalid JSON response from {request.provider}: {exc}") from exc
    return normalize_provider_response(request.provider, request.model, response_payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_vlm_transport.py`

Expected: PASS for the execution and error tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/vlm_transport.py scripts/pipeline/test_vlm_transport.py
git commit -m "Add VLM transport execution path"
```

## Task 4: Migrate pre-model annotations to the transport

**Files:**
- Modify: `scripts/pipeline/build_photo_pre_model_annotations.py`
- Modify: `scripts/pipeline/test_build_photo_pre_model_annotations.py`

- [ ] **Step 1: Write failing regression tests for provider-neutral pre-model execution**

```python
from unittest import mock


class BuildPhotoPreModelTransportTests(unittest.TestCase):
    @mock.patch("scripts.pipeline.build_photo_pre_model_annotations.run_vlm_request")
    def test_process_image_uses_transport_request_with_llamacpp_provider(self, run_vlm_request_mock):
        run_vlm_request_mock.return_value = mock.Mock(
            text='{"people_count":"1"}',
            json_payload={"people_count": "1"},
            raw_response={"ok": True},
            metrics={},
        )

        args = build_pre_model.parse_args([
            "/tmp/day",
            "--provider",
            "llamacpp",
            "--base-url",
            "http://127.0.0.1:8002",
            "--model-name",
            "demo-model",
        ])

        request = build_pre_model.build_vlm_request(
            args=args,
            prompt_text="Describe costume.",
            image_path=Path("/tmp/frame.jpg"),
        )

        self.assertEqual(request.provider, "llamacpp")
        self.assertEqual(request.base_url, "http://127.0.0.1:8002")
        self.assertEqual(request.model, "demo-model")
        self.assertEqual(request.max_output_tokens, args.max_output_tokens)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_build_photo_pre_model_annotations.py`

Expected: FAIL with missing `build_vlm_request` or missing transport integration.

- [ ] **Step 3: Implement transport-backed request assembly in the pre-model script**

```python
from scripts.pipeline.lib.vlm_transport import VlmRequest, get_vlm_capabilities, run_vlm_request


def build_vlm_request(args: argparse.Namespace, prompt_text: str, image_path: Path) -> VlmRequest:
    capabilities = get_vlm_capabilities(args.provider)
    response_format = build_response_format() if capabilities.supports_json_schema else None
    return VlmRequest(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model_name,
        messages=[{"role": "user", "content": prompt_text}],
        image_paths=[image_path],
        timeout_seconds=args.timeout_seconds,
        response_format=response_format,
        temperature=args.temperature,
        context_tokens=args.context_tokens,
        max_output_tokens=args.max_output_tokens,
        reasoning_level=args.reasoning_level,
        keep_alive=args.keep_alive,
    )


def request_annotation(args: argparse.Namespace, prompt_text: str, image_path: Path) -> Dict[str, Any]:
    response = run_vlm_request(build_vlm_request(args, prompt_text, image_path))
    if response.json_payload is not None:
        return response.json_payload
    return json.loads(response.text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_build_photo_pre_model_annotations.py`

Expected: PASS with the new transport-backed request path while preserving existing annotation behavior.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/test_build_photo_pre_model_annotations.py scripts/pipeline/lib/vlm_transport.py
git commit -m "Migrate pre-model annotations to VLM transport"
```

## Task 5: Migrate boundary probing to the transport

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write failing regression tests for provider-neutral probe execution**

```python
class ProbeVlmTransportTests(unittest.TestCase):
    def test_build_vlm_request_uses_neutral_fields(self):
        args = probe.parse_args([
            "/tmp/day",
            "--provider",
            "ollama",
            "--model",
            "qwen3.5:9b",
            "--base-url",
            "http://127.0.0.1:11434",
            "--response-schema-mode",
            "on",
            "--ollama-num-ctx",
            "16384",
            "--ollama-num-predict",
            "256",
            "--temperature",
            "0",
        ])

        request = probe.build_vlm_request(
            args=args,
            prompt_text="Choose at most one boundary.",
            image_paths=[Path("/tmp/a.jpg"), Path("/tmp/b.jpg")],
        )

        self.assertEqual(request.provider, "ollama")
        self.assertEqual(request.context_tokens, 16384)
        self.assertEqual(request.max_output_tokens, 256)
        self.assertEqual(request.temperature, 0.0)
        self.assertIsNotNone(request.response_format)

    @mock.patch("scripts.pipeline.probe_vlm_photo_boundaries.run_vlm_request")
    def test_probe_batch_uses_transport_instead_of_direct_ollama_call(self, run_vlm_request_mock):
        run_vlm_request_mock.return_value = mock.Mock(
            text='{"boundary_after_frame":"frame_02","left_segment_type":"dance","right_segment_type":"dance","frame_notes":{},"primary_evidence":[],"summary":"ok"}',
            json_payload={
                "boundary_after_frame": "frame_02",
                "left_segment_type": "dance",
                "right_segment_type": "dance",
                "frame_notes": {},
                "primary_evidence": [],
                "summary": "ok",
            },
            raw_response={"provider": "mock"},
            metrics={"prompt_tokens": 10},
            finish_reason="stop",
        )

        result = probe.call_vlm_for_candidate_batch(
            args=mock.Mock(provider="ollama"),
            prompt_text="Choose boundary.",
            image_paths=[Path("/tmp/a.jpg"), Path("/tmp/b.jpg")],
        )

        self.assertEqual(result["response_status"], "ok")
        run_vlm_request_mock.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py`

Expected: FAIL with missing `build_vlm_request` or because direct Ollama call sites are still used.

- [ ] **Step 3: Implement transport-backed probe execution**

```python
from scripts.pipeline.lib.vlm_transport import VlmRequest, get_vlm_capabilities, run_vlm_request


def build_vlm_request(args: argparse.Namespace, prompt_text: str, image_paths: List[Path]) -> VlmRequest:
    capabilities = get_vlm_capabilities(args.provider)
    response_format = build_response_format() if args.response_schema_mode == "on" and capabilities.supports_json_schema else None
    return VlmRequest(
        provider=args.provider,
        base_url=args.ollama_base_url,
        model=args.model,
        messages=[{"role": "user", "content": prompt_text}],
        image_paths=image_paths,
        timeout_seconds=args.timeout_seconds,
        response_format=response_format,
        temperature=args.temperature,
        context_tokens=args.ollama_num_ctx,
        max_output_tokens=args.ollama_num_predict,
        reasoning_level=args.ollama_think,
        keep_alive=args.ollama_keep_alive,
    )


def call_vlm_for_candidate_batch(args: argparse.Namespace, prompt_text: str, image_paths: List[Path]) -> Dict[str, Any]:
    response = run_vlm_request(build_vlm_request(args, prompt_text, image_paths))
    raw_text = response.text
    parsed = parse_model_response(raw_text, args.json_validation_mode)
    parsed["raw_response"] = response.raw_response
    parsed["transport_metrics"] = response.metrics
    return parsed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py`

Expected: PASS with probe behavior preserved and no direct provider-specific execution path left in the main request flow.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/lib/vlm_transport.py
git commit -m "Migrate boundary probe to VLM transport"
```

## Task 6: Final cleanup, docs, and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `scripts/pipeline/build_photo_pre_model_annotations.py`
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Modify: `scripts/pipeline/test_vlm_transport.py`
- Modify: `scripts/pipeline/test_build_photo_pre_model_annotations.py`
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write failing documentation and parity checks**

```python
class VlmTransportParityTests(unittest.TestCase):
    def test_probe_vocatio_provider_is_not_restricted_to_ollama_only(self):
        args = probe.parse_args(["/tmp/day", "--provider", "vllm"])
        self.assertEqual(args.provider, "vllm")

    def test_pre_model_vocatio_provider_is_not_restricted_to_llamacpp_only(self):
        args = build_pre_model.parse_args(["/tmp/day", "--provider", "ollama"])
        self.assertEqual(args.provider, "ollama")
```

- [ ] **Step 2: Run tests to verify the remaining guard rails fail**

Run: `python3 scripts/pipeline/test_build_photo_pre_model_annotations.py && python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py`

Expected: FAIL where the scripts still reject providers too early or docs are stale.

- [ ] **Step 3: Remove obsolete single-provider guard rails and update docs**

```python
if args.provider not in {"ollama", "llamacpp", "vllm"}:
    raise SystemExit(f"Unsupported VLM provider: {args.provider}")
```

```markdown
`build_photo_pre_model_annotations.py` and `probe_vlm_photo_boundaries.py` now execute through a shared transport layer and can target multiple configured providers through `.vocatio` or CLI, subject to backend capability support.
```

- [ ] **Step 4: Run the full verification set**

Run:

```bash
python3 scripts/pipeline/test_vlm_transport.py
python3 scripts/pipeline/test_build_photo_pre_model_annotations.py
python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py
python3 -m py_compile scripts/pipeline/lib/vlm_transport.py scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_vlm_transport.py scripts/pipeline/test_build_photo_pre_model_annotations.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:

- all test scripts pass
- `py_compile` succeeds

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/pipeline/lib/vlm_transport.py scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_vlm_transport.py scripts/pipeline/test_build_photo_pre_model_annotations.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git commit -m "Document shared VLM transport support"
```

## Self-Review

### Spec coverage

- Neutral request/response contract: covered by Tasks 1-3.
- Provider mapping for `ollama`, `llamacpp`, `vllm`: covered by Tasks 2-3.
- Capability checks: covered by Tasks 1-2.
- Pre-model migration: covered by Task 4.
- Boundary probe migration: covered by Task 5.
- Error handling and normalized metrics: covered by Tasks 2-3.
- Regression coverage and benchmark preservation: covered by Tasks 4-6.

No spec gaps remain.

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” markers remain.
- Every code-changing step includes concrete code.
- Every verification step includes an exact command.

### Type consistency

- Transport contract names are consistent across tasks:
  - `VlmRequest`
  - `VlmResponse`
  - `VlmCapabilities`
  - `VlmTransportError`
  - `run_vlm_request`
  - `get_vlm_capabilities`
- Script integration steps consistently build transport requests first and then call `run_vlm_request`.
