from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
    messages: list[dict[str, Any]]
    image_paths: list[Path]
    timeout_seconds: float
    response_format: dict[str, Any] | None = None
    temperature: float | None = None
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    reasoning_level: str | None = None
    keep_alive: str | None = None


@dataclass(frozen=True)
class VlmResponse:
    provider: str
    model: str
    text: str
    json_payload: dict[str, Any] | None
    finish_reason: str | None
    metrics: dict[str, Any]
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class VlmTransportError(Exception):
    category: str
    message: str

    def __str__(self) -> str:
        return self.message


VLM_CAPABILITIES_BY_PROVIDER = {
    "ollama": VlmCapabilities(
        supports_json_schema=True,
        supports_json_object=True,
        supports_reasoning_control=True,
        supports_keep_alive=True,
        supports_multi_image=True,
    ),
    "llamacpp": VlmCapabilities(
        supports_json_schema=True,
        supports_json_object=True,
        supports_reasoning_control=False,
        supports_keep_alive=False,
        supports_multi_image=True,
    ),
    "vllm": VlmCapabilities(
        supports_json_schema=True,
        supports_json_object=True,
        supports_reasoning_control=False,
        supports_keep_alive=False,
        supports_multi_image=True,
    ),
}


def get_vlm_capabilities(provider: str) -> VlmCapabilities:
    capabilities = VLM_CAPABILITIES_BY_PROVIDER.get(provider)
    if capabilities is not None:
        return capabilities
    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {provider}")


def validate_vlm_request(request: VlmRequest) -> None:
    capabilities = get_vlm_capabilities(request.provider)
    if not str(request.base_url or "").strip():
        raise VlmTransportError("invalid_request", "VLM request base_url must be non-empty")
    if not str(request.model or "").strip():
        raise VlmTransportError("invalid_request", "VLM request model must be non-empty")
    if not isinstance(request.messages, list) or not request.messages:
        raise VlmTransportError("invalid_request", "VLM request messages must be a non-empty list")
    for message in request.messages:
        if not isinstance(message, dict):
            raise VlmTransportError("invalid_request", "VLM request messages must contain dict items")
        if not str(message.get("role", "") or "").strip():
            raise VlmTransportError("invalid_request", "VLM request message role must be non-empty")
        if not str(message.get("content", "") or "").strip():
            raise VlmTransportError("invalid_request", "VLM request message content must be non-empty")
    if not isinstance(request.image_paths, list):
        raise VlmTransportError("invalid_request", "VLM request image_paths must be a list")
    for image_path in request.image_paths:
        if not isinstance(image_path, Path):
            raise VlmTransportError("invalid_request", "VLM request image_paths must contain Path items")
        if not image_path.exists():
            raise VlmTransportError("invalid_request", f"VLM request image path does not exist: {image_path}")
    if not math.isfinite(request.timeout_seconds) or request.timeout_seconds <= 0:
        raise VlmTransportError("invalid_request", "VLM request timeout_seconds must be finite and greater than zero")
    if request.temperature is not None and not math.isfinite(request.temperature):
        raise VlmTransportError("invalid_request", "VLM request temperature must be finite")
    if request.context_tokens is not None and request.context_tokens <= 0:
        raise VlmTransportError("invalid_request", "VLM request context_tokens must be greater than zero")
    if request.max_output_tokens is not None and request.max_output_tokens <= 0:
        raise VlmTransportError("invalid_request", "VLM request max_output_tokens must be greater than zero")
    if request.response_format is not None:
        if not isinstance(request.response_format, dict):
            raise VlmTransportError("invalid_request", "VLM request response_format must be a dict")
        if not str(request.response_format.get("type", "") or "").strip():
            raise VlmTransportError("invalid_request", "VLM request response_format type must be non-empty")
    if request.keep_alive is not None:
        if not str(request.keep_alive or "").strip():
            raise VlmTransportError("invalid_request", "VLM request keep_alive must be non-empty when set")
        if not capabilities.supports_keep_alive:
            raise VlmTransportError(
                "unsupported_configuration",
                f"Provider does not support keep_alive: {request.provider}",
            )
    if request.reasoning_level is not None:
        if not str(request.reasoning_level or "").strip():
            raise VlmTransportError("invalid_request", "VLM request reasoning_level must be non-empty when set")
        if not capabilities.supports_reasoning_control:
            raise VlmTransportError(
                "unsupported_configuration",
                f"Provider does not support reasoning_level: {request.provider}",
            )


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _encode_image_data_url(image_path: Path) -> str:
    return f"data:image/jpeg;base64,{_encode_image_base64(image_path)}"


def _build_ollama_messages(request: VlmRequest) -> list[dict[str, Any]]:
    messages = [dict(message) for message in request.messages]
    if not request.image_paths:
        return messages
    last_message = dict(messages[-1])
    last_message["images"] = [_encode_image_base64(image_path) for image_path in request.image_paths]
    messages[-1] = last_message
    return messages


def _build_openai_messages(request: VlmRequest) -> list[dict[str, Any]]:
    messages = [dict(message) for message in request.messages]
    if not request.image_paths:
        return messages
    last_message = dict(messages[-1])
    content_items: list[dict[str, Any]] = [
        {"type": "text", "text": str(last_message.get("content", "") or "")}
    ]
    content_items.extend(
        {"type": "image_url", "image_url": {"url": _encode_image_data_url(image_path)}}
        for image_path in request.image_paths
    )
    last_message["content"] = content_items
    messages[-1] = last_message
    return messages


def _normalize_reasoning_level(reasoning_level: str) -> tuple[str, bool]:
    if reasoning_level == "false":
        return "none", False
    return reasoning_level, True


def build_provider_request_payload(request: VlmRequest) -> dict[str, Any]:
    if request.provider == "ollama":
        options: dict[str, Any] = {}
        if request.context_tokens is not None:
            options["num_ctx"] = request.context_tokens
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _build_ollama_messages(request),
            "stream": False,
        }
        if options:
            payload["options"] = options
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        if request.reasoning_level is not None:
            reasoning_effort, think_enabled = _normalize_reasoning_level(request.reasoning_level)
            payload["reasoning_effort"] = reasoning_effort
            payload["reasoning"] = {"effort": reasoning_effort}
            payload["think"] = think_enabled
        if request.response_format is not None:
            payload["format"] = request.response_format
        return payload
    if request.provider in {"llamacpp", "vllm"}:
        payload = {
            "model": request.model,
            "messages": _build_openai_messages(request),
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        return payload
    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {request.provider}")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_response_text(payload: dict[str, Any]) -> str:
    message = payload.get("message", {})
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return ""


def _extract_openai_choice(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices", [])
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    text = str(message.get("content", "") or "") if isinstance(message, dict) else ""
    finish_reason = None
    if isinstance(first_choice, dict):
        finish_reason = str(first_choice.get("finish_reason", "") or "") or None
    return text, finish_reason


def _extract_normalized_openai_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage", {})
    metrics: dict[str, Any] = {}
    if isinstance(usage, dict):
        if "prompt_tokens" in usage:
            metrics["prompt_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            metrics["completion_tokens"] = usage["completion_tokens"]
        if "total_tokens" in usage:
            metrics["total_tokens"] = usage["total_tokens"]
    return metrics


def normalize_provider_response(provider: str, model: str, payload: dict[str, Any]) -> VlmResponse:
    if provider == "ollama":
        text = _extract_response_text(payload)
        metrics: dict[str, Any] = {}
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
            json_payload=_parse_json_object(text),
            finish_reason=str(payload.get("done_reason", "") or "") or None,
            metrics=metrics,
            raw_response=payload,
        )
    if provider == "llamacpp":
        text, finish_reason = _extract_openai_choice(payload)
        return VlmResponse(
            provider=provider,
            model=model,
            text=text,
            json_payload=_parse_json_object(text),
            finish_reason=finish_reason,
            metrics=_extract_normalized_openai_metrics(payload),
            raw_response=payload,
        )
    if provider == "vllm":
        text, finish_reason = _extract_openai_choice(payload)
        return VlmResponse(
            provider=provider,
            model=model,
            text=text,
            json_payload=_parse_json_object(text),
            finish_reason=finish_reason,
            metrics=_extract_normalized_openai_metrics(payload),
            raw_response=payload,
        )
    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {provider}")


def _validate_provider_success_payload(provider: str, payload: dict[str, Any]) -> None:
    if provider == "ollama":
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise VlmTransportError(
                "invalid_response",
                f"Invalid response payload from {provider}: missing message.content",
            )
        return
    if provider in {"llamacpp", "vllm"}:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VlmTransportError(
                "invalid_response",
                f"Invalid response payload from {provider}: missing choices[0].message.content",
            )
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise VlmTransportError(
                "invalid_response",
                f"Invalid response payload from {provider}: missing choices[0].message.content",
            )
        return
    raise VlmTransportError("unsupported_configuration", f"Unsupported VLM provider: {provider}")


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise VlmTransportError("invalid_response", "VLM transport response must be a JSON object")
    return response_payload


def run_vlm_request(request: VlmRequest) -> VlmResponse:
    validate_vlm_request(request)
    payload = build_provider_request_payload(request)
    path = "/api/chat" if request.provider == "ollama" else "/v1/chat/completions"
    try:
        response_payload = post_json(request.base_url, path, payload, request.timeout_seconds)
    except VlmTransportError:
        raise
    except HTTPError as exc:
        raise VlmTransportError("http", f"HTTP error from {request.provider}: {exc}") from exc
    except TimeoutError as exc:
        raise VlmTransportError("timeout", f"Timeout from {request.provider}: {exc}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            raise VlmTransportError("timeout", f"Timeout from {request.provider}: {exc}") from exc
        raise VlmTransportError("connection", f"Connection error from {request.provider}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VlmTransportError("invalid_response", f"Invalid JSON response from {request.provider}: {exc}") from exc
    _validate_provider_success_payload(request.provider, response_payload)
    return normalize_provider_response(request.provider, request.model, response_payload)
