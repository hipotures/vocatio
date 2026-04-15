from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
