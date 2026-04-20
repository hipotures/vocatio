#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import vlm_transport


class VlmTransportContractTests(unittest.TestCase):
    def make_request(self, **overrides: object) -> vlm_transport.VlmRequest:
        values = {
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "model": "demo",
            "messages": [{"role": "user", "content": "hi"}],
            "image_paths": [],
            "timeout_seconds": 30.0,
        }
        values.update(overrides)
        return vlm_transport.VlmRequest(**values)

    def test_vlm_response_smoke(self) -> None:
        response = vlm_transport.VlmResponse(
            provider="ollama",
            model="demo",
            text="ok",
            json_payload={"answer": "ok"},
            finish_reason="stop",
            metrics={"prompt_tokens": 1},
            raw_response={"message": {"content": "ok"}},
        )

        self.assertEqual(response.provider, "ollama")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.json_payload, {"answer": "ok"})

    def test_get_vlm_capabilities_returns_expected_flags_for_ollama(self) -> None:
        capabilities = vlm_transport.get_vlm_capabilities("ollama")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertTrue(capabilities.supports_reasoning_control)
        self.assertTrue(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_get_vlm_capabilities_returns_expected_flags_for_llamacpp(self) -> None:
        capabilities = vlm_transport.get_vlm_capabilities("llamacpp")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertFalse(capabilities.supports_reasoning_control)
        self.assertFalse(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_get_vlm_capabilities_returns_expected_flags_for_vllm(self) -> None:
        capabilities = vlm_transport.get_vlm_capabilities("vllm")

        self.assertTrue(capabilities.supports_json_schema)
        self.assertTrue(capabilities.supports_json_object)
        self.assertFalse(capabilities.supports_reasoning_control)
        self.assertFalse(capabilities.supports_keep_alive)
        self.assertTrue(capabilities.supports_multi_image)

    def test_validate_vlm_request_rejects_unknown_provider(self) -> None:
        request = self.make_request(provider="bad-provider")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")

    def test_validate_vlm_request_accepts_valid_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"jpg")
            request = self.make_request(
                provider="ollama",
                keep_alive="15m",
                reasoning_level="low",
                image_paths=[image_path],
                temperature=0.0,
                context_tokens=4096,
                max_output_tokens=256,
                response_format={"type": "json_object"},
            )

            vlm_transport.validate_vlm_request(request)

    def test_validate_vlm_request_rejects_empty_base_url(self) -> None:
        request = self.make_request(base_url="   ")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_empty_model(self) -> None:
        request = self.make_request(model="")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_empty_messages(self) -> None:
        request = self.make_request(messages=[])

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_malformed_message(self) -> None:
        request = self.make_request(messages=[{"role": "user", "content": ""}])

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_non_path_image_path(self) -> None:
        request = self.make_request(image_paths=["/tmp/example.jpg"])

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_missing_image_path(self) -> None:
        request = self.make_request(image_paths=[Path('/tmp/does-not-exist.jpg')])

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_non_positive_timeout(self) -> None:
        request = self.make_request(timeout_seconds=0.0)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_nan_timeout(self) -> None:
        request = self.make_request(timeout_seconds=math.nan)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_inf_timeout(self) -> None:
        request = self.make_request(timeout_seconds=math.inf)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_non_finite_temperature(self) -> None:
        request = self.make_request(temperature=math.nan)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_non_positive_context_tokens(self) -> None:
        request = self.make_request(context_tokens=0)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_non_positive_max_output_tokens(self) -> None:
        request = self.make_request(max_output_tokens=-5)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_invalid_response_format(self) -> None:
        request = self.make_request(response_format={"bogus": True})

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_blank_keep_alive(self) -> None:
        request = self.make_request(keep_alive="   ")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_blank_reasoning_level(self) -> None:
        request = self.make_request(reasoning_level="  ")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_request")

    def test_validate_vlm_request_rejects_unsupported_keep_alive(self) -> None:
        request = self.make_request(provider="llamacpp", keep_alive="15m")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")

    def test_validate_vlm_request_rejects_unsupported_reasoning_level(self) -> None:
        request = self.make_request(provider="vllm", reasoning_level="low")

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")

    def test_validate_vlm_request_rejects_unsupported_context_tokens(self) -> None:
        request = self.make_request(provider="llamacpp", context_tokens=4096)

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.validate_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")


class VlmTransportPayloadTests(unittest.TestCase):
    def test_build_ollama_request_payload_maps_neutral_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"jpg")
            request = vlm_transport.VlmRequest(
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                model="qwen3.5:9b",
                messages=[{"role": "user", "content": "Describe image."}],
                image_paths=[image_path],
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
        self.assertEqual(payload["messages"][0]["content"], "Describe image.")
        self.assertEqual(len(payload["messages"][0]["images"]), 1)
        self.assertIn("format", payload)

    def test_build_ollama_request_payload_maps_json_object_to_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"jpg")
            request = vlm_transport.VlmRequest(
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                model="gemma4:e4b",
                messages=[{"role": "user", "content": "Describe image."}],
                image_paths=[image_path],
                timeout_seconds=60.0,
                response_format={"type": "json_object"},
            )

            payload = vlm_transport.build_provider_request_payload(request)

        self.assertEqual(payload["format"], "json")

    def test_build_llamacpp_request_payload_maps_neutral_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"jpg")
            request = vlm_transport.VlmRequest(
                provider="llamacpp",
                base_url="http://127.0.0.1:8002",
                model="unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
                messages=[{"role": "user", "content": "Describe image."}],
                image_paths=[image_path],
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
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "Describe image.")
        self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_build_llamacpp_request_payload_moves_json_schema_to_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"jpg")
            request = vlm_transport.VlmRequest(
                provider="llamacpp",
                base_url="http://127.0.0.1:8002",
                model="unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
                messages=[{"role": "user", "content": "Describe image."}],
                image_paths=[image_path],
                timeout_seconds=60.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
                },
            )

            payload = vlm_transport.build_provider_request_payload(request)

        self.assertNotIn("response_format", payload)
        self.assertEqual(
            payload["json_schema"],
            {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )

    def test_build_vllm_request_payload_maps_neutral_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"jpg")
            request = vlm_transport.VlmRequest(
                provider="vllm",
                base_url="http://127.0.0.1:8000",
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[{"role": "user", "content": "Describe image."}],
                image_paths=[image_path],
                timeout_seconds=60.0,
                temperature=0.1,
                max_output_tokens=300,
                response_format={"type": "json_object"},
            )

            payload = vlm_transport.build_provider_request_payload(request)

        self.assertEqual(payload["model"], "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(payload["max_tokens"], 300)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["response_format"]["type"], "json_object")
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "Describe image.")
        self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_normalize_ollama_response_extracts_text_json_and_metrics(self) -> None:
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
        self.assertEqual(response.metrics["total_duration_seconds"], 1.5)
        self.assertEqual(response.metrics["eval_duration_seconds"], 0.3)

    def test_normalize_llamacpp_response_extracts_text_json_and_metrics(self) -> None:
        payload = {
            "choices": [{"message": {"content": "{\"answer\":\"ok\"}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "timings": {"prompt_ms": 12.5, "predicted_ms": 22.0},
        }

        response = vlm_transport.normalize_provider_response("llamacpp", "demo", payload)

        self.assertEqual(response.text, "{\"answer\":\"ok\"}")
        self.assertEqual(response.json_payload, {"answer": "ok"})
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.metrics, {"prompt_tokens": 11, "completion_tokens": 7})

    def test_normalize_vllm_response_extracts_text_json_and_metrics(self) -> None:
        payload = {
            "choices": [{"message": {"content": "{\"answer\":\"fine\"}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 5},
            "timings": {"queue_ms": 4.0},
        }

        response = vlm_transport.normalize_provider_response("vllm", "demo", payload)

        self.assertEqual(response.text, "{\"answer\":\"fine\"}")
        self.assertEqual(response.json_payload, {"answer": "fine"})
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.metrics, {"prompt_tokens": 19, "completion_tokens": 5})


class VlmTransportExecutionTests(unittest.TestCase):
    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_uses_api_chat_for_ollama(self, post_json_mock: mock.Mock) -> None:
        post_json_mock.return_value = {
            "message": {"content": "ok"},
            "done_reason": "stop",
        }
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

    @mock.patch.object(vlm_transport, "fetch_json")
    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_uses_openai_chat_completions_for_llamacpp(
        self,
        post_json_mock: mock.Mock,
        fetch_json_mock: mock.Mock,
    ) -> None:
        fetch_json_mock.return_value = {
            "data": [{"id": "demo"}],
        }
        post_json_mock.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        request = vlm_transport.VlmRequest(
            provider="llamacpp",
            base_url="http://127.0.0.1:8002",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        vlm_transport.run_vlm_request(request)

        fetch_json_mock.assert_called_once_with("http://127.0.0.1:8002", "/v1/models", 10.0)
        self.assertEqual(post_json_mock.call_args.args[1], "/v1/chat/completions")

    @mock.patch.object(vlm_transport, "fetch_json")
    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_rejects_llamacpp_model_mismatch(
        self,
        post_json_mock: mock.Mock,
        fetch_json_mock: mock.Mock,
    ) -> None:
        fetch_json_mock.return_value = {
            "data": [{"id": "actual-model"}],
            "models": [{"model": "actual-model"}],
        }
        request = vlm_transport.VlmRequest(
            provider="llamacpp",
            base_url="http://127.0.0.1:8002",
            model="configured-model",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.run_vlm_request(request)

        self.assertEqual(error.exception.category, "unsupported_configuration")
        self.assertEqual(
            str(error.exception),
            'llama.cpp model mismatch: configured model "configured-model" but server advertises "actual-model"',
        )
        post_json_mock.assert_not_called()

    @mock.patch.object(vlm_transport, "fetch_json")
    def test_run_vlm_request_rejects_llamacpp_models_payload_without_ids(
        self,
        fetch_json_mock: mock.Mock,
    ) -> None:
        fetch_json_mock.return_value = {
            "data": [{}],
            "models": [{}],
        }
        request = vlm_transport.VlmRequest(
            provider="llamacpp",
            base_url="http://127.0.0.1:8002",
            model="configured-model",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.run_vlm_request(request)

        self.assertEqual(error.exception.category, "invalid_response")
        self.assertIn("missing advertised model ids at /v1/models", str(error.exception))

    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_wraps_url_errors_as_connection(
        self,
        post_json_mock: mock.Mock,
    ) -> None:
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

    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_wraps_http_errors_as_http(self, post_json_mock: mock.Mock) -> None:
        post_json_mock.side_effect = HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=500,
            msg="server error",
            hdrs=None,
            fp=None,
        )
        request = vlm_transport.VlmRequest(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            model="demo",
            messages=[{"role": "user", "content": "hi"}],
            image_paths=[],
            timeout_seconds=10.0,
        )

        with self.assertRaises(vlm_transport.VlmTransportError) as error:
            vlm_transport.run_vlm_request(request)

        self.assertEqual(error.exception.category, "http")

    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_wraps_timeouts_as_timeout(self, post_json_mock: mock.Mock) -> None:
        post_json_mock.side_effect = TimeoutError("timed out")
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

        self.assertEqual(error.exception.category, "timeout")

    @mock.patch.object(vlm_transport, "fetch_json")
    @mock.patch.object(vlm_transport, "post_json")
    def test_run_vlm_request_rejects_malformed_success_payloads_as_invalid_response(
        self,
        fetch_json_mock: mock.Mock,
        post_json_mock: mock.Mock,
    ) -> None:
        for provider, base_url, payload in (
            ("ollama", "http://127.0.0.1:11434", {"message": {}}),
            ("llamacpp", "http://127.0.0.1:8002", {"choices": [{"message": {}}]}),
        ):
            with self.subTest(provider=provider):
                post_json_mock.reset_mock()
                fetch_json_mock.reset_mock()
                post_json_mock.return_value = payload
                if provider == "llamacpp":
                    fetch_json_mock.return_value = {"data": [{"id": "demo"}]}
                request = vlm_transport.VlmRequest(
                    provider=provider,
                    base_url=base_url,
                    model="demo",
                    messages=[{"role": "user", "content": "hi"}],
                    image_paths=[],
                    timeout_seconds=10.0,
                )

                with self.assertRaises(vlm_transport.VlmTransportError) as error:
                    vlm_transport.run_vlm_request(request)

                self.assertEqual(error.exception.category, "invalid_response")


if __name__ == "__main__":
    unittest.main()
