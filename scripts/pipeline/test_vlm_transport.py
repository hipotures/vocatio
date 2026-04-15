#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
