#!/usr/bin/env python3

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.lib import manual_vlm_models


class ManualVlmModelsTest(unittest.TestCase):
    def write_yaml(self, text: str) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="manual-vlm-models-"))
        path = temp_dir / "manual_vlm_models.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def write_models(self, models: list[dict[str, object]]) -> Path:
        return self.write_yaml(
            yaml.safe_dump(
                {"models": models},
                sort_keys=False,
            )
        )

    def build_model(self) -> dict[str, object]:
        return {
            "VLM_NAME": "Preset A",
            "VLM_PROVIDER": "ollama",
            "VLM_BASE_URL": "http://127.0.0.1:11434",
            "VLM_MODEL": "qwen",
            "VLM_CONTEXT_TOKENS": 4096,
            "VLM_MAX_OUTPUT_TOKENS": 512,
            "VLM_KEEP_ALIVE": "30m",
            "VLM_TIMEOUT_SECONDS": 180,
            "VLM_TEMPERATURE": 0.0,
            "VLM_REASONING_LEVEL": "low",
            "VLM_RESPONSE_SCHEMA_MODE": "on",
            "VLM_JSON_VALIDATION_MODE": "strict",
        }

    def test_checked_in_sample_config_loads(self) -> None:
        path = REPO_ROOT / "conf" / "manual_vlm_models.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        loaded = manual_vlm_models.load_manual_vlm_models(path)
        self.assertEqual(len(loaded.models), len(raw["models"]))
        self.assertTrue(loaded.md5_hex)

    def test_load_models_parses_complete_entries(self) -> None:
        path = self.write_yaml(
            """
models:
  - VLM_NAME: "Preset A"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "qwen"
    VLM_CONTEXT_TOKENS: 4096
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "30m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "low"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
"""
        )
        loaded = manual_vlm_models.load_manual_vlm_models(path)
        self.assertEqual(len(loaded.models), 1)
        self.assertEqual(loaded.models[0]["VLM_NAME"], "Preset A")
        self.assertTrue(loaded.md5_hex)

    def test_load_models_accepts_optional_description(self) -> None:
        path = self.write_yaml(
            """
models:
  - VLM_NAME: "Preset A"
    VLM_DESCRIPTION: "Long description"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "qwen"
    VLM_CONTEXT_TOKENS: 4096
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "30m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "low"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
"""
        )

        loaded = manual_vlm_models.load_manual_vlm_models(path)

        self.assertEqual(loaded.models[0]["VLM_DESCRIPTION"], "Long description")

    def test_load_models_allows_missing_description(self) -> None:
        path = self.write_models([self.build_model()])

        loaded = manual_vlm_models.load_manual_vlm_models(path)

        self.assertNotIn("VLM_DESCRIPTION", loaded.models[0])

    def test_load_models_treats_blank_description_as_missing(self) -> None:
        model = self.build_model()
        model["VLM_DESCRIPTION"] = "   "
        path = self.write_models([model])

        loaded = manual_vlm_models.load_manual_vlm_models(path)

        self.assertNotIn("VLM_DESCRIPTION", loaded.models[0])

    def test_load_models_rejects_duplicate_names(self) -> None:
        path = self.write_yaml(
            """
models:
  - VLM_NAME: "Preset A"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "a"
    VLM_CONTEXT_TOKENS: 4096
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "30m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "low"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
  - VLM_NAME: "Preset A"
    VLM_PROVIDER: "vllm"
    VLM_BASE_URL: "http://127.0.0.1:8000"
    VLM_MODEL: "b"
    VLM_CONTEXT_TOKENS: 4096
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.2
    VLM_REASONING_LEVEL: "inherit"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
"""
        )
        with self.assertRaisesRegex(ValueError, 'duplicate VLM_NAME "Preset A"'):
            manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_missing_required_fields(self) -> None:
        for field_name in manual_vlm_models.MODEL_FIELDS:
            with self.subTest(field_name=field_name):
                model = self.build_model()
                model.pop(field_name)
                path = self.write_models([model])
                with self.assertRaisesRegex(
                    ValueError,
                    f"missing {field_name} in preset at index 0",
                ):
                    manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_unsupported_provider(self) -> None:
        model = self.build_model()
        model["VLM_PROVIDER"] = "unsupported-provider"
        path = self.write_models([model])

        with self.assertRaisesRegex(
            ValueError,
            'unsupported VLM_PROVIDER "unsupported-provider" in preset "Preset A"',
        ):
            manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_blank_required_strings(self) -> None:
        cases = (
            ("VLM_BASE_URL", ""),
            ("VLM_BASE_URL", "   "),
            ("VLM_MODEL", ""),
            ("VLM_MODEL", "   "),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                model = self.build_model()
                model[field_name] = value
                path = self.write_models([model])
                with self.assertRaisesRegex(
                    ValueError,
                    f'{field_name} must be a non-empty string in preset "Preset A"',
                ):
                    manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_non_positive_numeric_limits(self) -> None:
        cases = (
            ("VLM_CONTEXT_TOKENS", 0),
            ("VLM_CONTEXT_TOKENS", -1),
            ("VLM_MAX_OUTPUT_TOKENS", 0),
            ("VLM_MAX_OUTPUT_TOKENS", -1),
            ("VLM_TIMEOUT_SECONDS", 0),
            ("VLM_TIMEOUT_SECONDS", -0.5),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                model = self.build_model()
                model[field_name] = value
                path = self.write_models([model])
                with self.assertRaisesRegex(
                    ValueError,
                    f'{field_name} must be > 0 in preset "Preset A"',
                ):
                    manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_malformed_numeric_text(self) -> None:
        cases = (
            ("VLM_CONTEXT_TOKENS", "nope"),
            ("VLM_MAX_OUTPUT_TOKENS", "bad"),
            ("VLM_TIMEOUT_SECONDS", "oops"),
        )
        expected_types = {
            "VLM_CONTEXT_TOKENS": "integer",
            "VLM_MAX_OUTPUT_TOKENS": "integer",
            "VLM_TIMEOUT_SECONDS": "number",
        }
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                model = self.build_model()
                model[field_name] = value
                path = self.write_models([model])
                with self.assertRaisesRegex(
                    ValueError,
                    f'{field_name} must be a positive {expected_types[field_name]} in preset "Preset A"',
                ):
                    manual_vlm_models.load_manual_vlm_models(path)

    def test_load_models_rejects_non_integer_yaml_values_for_int_fields(self) -> None:
        cases = (
            ("VLM_CONTEXT_TOKENS", True),
            ("VLM_CONTEXT_TOKENS", False),
            ("VLM_CONTEXT_TOKENS", 1.0),
            ("VLM_MAX_OUTPUT_TOKENS", 1.5),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                model = self.build_model()
                model[field_name] = value
                path = self.write_models([model])
                with self.assertRaisesRegex(
                    ValueError,
                    f'{field_name} must be a positive integer in preset "Preset A"',
                ):
                    manual_vlm_models.load_manual_vlm_models(path)

    def test_compute_models_md5_changes_with_file_content(self) -> None:
        path = self.write_yaml("models: []\n")
        before = manual_vlm_models.compute_manual_vlm_models_md5(path)
        path.write_text("models:\n  - VLM_NAME: \"Changed\"\n", encoding="utf-8")
        after = manual_vlm_models.compute_manual_vlm_models_md5(path)
        self.assertNotEqual(before, after)

    def test_load_models_md5_matches_exact_file_bytes(self) -> None:
        path = self.write_models([self.build_model()])
        expected = hashlib.md5(path.read_bytes()).hexdigest()

        loaded = manual_vlm_models.load_manual_vlm_models(path)

        self.assertEqual(loaded.md5_hex, expected)


if __name__ == "__main__":
    unittest.main()
