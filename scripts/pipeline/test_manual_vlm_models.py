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
        path = temp_dir / "vlm_models.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def repo_example_path(self) -> Path:
        return REPO_ROOT / "conf" / "vlm_models.yaml.example"

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

    def build_premodel(self) -> dict[str, object]:
        return {
            "PREMODEL_NAME": "Pre Preset A",
            "PREMODEL_PROVIDER": "llamacpp",
            "PREMODEL_BASE_URL": "http://127.0.0.1:8002",
            "PREMODEL_MODEL": "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
            "PREMODEL_MAX_OUTPUT_TOKENS": 1024,
            "PREMODEL_TEMPERATURE": 0.0,
            "PREMODEL_TIMEOUT_SECONDS": 120,
        }

    def test_checked_in_sample_config_loads(self) -> None:
        path = self.repo_example_path()
        loaded = manual_vlm_models.load_manual_vlm_models(path)
        self.assertEqual(len(loaded.models), 4)
        self.assertEqual(
            [model["VLM_NAME"] for model in loaded.models if "VLM_NAME" in model],
            [
                "Ollama localhost qwen3.5:9b",
                "llama.cpp localhost gemma-4-E4B-it-GGUF:Q8_0",
                "vLLM localhost Qwen3.5-0.8B",
            ],
        )
        self.assertTrue(loaded.md5_hex)

    def test_checked_in_sample_config_includes_premodel_entry(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())
        self.assertEqual(
            [model["PREMODEL_NAME"] for model in loaded.models if "PREMODEL_NAME" in model],
            ["llama.cpp localhost qwen3.5-4b-pre"],
        )

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

    def test_load_models_preserves_key_order_with_description(self) -> None:
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

        self.assertEqual(
            list(loaded.models[0].keys()),
            [
                "VLM_NAME",
                "VLM_DESCRIPTION",
                "VLM_PROVIDER",
                "VLM_BASE_URL",
                "VLM_MODEL",
                "VLM_CONTEXT_TOKENS",
                "VLM_MAX_OUTPUT_TOKENS",
                "VLM_KEEP_ALIVE",
                "VLM_TIMEOUT_SECONDS",
                "VLM_TEMPERATURE",
                "VLM_REASONING_LEVEL",
                "VLM_RESPONSE_SCHEMA_MODE",
                "VLM_JSON_VALIDATION_MODE",
            ],
        )

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

    def test_load_models_preserves_key_order_when_blank_description_is_omitted(self) -> None:
        model = self.build_model()
        model["VLM_DESCRIPTION"] = "   "
        path = self.write_models([model])

        loaded = manual_vlm_models.load_manual_vlm_models(path)

        self.assertEqual(
            list(loaded.models[0].keys()),
            [
                "VLM_NAME",
                "VLM_PROVIDER",
                "VLM_BASE_URL",
                "VLM_MODEL",
                "VLM_CONTEXT_TOKENS",
                "VLM_MAX_OUTPUT_TOKENS",
                "VLM_KEEP_ALIVE",
                "VLM_TIMEOUT_SECONDS",
                "VLM_TEMPERATURE",
                "VLM_REASONING_LEVEL",
                "VLM_RESPONSE_SCHEMA_MODE",
                "VLM_JSON_VALIDATION_MODE",
            ],
        )

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
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:8000"
    VLM_MODEL: "b"
    VLM_CONTEXT_TOKENS: 4096
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "15m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.2
    VLM_REASONING_LEVEL: "low"
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

    def test_load_models_rejects_missing_required_premodel_fields(self) -> None:
        for field_name in manual_vlm_models.PREMODEL_MODEL_FIELDS:
            with self.subTest(field_name=field_name):
                model = self.build_premodel()
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

    def test_load_models_rejects_non_ollama_only_transport_fields(self) -> None:
        cases = (
            ("VLM_CONTEXT_TOKENS", 4096, "Provider does not support context_tokens: llamacpp"),
            ("VLM_KEEP_ALIVE", "15m", "Provider does not support keep_alive: llamacpp"),
            ("VLM_REASONING_LEVEL", "low", "Provider does not support reasoning_level: llamacpp"),
        )
        for field_name, value, message in cases:
            with self.subTest(field_name=field_name):
                model = {
                    "VLM_NAME": "Preset A",
                    "VLM_PROVIDER": "llamacpp",
                    "VLM_BASE_URL": "http://127.0.0.1:8002",
                    "VLM_MODEL": "demo",
                    "VLM_MAX_OUTPUT_TOKENS": 512,
                    "VLM_TIMEOUT_SECONDS": 180,
                    "VLM_TEMPERATURE": 0.0,
                    "VLM_RESPONSE_SCHEMA_MODE": "on",
                    "VLM_JSON_VALIDATION_MODE": "strict",
                }
                model[field_name] = value
                path = self.write_models([model])
                with self.assertRaisesRegex(ValueError, message):
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

    def test_resolve_vlm_model_config_requires_exact_name(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(ValueError, 'unknown VLM_NAME "qwen3.5:19b"'):
            manual_vlm_models.resolve_vlm_model_config(
                loaded.models,
                {"VLM_NAME": "qwen3.5:19b"},
            )

    def test_resolve_vlm_model_config_requires_name(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(ValueError, "missing VLM_NAME in .vocatio"):
            manual_vlm_models.resolve_vlm_model_config(loaded.models, {})

    def test_resolve_vlm_model_config_applies_local_override(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        resolved = manual_vlm_models.resolve_vlm_model_config(
            loaded.models,
            {
                "VLM_NAME": "Ollama localhost qwen3.5:9b",
                "VLM_TEMPERATURE": "1.0",
                "IGNORED_FIELD": "value",
            },
        )

        self.assertEqual(resolved["VLM_NAME"], "Ollama localhost qwen3.5:9b")
        self.assertEqual(resolved["VLM_TEMPERATURE"], 1.0)
        self.assertEqual(resolved["VLM_MODEL"], "qwen3.5:9b")
        self.assertNotIn("IGNORED_FIELD", resolved)

    def test_resolve_checked_in_llamacpp_example_preset(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        resolved = manual_vlm_models.resolve_vlm_model_config(
            loaded.models,
            {
                "VLM_NAME": "llama.cpp localhost gemma-4-E4B-it-GGUF:Q8_0",
            },
        )

        self.assertEqual(
            resolved["VLM_NAME"],
            "llama.cpp localhost gemma-4-E4B-it-GGUF:Q8_0",
        )
        self.assertEqual(resolved["VLM_PROVIDER"], "llamacpp")
        self.assertNotIn("VLM_CONTEXT_TOKENS", resolved)
        self.assertNotIn("VLM_KEEP_ALIVE", resolved)
        self.assertNotIn("VLM_REASONING_LEVEL", resolved)

    def test_resolve_checked_in_vllm_example_preset(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        resolved = manual_vlm_models.resolve_vlm_model_config(
            loaded.models,
            {
                "VLM_NAME": "vLLM localhost Qwen3.5-0.8B",
            },
        )

        self.assertEqual(resolved["VLM_NAME"], "vLLM localhost Qwen3.5-0.8B")
        self.assertEqual(resolved["VLM_PROVIDER"], "vllm")
        self.assertNotIn("VLM_CONTEXT_TOKENS", resolved)
        self.assertNotIn("VLM_KEEP_ALIVE", resolved)
        self.assertNotIn("VLM_REASONING_LEVEL", resolved)

    def test_resolve_vlm_model_config_rejects_provider_mismatch(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(
            ValueError,
            "Provider does not support context_tokens: llamacpp",
        ):
            manual_vlm_models.resolve_vlm_model_config(
                loaded.models,
                {
                    "VLM_NAME": "Ollama localhost qwen3.5:9b",
                    "VLM_PROVIDER": "llamacpp",
                },
            )

    def test_resolve_vlm_model_config_rejects_non_ollama_keep_alive_override(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(
            ValueError,
            "Provider does not support keep_alive: llamacpp",
        ):
            manual_vlm_models.resolve_vlm_model_config(
                loaded.models,
                {
                    "VLM_NAME": "llama.cpp localhost gemma-4-E4B-it-GGUF:Q8_0",
                    "VLM_KEEP_ALIVE": "15m",
                },
            )

    def test_resolve_vlm_model_config_rejects_non_ollama_reasoning_override(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(
            ValueError,
            "Provider does not support reasoning_level: vllm",
        ):
            manual_vlm_models.resolve_vlm_model_config(
                loaded.models,
                {
                    "VLM_NAME": "vLLM localhost Qwen3.5-0.8B",
                    "VLM_REASONING_LEVEL": "low",
                },
            )

    def test_resolve_vlm_model_config_rejects_non_ollama_context_override(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(
            ValueError,
            "Provider does not support context_tokens: llamacpp",
        ):
            manual_vlm_models.resolve_vlm_model_config(
                loaded.models,
                {
                    "VLM_NAME": "llama.cpp localhost gemma-4-E4B-it-GGUF:Q8_0",
                    "VLM_CONTEXT_TOKENS": "4096",
                },
            )

    def test_resolve_premodel_model_config_requires_name(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(ValueError, "missing PREMODEL_NAME in .vocatio"):
            manual_vlm_models.resolve_premodel_model_config(loaded.models, {})

    def test_resolve_premodel_model_config_requires_exact_name(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        with self.assertRaisesRegex(
            ValueError,
            'unknown PREMODEL_NAME "missing-preset"',
        ):
            manual_vlm_models.resolve_premodel_model_config(
                loaded.models,
                {"PREMODEL_NAME": "missing-preset"},
            )

    def test_resolve_premodel_model_config_applies_local_override(self) -> None:
        loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())

        resolved = manual_vlm_models.resolve_premodel_model_config(
            loaded.models,
            {
                "PREMODEL_NAME": "llama.cpp localhost qwen3.5-4b-pre",
                "PREMODEL_TIMEOUT_SECONDS": "240",
                "UNRELATED": "ignored",
            },
        )

        self.assertEqual(
            resolved["PREMODEL_NAME"],
            "llama.cpp localhost qwen3.5-4b-pre",
        )
        self.assertEqual(resolved["PREMODEL_TIMEOUT_SECONDS"], 240.0)
        self.assertEqual(
            resolved["PREMODEL_MODEL"],
            "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
        )
        self.assertNotIn("UNRELATED", resolved)


if __name__ == "__main__":
    unittest.main()
