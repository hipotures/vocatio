#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.lib import manual_vlm_models


class ManualVlmModelsTest(unittest.TestCase):
    def write_yaml(self, text: str) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="manual-vlm-models-"))
        path = temp_dir / "manual_vlm_models.yaml"
        path.write_text(text, encoding="utf-8")
        return path

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

    def test_compute_models_md5_changes_with_file_content(self) -> None:
        path = self.write_yaml("models: []\n")
        before = manual_vlm_models.compute_manual_vlm_models_md5(path)
        path.write_text("models:\n  - VLM_NAME: \"Changed\"\n", encoding="utf-8")
        after = manual_vlm_models.compute_manual_vlm_models_md5(path)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
