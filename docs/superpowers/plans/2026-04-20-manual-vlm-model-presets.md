# Manual VLM Model Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manual VLM model preset selection in the review GUI, backed by `conf/manual_vlm_models.yaml`, with MD5-based reload, retry behavior, and result rendering that exposes the selected preset and full request config.

**Architecture:** Introduce a small preset-loading layer for manual VLM only, keep day/runtime context separate from model/request configuration, and thread the selected preset through the existing async manual VLM worker path. The GUI owns preset loading, MD5 tracking, dropdown refresh, and config messaging; the actual VLM request continues to reuse the shared probe helpers, but with request parameters sourced from the chosen preset instead of `.vocatio`.

**Tech Stack:** Python 3, PySide6, PyYAML, existing `probe_vlm_photo_boundaries.py` VLM transport helpers, `unittest`

---

## File Structure

- Create: `conf/manual_vlm_models.yaml`
  - checked-in example preset file for manual GUI VLM presets
- Create: `scripts/pipeline/lib/manual_vlm_models.py`
  - focused loader/validator/MD5 helpers for `conf/manual_vlm_models.yaml`
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - GUI preset dropdown, startup load, MD5 reload, manual VLM retry flow, result formatting, config/error subtitle messaging
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - GUI and formatting tests for preset loading, reload, retries, and result rendering
- Create: `scripts/pipeline/test_manual_vlm_models.py`
  - direct unit tests for YAML parsing, validation, and MD5 handling
- Modify: `README.md`
  - document the new manual GUI preset file and clarify that it applies only to `Manual VLM analyze`

### Task 1: Add preset config file and loader module

**Files:**
- Create: `conf/manual_vlm_models.yaml`
- Create: `scripts/pipeline/lib/manual_vlm_models.py`
- Test: `scripts/pipeline/test_manual_vlm_models.py`

- [ ] **Step 1: Write the failing tests for preset loading and validation**

```python
import tempfile
import unittest
from pathlib import Path

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
```

- [ ] **Step 2: Run the new loader tests to verify they fail**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected:

- FAIL with `ModuleNotFoundError` or missing symbol errors for `manual_vlm_models`

- [ ] **Step 3: Create the loader module with explicit validation and MD5 helpers**

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import yaml


MODEL_FIELDS = (
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
)

SUPPORTED_PROVIDERS = {"ollama", "llamacpp", "vllm"}


@dataclass(frozen=True)
class ManualVlmModelsConfig:
    models: List[Dict[str, Any]]
    md5_hex: str


def compute_manual_vlm_models_md5(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.md5(payload).hexdigest()


def _validate_model_entry(model: Any, index: int) -> Dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError(f"models[{index}] must be a mapping")
    normalized = dict(model)
    for field in MODEL_FIELDS:
        if field not in normalized:
            raise ValueError(f'missing {field} in preset at index {index}')
    name = str(normalized.get("VLM_NAME", "") or "").strip()
    if not name:
        raise ValueError(f"models[{index}] has empty VLM_NAME")
    provider = str(normalized["VLM_PROVIDER"])
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f'unsupported VLM_PROVIDER "{provider}" in preset "{name}"')
    if int(normalized["VLM_CONTEXT_TOKENS"]) <= 0:
        raise ValueError(f'VLM_CONTEXT_TOKENS must be > 0 in preset "{name}"')
    if int(normalized["VLM_MAX_OUTPUT_TOKENS"]) <= 0:
        raise ValueError(f'VLM_MAX_OUTPUT_TOKENS must be > 0 in preset "{name}"')
    if float(normalized["VLM_TIMEOUT_SECONDS"]) <= 0:
        raise ValueError(f'VLM_TIMEOUT_SECONDS must be > 0 in preset "{name}"')
    return normalized


def load_manual_vlm_models(path: Path) -> ManualVlmModelsConfig:
    if not path.is_file():
        raise ValueError(f"manual VLM model config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manual VLM model config must be a mapping")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manual VLM model config must define a non-empty models list")
    normalized_models: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, model in enumerate(models):
        normalized = _validate_model_entry(model, index)
        name = str(normalized["VLM_NAME"])
        if name in seen_names:
            raise ValueError(f'duplicate VLM_NAME "{name}"')
        seen_names.add(name)
        normalized_models.append(normalized)
    return ManualVlmModelsConfig(
        models=normalized_models,
        md5_hex=compute_manual_vlm_models_md5(path),
    )
```

- [ ] **Step 4: Add the checked-in preset file**

```yaml
models:
  - VLM_NAME: "Qwen2.5-VL 7B temp 0"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "qwen2.5vl:7b"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "30m"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "low"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"

  - VLM_NAME: "Qwen2.5-VL GGUF think off"
    VLM_PROVIDER: "llamacpp"
    VLM_BASE_URL: "http://127.0.0.1:8080"
    VLM_MODEL: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "off"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"

  - VLM_NAME: "InternVL3 8B temp 0.2"
    VLM_PROVIDER: "vllm"
    VLM_BASE_URL: "http://127.0.0.1:8000"
    VLM_MODEL: "OpenGVLab/InternVL3-8B"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "0"
    VLM_TIMEOUT_SECONDS: 180
    VLM_TEMPERATURE: 0.2
    VLM_REASONING_LEVEL: "inherit"
    VLM_RESPONSE_SCHEMA_MODE: "on"
    VLM_JSON_VALIDATION_MODE: "strict"
```

- [ ] **Step 5: Run the loader tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected:

- PASS with all tests green

- [ ] **Step 6: Commit the loader and preset file**

```bash
git add conf/manual_vlm_models.yaml scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/test_manual_vlm_models.py
git commit -m "Add manual VLM model preset loader"
```

### Task 2: Wire preset loading and dropdown state into the review GUI

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write failing GUI tests for startup preset loading and invalid config state**

```python
def test_manual_vlm_section_uses_first_loaded_model_by_default() -> None:
    window = build_test_window_with_two_selected_rows()
    window.manual_vlm_models = [
        {"VLM_NAME": "Preset A"},
        {"VLM_NAME": "Preset B"},
    ]
    window.manual_vlm_models_md5 = "abc"
    section = review_gui.build_manual_vlm_analyze_section(window)
    assert section["choice_items"] == ["Preset A", "Preset B"]
    assert section["choice_value"] == "Preset A"


def test_manual_vlm_section_disables_analyze_when_model_config_error_exists() -> None:
    window = build_test_window_with_two_selected_rows()
    window.manual_vlm_models = []
    window.manual_vlm_models_error = "bad yaml"
    section = review_gui.build_manual_vlm_analyze_section(window)
    assert section["action_enabled"] is False
    assert "bad yaml" in section["description"]
```

- [ ] **Step 2: Run the GUI test file to verify the new tests fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL with missing keys such as `choice_items`, `choice_value`, or missing manual VLM preset state

- [ ] **Step 3: Add GUI-owned preset state and startup loading**

```python
MANUAL_VLM_MODELS_PATH = Path("conf/manual_vlm_models.yaml")


def load_manual_vlm_models_for_gui(repo_root: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    config_path = repo_root / MANUAL_VLM_MODELS_PATH
    try:
        loaded = manual_vlm_models.load_manual_vlm_models(config_path)
    except ValueError as exc:
        return [], None, f"Model config error: {exc}"
    return loaded.models, loaded.md5_hex, None


class MainWindow(QMainWindow):
    def __init__(...):
        ...
        self.manual_vlm_models: List[Dict[str, Any]] = []
        self.manual_vlm_models_md5: Optional[str] = None
        self.manual_vlm_models_error: Optional[str] = None
        self.manual_vlm_selected_name: Optional[str] = None
        self.reload_manual_vlm_models(startup=True)

    def reload_manual_vlm_models(self, startup: bool = False) -> None:
        models, md5_hex, error_text = load_manual_vlm_models_for_gui(REPO_ROOT)
        self.manual_vlm_models = models
        self.manual_vlm_models_md5 = md5_hex
        self.manual_vlm_models_error = error_text
        if models:
            available_names = [str(model["VLM_NAME"]) for model in models]
            if self.manual_vlm_selected_name not in available_names:
                self.manual_vlm_selected_name = available_names[0]
        else:
            self.manual_vlm_selected_name = None
```

- [ ] **Step 4: Extend the manual VLM section structure with dropdown data**

```python
def build_manual_vlm_analyze_section(window: "MainWindow") -> Dict[str, Any]:
    ...
    preset_names = [str(model["VLM_NAME"]) for model in getattr(window, "manual_vlm_models", [])]
    selected_name = getattr(window, "manual_vlm_selected_name", None)
    description = getattr(window, "manual_vlm_models_error", None) or "Ephemeral runtime state for manual VLM boundary analysis."
    return {
        "title": "Manual VLM analyze",
        "description": description,
        "choice_items": preset_names,
        "choice_value": selected_name,
        "choice_enabled": bool(preset_names) and not active_action_locked,
        "action_text": "Analyze",
        "action_enabled": bool(preset_names) and selected_name is not None and not active_action_locked,
        ...
    }
```

- [ ] **Step 5: Render the dropdown in the section widget**

```python
if choice_items is not None:
    combo = QComboBox(container)
    combo.setObjectName("infoSectionChoiceCombo")
    combo.addItems(choice_items)
    if choice_value in choice_items:
        combo.setCurrentText(choice_value)
    combo.setEnabled(bool(choice_enabled))
    combo.currentTextChanged.connect(on_choice_changed)
    controls_layout.addWidget(combo)
```

- [ ] **Step 6: Run the GUI tests again to verify startup preset behavior passes**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS for the new startup-loading and disabled-error tests

- [ ] **Step 7: Commit the GUI preset loading skeleton**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Add manual VLM preset selection UI"
```

### Task 3: Add MD5-triggered preset reload on Analyze

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write failing tests for MD5-based reload behavior**

```python
def test_manual_vlm_analyze_reloads_models_when_md5_changes() -> None:
    window = build_test_window_with_two_selected_rows()
    window.manual_vlm_models = [{"VLM_NAME": "Preset A"}]
    window.manual_vlm_models_md5 = "old"
    reloaded = {"called": False}

    def fake_reload(startup: bool = False) -> None:
        reloaded["called"] = True
        window.manual_vlm_models = [{"VLM_NAME": "Preset B"}]
        window.manual_vlm_models_md5 = "new"
        window.manual_vlm_selected_name = "Preset B"
        window.manual_vlm_models_error = None

    window.reload_manual_vlm_models = fake_reload
    review_gui.refresh_manual_vlm_models_if_needed(window, latest_md5="new")
    assert reloaded["called"] is True
    assert window.manual_vlm_selected_name == "Preset B"


def test_manual_vlm_analyze_keeps_selection_when_name_survives_reload() -> None:
    window = build_test_window_with_two_selected_rows()
    window.manual_vlm_selected_name = "Preset B"
    review_gui.apply_reloaded_manual_vlm_models(
        window,
        models=[{"VLM_NAME": "Preset A"}, {"VLM_NAME": "Preset B"}],
        md5_hex="next",
        message="Models reloaded from config.",
    )
    assert window.manual_vlm_selected_name == "Preset B"
```

- [ ] **Step 2: Run the GUI test file to verify reload tests fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL with missing reload helpers or wrong selection retention

- [ ] **Step 3: Implement explicit reload helpers and subtitle messaging**

```python
def apply_reloaded_manual_vlm_models(
    window: "MainWindow",
    models: List[Dict[str, Any]],
    md5_hex: Optional[str],
    message: Optional[str],
) -> None:
    previous_name = getattr(window, "manual_vlm_selected_name", None)
    names = [str(model["VLM_NAME"]) for model in models]
    window.manual_vlm_models = list(models)
    window.manual_vlm_models_md5 = md5_hex
    window.manual_vlm_models_error = None
    window.manual_vlm_status_message = message
    if previous_name in names:
        window.manual_vlm_selected_name = previous_name
    else:
        window.manual_vlm_selected_name = names[0] if names else None


def refresh_manual_vlm_models_if_needed(window: "MainWindow", latest_md5: str) -> None:
    current_md5 = getattr(window, "manual_vlm_models_md5", None)
    if current_md5 == latest_md5:
        return
    models, md5_hex, error_text = load_manual_vlm_models_for_gui(REPO_ROOT)
    if error_text is not None:
        window.manual_vlm_models = []
        window.manual_vlm_models_md5 = md5_hex
        window.manual_vlm_models_error = error_text
        window.manual_vlm_selected_name = None
        return
    apply_reloaded_manual_vlm_models(
        window,
        models=models,
        md5_hex=md5_hex,
        message="Models reloaded from config.",
    )
```

- [ ] **Step 4: Call the MD5 check before starting manual VLM work**

```python
def run_manual_vlm_analyze(window: "MainWindow") -> None:
    latest_md5 = manual_vlm_models.compute_manual_vlm_models_md5(REPO_ROOT / MANUAL_VLM_MODELS_PATH)
    refresh_manual_vlm_models_if_needed(window, latest_md5)
    if getattr(window, "manual_vlm_models_error", None):
        set_manual_action_state(window, "vlm", {"status": "error", "error": window.manual_vlm_models_error})
        refresh_info_panel(window)
        return
    ...
```

- [ ] **Step 5: Run the GUI tests again to verify reload behavior passes**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS for MD5 reload, preserved selection, and fallback-to-first behavior

- [ ] **Step 6: Commit the MD5 reload integration**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Reload manual VLM presets on config change"
```

### Task 4: Use selected preset for manual VLM requests and add retry handling

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write failing tests for preset-backed request construction and retries**

```python
def test_manual_vlm_uses_selected_preset_values_in_request() -> None:
    window = build_test_window_with_two_selected_rows()
    window.manual_vlm_models = [
        {
            "VLM_NAME": "Preset A",
            "VLM_PROVIDER": "vllm",
            "VLM_BASE_URL": "http://127.0.0.1:8000",
            "VLM_MODEL": "model-a",
            "VLM_CONTEXT_TOKENS": 4096,
            "VLM_MAX_OUTPUT_TOKENS": 256,
            "VLM_KEEP_ALIVE": "0",
            "VLM_TIMEOUT_SECONDS": 90,
            "VLM_TEMPERATURE": 0.2,
            "VLM_REASONING_LEVEL": "inherit",
            "VLM_RESPONSE_SCHEMA_MODE": "on",
            "VLM_JSON_VALIDATION_MODE": "strict",
        }
    ]
    window.manual_vlm_selected_name = "Preset A"
    result = review_gui.compute_manual_vlm_analyze_result(window, fake_vlm_context())
    assert result["model_name"] == "Preset A"
    assert result["model_config"]["VLM_PROVIDER"] == "vllm"


def test_manual_vlm_retries_twice_before_success() -> None:
    attempts = {"count": 0}

    def flaky_call(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("temporary failure")
        return {"decision": "no_cut"}

    result = review_gui.run_manual_vlm_with_retries(flaky_call, sleep_seconds=0)
    assert result["attempts"] == 3
    assert result["succeeded_on_attempt"] == 3
```

- [ ] **Step 2: Run the GUI tests to verify the new request/retry tests fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL with missing `model_name`, missing retry helper, or old `.vocatio`-based request fields

- [ ] **Step 3: Add a preset lookup helper and use it to build the VLM request**

```python
def get_selected_manual_vlm_model(window: "MainWindow") -> Dict[str, Any]:
    selected_name = str(getattr(window, "manual_vlm_selected_name", "") or "").strip()
    for model in getattr(window, "manual_vlm_models", []):
        if str(model.get("VLM_NAME", "")) == selected_name:
            return dict(model)
    raise ValueError(f'manual VLM preset not found: "{selected_name}"')


def build_manual_vlm_runtime_from_preset(window: "MainWindow") -> Dict[str, Any]:
    preset = get_selected_manual_vlm_model(window)
    return {
        "model_name": str(preset["VLM_NAME"]),
        "provider": str(preset["VLM_PROVIDER"]),
        "base_url": str(preset["VLM_BASE_URL"]),
        "model": str(preset["VLM_MODEL"]),
        "context_tokens": int(preset["VLM_CONTEXT_TOKENS"]),
        "max_output_tokens": int(preset["VLM_MAX_OUTPUT_TOKENS"]),
        "keep_alive": str(preset["VLM_KEEP_ALIVE"]),
        "timeout_seconds": float(preset["VLM_TIMEOUT_SECONDS"]),
        "temperature": float(preset["VLM_TEMPERATURE"]),
        "reasoning_level": str(preset["VLM_REASONING_LEVEL"]),
        "response_schema_mode": str(preset["VLM_RESPONSE_SCHEMA_MODE"]),
        "json_validation_mode": str(preset["VLM_JSON_VALIDATION_MODE"]),
        "model_config": dict(preset),
    }
```

- [ ] **Step 4: Add the retry wrapper and apply it inside the manual VLM worker path**

```python
def run_manual_vlm_with_retries(call_fn, sleep_seconds: int = 5) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt_index in range(1, 4):
        try:
            result = call_fn()
            result["attempts"] = attempt_index
            if attempt_index > 1:
                result["succeeded_on_attempt"] = attempt_index
            return result
        except ValueError as exc:
            last_error = exc
            if attempt_index >= 3:
                break
            time.sleep(sleep_seconds)
    raise ValueError(str(last_error) if last_error is not None else "manual VLM request failed")
```

- [ ] **Step 5: Run the GUI tests again to verify preset-backed request and retry behavior passes**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS for request parameter ownership and retry behavior

- [ ] **Step 6: Commit preset-backed manual VLM execution**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Use model presets for manual VLM analyze"
```

### Task 5: Render preset metadata and retry details in the manual VLM result

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write failing formatting tests for model metadata and attempts**

```python
def test_manual_vlm_result_text_includes_model_and_config_block() -> None:
    text = review_gui.format_manual_vlm_analyze_result_text(
        {
            "status": "result",
            "decision": "cut_after_3",
            "left_segment_type": "dance",
            "right_segment_type": "rehearsal",
            "summary": "Boundary detected.",
            "left_anchor": "left.jpg",
            "right_anchor": "right.jpg",
            "model_name": "Preset A",
            "model_config": {
                "VLM_PROVIDER": "ollama",
                "VLM_MODEL": "qwen",
            },
            "attempts": 2,
            "succeeded_on_attempt": 2,
            "debug_file_paths": ["/tmp/a.json"],
        }
    )
    assert "Model: Preset A" in text
    assert "Model config:" in text
    assert "Attempts: 2" in text
    assert "Succeeded on attempt: 2" in text


def test_manual_vlm_error_text_keeps_model_metadata() -> None:
    text = review_gui.format_manual_vlm_analyze_result_text(
        {
            "status": "error",
            "error": "timeout",
            "left_anchor": "left.jpg",
            "right_anchor": "right.jpg",
            "model_name": "Preset A",
            "model_config": {"VLM_PROVIDER": "ollama"},
            "attempts": 3,
            "debug_file_paths": ["/tmp/a.json"],
        }
    )
    assert "Status: error" in text
    assert "Model: Preset A" in text
    assert "Attempts: 3" in text
```

- [ ] **Step 2: Run the GUI tests to verify formatting tests fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL because current formatter does not include preset metadata and attempt details

- [ ] **Step 3: Expand the formatter to include anchors, model block, and attempts in fixed order**

```python
MANUAL_VLM_MODEL_FIELDS = (
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
)


def append_manual_vlm_model_block(lines: List[str], payload: Dict[str, Any]) -> None:
    lines.extend(
        (
            "",
            f'Model: {payload.get("model_name", "-")}',
            "Model config:",
        )
    )
    model_config = payload.get("model_config", {}) or {}
    for field in MANUAL_VLM_MODEL_FIELDS:
        if field in model_config:
            lines.append(f"  {field}: {model_config[field]}")
    attempts = payload.get("attempts")
    if attempts is not None:
        lines.extend(("", f"Attempts: {attempts}"))
    if payload.get("succeeded_on_attempt") is not None:
        lines.append(f'Succeeded on attempt: {payload["succeeded_on_attempt"]}')
```

- [ ] **Step 4: Run the GUI tests again to verify formatting passes**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS for result and error formatting with preset metadata

- [ ] **Step 5: Commit the manual VLM result rendering changes**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Show model preset details in manual VLM results"
```

### Task 6: Document the preset file and manual GUI behavior

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README documentation for the preset file**

```md
### Manual VLM model presets

`Manual VLM analyze` in `review_performance_proxy_gui.py` reads model presets from:

- `conf/manual_vlm_models.yaml`

This file applies only to the GUI's manual VLM action. It does not replace:

- `.vocatio` day settings such as `VLM_WINDOW_RADIUS`
- batch `probe_vlm_photo_boundaries.py`

Each preset entry must define:

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

When the GUI starts it loads the preset file and selects the first preset by default. When you click `Analyze`, the GUI recomputes the file MD5 and reloads the dropdown if the file changed.
```

- [ ] **Step 2: Verify the README section reads cleanly**

Run:

```bash
rg -n "Manual VLM model presets|manual_vlm_models.yaml" README.md
```

Expected:

- two matching lines in the new section

- [ ] **Step 3: Commit the README update**

```bash
git add README.md
git commit -m "Document manual VLM model presets"
```

### Task 7: Run the full focused verification pass

**Files:**
- Test: `scripts/pipeline/test_manual_vlm_models.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Run the loader unit tests**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected:

- PASS

- [ ] **Step 2: Run the GUI diagnostics test suite**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS

- [ ] **Step 3: Compile-check the touched Python files**

Run:

```bash
python3 -m py_compile scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- no output

- [ ] **Step 4: Commit any final verification-only cleanups**

```bash
git add conf/manual_vlm_models.yaml scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_review_gui_image_only_diagnostics.py README.md
git commit -m "Polish manual VLM preset integration"
```

## Self-Review

- Spec coverage:
  - dedicated YAML config path: covered in Tasks 1, 2, 3, 6
  - all model/request fields from YAML: covered in Task 4
  - dropdown with `VLM_NAME`: covered in Task 2
  - MD5 reload on `Analyze`: covered in Task 3
  - selection retention / fallback after reload: covered in Task 3
  - result rendering with model name, full config, anchors, spacing: covered in Task 5
  - retry behavior with 3 attempts and 5-second delays: covered in Task 4
  - invalid config disables analyze and shows error: covered in Tasks 1, 2, 3
- Placeholder scan:
  - no `TODO`/`TBD`
  - each task includes concrete files, code, commands, expected outcomes, and commit step
- Type consistency:
  - `VLM_NAME` is used consistently as the dropdown/display key
  - MD5 state is consistently `manual_vlm_models_md5`
  - result payload consistently uses `model_name`, `model_config`, `attempts`, and `succeeded_on_attempt`

