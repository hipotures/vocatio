# Manual VLM Description Tooltip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `VLM_DESCRIPTION` support to manual VLM presets so the GUI can keep preset labels short, show long descriptions in tooltips, and preserve YAML key order in rendered model metadata.

**Architecture:** Extend the preset loader to accept and preserve an optional `VLM_DESCRIPTION` field without disturbing the existing preset contract. Then wire the GUI combobox and manual VLM result formatter to consume that metadata with fallback-to-`VLM_NAME` behavior and preserve the original YAML field order in `Model config:`.

**Tech Stack:** Python 3, PySide6, PyYAML, unittest script tests

---

## File Map

- Modify: `conf/manual_vlm_models.yaml.example`
  - Add example `VLM_DESCRIPTION` values to show intended preset structure.
- Modify: `scripts/pipeline/lib/manual_vlm_models.py`
  - Accept optional `VLM_DESCRIPTION` during validation and preserve YAML key order for later rendering.
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - Populate combobox and popup-item tooltips from preset description metadata.
  - Preserve YAML key order in rendered `Model config:` for manual VLM results.
- Modify: `scripts/pipeline/test_manual_vlm_models.py`
  - Cover optional `VLM_DESCRIPTION` acceptance and fallback behavior.
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - Cover combobox tooltip behavior, popup item tooltip behavior, and YAML-order rendering in manual VLM results.

### Task 1: Extend Preset Loader Contract

**Files:**
- Modify: `scripts/pipeline/lib/manual_vlm_models.py`
- Test: `scripts/pipeline/test_manual_vlm_models.py`

- [ ] **Step 1: Write the failing loader tests**

Add tests like:

```python
def test_load_manual_vlm_models_accepts_optional_description(self):
    text = """
models:
  - VLM_NAME: "Preset A"
    VLM_DESCRIPTION: "Long description"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "model-a"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "15m"
    VLM_TIMEOUT_SECONDS: 300
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "false"
    VLM_RESPONSE_SCHEMA_MODE: "off"
    VLM_JSON_VALIDATION_MODE: "strict"
"""
    models, _, error = manual_vlm_models.load_manual_vlm_models(path)
    self.assertIsNone(error)
    self.assertEqual(models[0]["VLM_DESCRIPTION"], "Long description")

def test_load_manual_vlm_models_allows_missing_description(self):
    ...
    self.assertNotIn("VLM_DESCRIPTION", models[0])
```

- [ ] **Step 2: Run loader tests to verify failure**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected: FAIL in new `VLM_DESCRIPTION` tests.

- [ ] **Step 3: Implement minimal loader support**

Update the preset normalization path so:

```python
description = normalize_optional_scalar(model.get("VLM_DESCRIPTION"))
if description:
    normalized_model["VLM_DESCRIPTION"] = description
```

Requirements:
- `VLM_DESCRIPTION` stays optional
- empty string behaves as missing
- YAML key order is not destroyed by normalization for fields that survive into the normalized mapping

- [ ] **Step 4: Run loader tests to verify pass**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected: PASS.

- [ ] **Step 5: Commit loader contract change**

```bash
git add scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/test_manual_vlm_models.py
git commit -m "feat: add manual VLM preset descriptions"
```

### Task 2: Add Example Preset Descriptions

**Files:**
- Modify: `conf/manual_vlm_models.yaml.example`

- [ ] **Step 1: Update example presets**

Add example description fields directly after `VLM_NAME`, for example:

```yaml
  - VLM_NAME: "Ollama localhost qwen3.5:9b"
    VLM_DESCRIPTION: "Ollama qwen3.5:9b on localhost:11434, temp 0, strict JSON, 16k context"
```

Apply the same pattern to the llama.cpp and vLLM examples.

- [ ] **Step 2: Verify example file structure**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected: PASS with the updated example semantics still accepted by the loader tests.

- [ ] **Step 3: Commit example file update**

```bash
git add conf/manual_vlm_models.yaml.example
git commit -m "docs: add descriptions to manual VLM preset example"
```

### Task 3: Add Combobox and Popup Tooltip Support

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing GUI tooltip tests**

Add tests like:

```python
def test_manual_vlm_choice_uses_description_as_combobox_tooltip(self):
    section = review_gui.build_manual_vlm_analyze_section(
        {"status": "idle"},
        preset_names=["Short A"],
        selected_name="Short A",
        preset_descriptions={"Short A": "Long description"},
    )
    widget = window.build_info_section_widget(section)
    combo = widget.findChild(QComboBox)
    self.assertEqual(combo.toolTip(), "Long description")

def test_manual_vlm_choice_falls_back_to_name_when_description_missing(self):
    ...
```

Also add a popup-item tooltip test against the combobox model data or delegate-facing tooltip role.

- [ ] **Step 2: Run GUI tests to verify failure**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL in new tooltip tests.

- [ ] **Step 3: Implement combobox metadata wiring**

Update the GUI section config so it carries both:

```python
{
    "choice_items": [...],
    "choice_value": ...,
    "choice_tooltips": {...},
}
```

Then update widget construction to:

- set the combobox tooltip from the selected preset description
- attach each preset description to the corresponding popup item tooltip
- fall back to `VLM_NAME` when the description is missing

- [ ] **Step 4: Run GUI tests to verify pass**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: PASS.

- [ ] **Step 5: Commit tooltip behavior**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "feat: add manual VLM preset tooltips"
```

### Task 4: Preserve YAML Order in Manual VLM Result Metadata

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing result-order test**

Add a test like:

```python
def test_manual_vlm_result_renders_model_config_in_yaml_order(self):
    state = {
        "status": "result",
        "preset_name": "Short A",
        "model_config": {
            "VLM_NAME": "Short A",
            "VLM_DESCRIPTION": "Long description",
            "VLM_PROVIDER": "ollama",
            "VLM_BASE_URL": "http://127.0.0.1:11434",
        },
    }
    text = review_gui.format_manual_vlm_analyze_result_text(state)
    self.assertIn(
        "Model config:\\n"
        "  VLM_NAME: Short A\\n"
        "  VLM_DESCRIPTION: Long description\\n"
        "  VLM_PROVIDER: ollama\\n"
        "  VLM_BASE_URL: http://127.0.0.1:11434",
        text,
    )
```

- [ ] **Step 2: Run GUI tests to verify failure**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL if config rendering still reorders or drops the new field.

- [ ] **Step 3: Implement minimal ordered rendering**

Ensure the manual VLM state stores `model_config` as the normalized preset mapping in original YAML order and render it without sorting:

```python
for key, value in model_config.items():
    lines.append(f"  {key}: {format_value(value)}")
```

Do not extract `VLM_DESCRIPTION` into a separately reordered block.

- [ ] **Step 4: Run GUI tests to verify pass**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: PASS.

- [ ] **Step 5: Commit ordered metadata rendering**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "feat: preserve manual VLM config order in results"
```

### Task 5: Focused Regression Pass

**Files:**
- Verify only

- [ ] **Step 1: Run focused loader and GUI tests**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
python3 -m py_compile scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- manual VLM model tests: PASS
- GUI diagnostics tests: PASS
- py_compile: no output, exit 0

- [ ] **Step 2: Final commit if regression pass required follow-up**

If the regression pass required any small cleanup:

```bash
git add scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_review_gui_image_only_diagnostics.py conf/manual_vlm_models.yaml.example
git commit -m "test: finalize manual VLM description tooltip rollout"
```

- [ ] **Step 3: Confirm clean working tree**

Run:

```bash
git status --short
```

Expected: no output.
