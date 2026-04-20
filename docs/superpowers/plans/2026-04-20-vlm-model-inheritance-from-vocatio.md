# VLM Model Inheritance From `.vocatio` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `conf/vlm_models.yaml` the only source of VLM/PREMODEL model definitions, with exact `*_NAME` selection from `.vocatio` and local inheritance overrides only for allowed model fields.

**Architecture:** Keep a shared preset file and shared loader, then add two explicit resolution paths: one for VLM and one for PREMODEL. Each path reads `.vocatio`, requires an exact preset name, merges only allowed local overrides, and fails fast on invalid presets or provider-unsupported parameters. GUI manual VLM continues to use YAML presets directly, but all code and docs switch to the new file name `conf/vlm_models.yaml`.

**Tech Stack:** Python 3, PyYAML, existing `.vocatio` loader in `scripts/pipeline/lib/workspace_dir.py`, existing provider validation in `scripts/pipeline/lib/vlm_transport.py`, PySide6 GUI tests, `unittest`

---

## File Structure

- Create: `conf/vlm_models.yaml.example`
  - tracked example file using the new canonical filename
- Modify: `.gitignore`
  - ignore `conf/vlm_models.yaml` instead of `conf/manual_vlm_models.yaml`
- Modify: `README.md`
  - replace old filename references and document exact-name inheritance from `.vocatio`
- Modify: `scripts/pipeline/lib/manual_vlm_models.py`
  - keep loader responsibility, but broaden it to load from `conf/vlm_models.yaml`, support shared preset definitions, and expose exact-name resolution helpers for VLM and PREMODEL
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - replace direct model-field reads from `.vocatio` with resolved VLM preset + override config
- Modify: `scripts/pipeline/build_photo_pre_model_annotations.py`
  - replace direct model-field reads from `.vocatio` with resolved PREMODEL preset + override config
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - use `conf/vlm_models.yaml` only, keep manual VLM dropdown working against VLM presets
- Modify: `scripts/pipeline/test_manual_vlm_models.py`
  - loader, exact-name lookup, inheritance merge, provider validation, and new filename tests
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
  - VLM inheritance tests
- Modify: `scripts/pipeline/test_build_photo_pre_model_annotations.py`
  - PREMODEL inheritance tests
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - manual GUI path rename and VLM-only preset list coverage

### Task 1: Rename the preset file contract and cover it with tests

**Files:**
- Create: `conf/vlm_models.yaml.example`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_manual_vlm_models.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing path and doc expectation tests**

```python
def test_repo_example_uses_new_vlm_models_filename(self) -> None:
    path = REPO_ROOT / "conf" / "vlm_models.yaml.example"
    loaded = manual_vlm_models.load_manual_vlm_models(path)
    self.assertGreaterEqual(len(loaded.models), 1)


def test_manual_vlm_gui_uses_new_models_path(self) -> None:
    self.assertEqual(
        review_gui.MANUAL_VLM_MODELS_PATH,
        Path("conf/vlm_models.yaml"),
    )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL because only `conf/manual_vlm_models.yaml.example` exists
- FAIL because GUI constant still points at `conf/manual_vlm_models.yaml`

- [ ] **Step 3: Add the new example file and switch all code/docs references to `conf/vlm_models.yaml`**

```yaml
models:
  - VLM_NAME: "qwen3.5:9b"
    VLM_DESCRIPTION: "Ollama qwen3.5:9b on localhost:11434"
    VLM_PROVIDER: "ollama"
    VLM_BASE_URL: "http://127.0.0.1:11434"
    VLM_MODEL: "qwen3.5:9b"
    VLM_CONTEXT_TOKENS: 16384
    VLM_MAX_OUTPUT_TOKENS: 512
    VLM_KEEP_ALIVE: "15m"
    VLM_TIMEOUT_SECONDS: 300
    VLM_TEMPERATURE: 0.0
    VLM_REASONING_LEVEL: "inherit"
    VLM_RESPONSE_SCHEMA_MODE: "off"
    VLM_JSON_VALIDATION_MODE: "strict"
```

```python
MANUAL_VLM_MODELS_PATH = Path("conf/vlm_models.yaml")
```

```text
conf/vlm_models.yaml
```

- [ ] **Step 4: Re-run the focused tests**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS for the new file-path assertions
- remaining failures move to inheritance logic, not filename references

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md conf/vlm_models.yaml.example scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Rename VLM preset file to vlm_models.yaml"
```

### Task 2: Extend the preset loader with exact-name inheritance helpers

**Files:**
- Modify: `scripts/pipeline/lib/manual_vlm_models.py`
- Modify: `scripts/pipeline/test_manual_vlm_models.py`

- [ ] **Step 1: Write failing loader tests for exact-name resolution and `.vocatio` inheritance**

```python
def test_resolve_vlm_model_config_requires_exact_name(self) -> None:
    loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())
    with self.assertRaisesRegex(ValueError, 'unknown VLM_NAME "qwen3.5:19b"'):
        manual_vlm_models.resolve_vlm_model_config(
            loaded.models,
            {"VLM_NAME": "qwen3.5:19b"},
        )


def test_resolve_vlm_model_config_applies_local_override(self) -> None:
    loaded = manual_vlm_models.load_manual_vlm_models(self.repo_example_path())
    resolved = manual_vlm_models.resolve_vlm_model_config(
        loaded.models,
        {
            "VLM_NAME": "qwen3.5:9b",
            "VLM_TEMPERATURE": "1.0",
        },
    )
    self.assertEqual(resolved["VLM_NAME"], "qwen3.5:9b")
    self.assertEqual(resolved["VLM_TEMPERATURE"], 1.0)


def test_resolve_premodel_model_config_requires_name(self) -> None:
    loaded = manual_vlm_models.load_manual_vlm_models(self.write_yaml("""
models:
  - PREMODEL_NAME: "qwen3.5-4b-pre"
    PREMODEL_PROVIDER: "llamacpp"
    PREMODEL_BASE_URL: "http://127.0.0.1:8002"
    PREMODEL_MODEL: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"
    PREMODEL_MAX_OUTPUT_TOKENS: 1024
    PREMODEL_TEMPERATURE: 0.0
    PREMODEL_TIMEOUT_SECONDS: 120
"""))
    with self.assertRaisesRegex(ValueError, "missing PREMODEL_NAME"):
        manual_vlm_models.resolve_premodel_model_config(loaded.models, {})
```

- [ ] **Step 2: Run loader tests to verify they fail**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected:

- FAIL with missing `resolve_vlm_model_config` / `resolve_premodel_model_config`

- [ ] **Step 3: Implement shared-model loading plus exact-name resolution helpers**

```python
VLM_MODEL_FIELDS = (
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

PREMODEL_MODEL_FIELDS = (
    "PREMODEL_NAME",
    "PREMODEL_PROVIDER",
    "PREMODEL_BASE_URL",
    "PREMODEL_MODEL",
    "PREMODEL_MAX_OUTPUT_TOKENS",
    "PREMODEL_TEMPERATURE",
    "PREMODEL_TIMEOUT_SECONDS",
)


def resolve_vlm_model_config(
    models: Sequence[Mapping[str, Any]],
    vocatio_config: Mapping[str, str],
) -> dict[str, Any]:
    preset_name = str(vocatio_config.get("VLM_NAME", "") or "").strip()
    if not preset_name:
        raise ValueError("missing VLM_NAME in .vocatio")
    preset = _find_preset_by_exact_name(models, "VLM_NAME", preset_name)
    resolved = dict(preset)
    for field_name in VLM_MODEL_FIELDS:
        configured = str(vocatio_config.get(field_name, "") or "").strip()
        if field_name == "VLM_NAME" or not configured:
            continue
        resolved[field_name] = configured
    return _normalize_resolved_vlm_config(resolved)


def resolve_premodel_model_config(
    models: Sequence[Mapping[str, Any]],
    vocatio_config: Mapping[str, str],
) -> dict[str, Any]:
    preset_name = str(vocatio_config.get("PREMODEL_NAME", "") or "").strip()
    if not preset_name:
        raise ValueError("missing PREMODEL_NAME in .vocatio")
    preset = _find_preset_by_exact_name(models, "PREMODEL_NAME", preset_name)
    resolved = dict(preset)
    for field_name in PREMODEL_MODEL_FIELDS:
        configured = str(vocatio_config.get(field_name, "") or "").strip()
        if field_name == "PREMODEL_NAME" or not configured:
            continue
        resolved[field_name] = configured
    return _normalize_resolved_premodel_config(resolved)
```

- [ ] **Step 4: Add provider-compatibility validation to the resolved VLM config**

```python
from scripts.pipeline.lib.vlm_transport import VlmRequest, validate_vlm_request


def validate_resolved_vlm_provider_config(config: Mapping[str, Any]) -> None:
    request = VlmRequest(
        provider=str(config["VLM_PROVIDER"]),
        base_url=str(config["VLM_BASE_URL"]),
        model=str(config["VLM_MODEL"]),
        messages=[{"role": "user", "content": "config validation"}],
        image_paths=[],
        timeout_seconds=float(config["VLM_TIMEOUT_SECONDS"]),
        temperature=float(config["VLM_TEMPERATURE"]),
        context_tokens=int(config["VLM_CONTEXT_TOKENS"]),
        max_output_tokens=int(config["VLM_MAX_OUTPUT_TOKENS"]),
        reasoning_level=str(config["VLM_REASONING_LEVEL"]),
        keep_alive=str(config["VLM_KEEP_ALIVE"]),
    )
    validate_vlm_request(request)
```

- [ ] **Step 5: Re-run loader tests**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
```

Expected:

- PASS with exact-name matching
- PASS with local override inheritance
- PASS with clear errors for missing/unknown names and provider-incompatible fields

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/test_manual_vlm_models.py
git commit -m "Add VLM and PREMODEL preset inheritance helpers"
```

### Task 3: Wire VLM CLI defaults through the inherited preset resolver

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write failing VLM inheritance tests**

```python
def test_apply_vocatio_defaults_uses_named_vlm_preset_and_override(self) -> None:
    day_dir = self.make_day_dir(
        ".vocatio",
        "\n".join(
            [
                "VLM_NAME=qwen3.5:9b",
                "VLM_TEMPERATURE=1.0",
                "VLM_WINDOW_RADIUS=4",
            ]
        ),
    )
    args = probe.parse_args([str(day_dir)])
    updated = probe.apply_vocatio_defaults(args, day_dir)
    self.assertEqual(updated.model, "qwen3.5:9b")
    self.assertEqual(updated.temperature, 1.0)
    self.assertEqual(updated.window_radius, 4)
```

- [ ] **Step 2: Run the probe tests to verify they fail**

Run:

```bash
python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:

- FAIL because `apply_vocatio_defaults()` still reads model fields directly from `.vocatio`

- [ ] **Step 3: Load `conf/vlm_models.yaml`, resolve the VLM preset, and apply only workflow fields locally**

```python
def apply_vocatio_defaults(args: argparse.Namespace, day_dir: Path) -> argparse.Namespace:
    config = load_vocatio_config(day_dir)
    preset_config = manual_vlm_models.resolve_vlm_model_config(
        manual_vlm_models.load_manual_vlm_models(REPO_ROOT / "conf" / "vlm_models.yaml").models,
        config,
    )
    args.provider = str(preset_config["VLM_PROVIDER"])
    args.model = str(preset_config["VLM_MODEL"])
    args.ollama_base_url = str(preset_config["VLM_BASE_URL"])
    args.ollama_num_ctx = int(preset_config["VLM_CONTEXT_TOKENS"])
    args.ollama_num_predict = int(preset_config["VLM_MAX_OUTPUT_TOKENS"])
    args.ollama_keep_alive = str(preset_config["VLM_KEEP_ALIVE"])
    args.timeout_seconds = float(preset_config["VLM_TIMEOUT_SECONDS"])
    args.temperature = float(preset_config["VLM_TEMPERATURE"])
    args.ollama_think = str(preset_config["VLM_REASONING_LEVEL"])
```

- [ ] **Step 4: Re-run the probe tests**

Run:

```bash
python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:

- PASS with inherited VLM model config
- PASS with workflow fields like `VLM_WINDOW_RADIUS` still coming from `.vocatio`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git commit -m "Resolve VLM model config from preset inheritance"
```

### Task 4: Wire PREMODEL CLI defaults through the inherited preset resolver

**Files:**
- Modify: `scripts/pipeline/build_photo_pre_model_annotations.py`
- Modify: `scripts/pipeline/test_build_photo_pre_model_annotations.py`

- [ ] **Step 1: Write failing PREMODEL inheritance tests**

```python
def test_apply_vocatio_defaults_uses_named_premodel_preset_and_override(self) -> None:
    day_dir = self.make_day_dir(
        ".vocatio",
        "\n".join(
            [
                "PREMODEL_NAME=qwen3.5-4b-pre",
                "PREMODEL_TEMPERATURE=0.2",
                "PREMODEL_IMAGE_COLUMN=thumb_path",
            ]
        ),
    )
    args = premodel.parse_args([str(day_dir)])
    updated = premodel.apply_vocatio_defaults(args, day_dir)
    self.assertEqual(updated.model_name, "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL")
    self.assertEqual(updated.temperature, 0.2)
    self.assertEqual(updated.image_column, "thumb_path")
```

- [ ] **Step 2: Run the PREMODEL tests to verify they fail**

Run:

```bash
python3 scripts/pipeline/test_build_photo_pre_model_annotations.py
```

Expected:

- FAIL because `apply_vocatio_defaults()` still expects raw `PREMODEL_*` model fields in `.vocatio`

- [ ] **Step 3: Resolve PREMODEL presets from `conf/vlm_models.yaml` and keep workflow settings separate**

```python
def apply_vocatio_defaults(args: argparse.Namespace, day_dir: Path) -> argparse.Namespace:
    config = load_vocatio_config(day_dir)
    preset_config = manual_vlm_models.resolve_premodel_model_config(
        manual_vlm_models.load_manual_vlm_models(REPO_ROOT / "conf" / "vlm_models.yaml").models,
        config,
    )
    args.provider = str(preset_config["PREMODEL_PROVIDER"])
    args.base_url = str(preset_config["PREMODEL_BASE_URL"])
    args.model_name = str(preset_config["PREMODEL_MODEL"])
    args.max_tokens = int(preset_config["PREMODEL_MAX_OUTPUT_TOKENS"])
    args.temperature = float(preset_config["PREMODEL_TEMPERATURE"])
    args.timeout_seconds = float(preset_config["PREMODEL_TIMEOUT_SECONDS"])
```

- [ ] **Step 4: Re-run the PREMODEL tests**

Run:

```bash
python3 scripts/pipeline/test_build_photo_pre_model_annotations.py
```

Expected:

- PASS with PREMODEL preset inheritance
- PASS with workflow fields like `PREMODEL_IMAGE_COLUMN` and `PREMODEL_OUTPUT_DIR` still using their existing local behavior

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/test_build_photo_pre_model_annotations.py
git commit -m "Resolve PREMODEL config from preset inheritance"
```

### Task 5: Keep manual GUI VLM selection working against the new shared preset file

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Modify: `scripts/pipeline/test_manual_vlm_models.py`

- [ ] **Step 1: Write failing GUI tests for VLM-only preset selection from the renamed file**

```python
def test_manual_vlm_gui_ignores_premodel_presets(self) -> None:
    window = self.build_main_window()
    window.manual_vlm_models = [
        {"VLM_NAME": "qwen3.5:9b"},
        {"PREMODEL_NAME": "qwen3.5-4b-pre"},
    ]
    review_gui.refresh_manual_vlm_model_choice_metadata(window)
    self.assertEqual(window.manual_vlm_choice_values, ["qwen3.5:9b"])
```

- [ ] **Step 2: Run the GUI diagnostics tests to verify they fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- FAIL because current manual GUI assumes every entry is a VLM preset

- [ ] **Step 3: Filter GUI-visible presets to VLM entries and keep the dropdown path on `conf/vlm_models.yaml`**

```python
def _is_vlm_model_entry(model: Mapping[str, Any]) -> bool:
    return bool(str(model.get("VLM_NAME", "") or "").strip())


def load_manual_vlm_models_for_gui(repo_root: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    config_path = repo_root / Path("conf/vlm_models.yaml")
    loaded = manual_vlm_models.load_manual_vlm_models(config_path)
    visible_models = [dict(model) for model in loaded.models if _is_vlm_model_entry(model)]
    return visible_models, loaded.md5_hex, None
```

- [ ] **Step 4: Re-run the GUI diagnostics tests**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:

- PASS with the renamed file path
- PASS with PREMODEL presets excluded from manual VLM dropdowns

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py scripts/pipeline/test_manual_vlm_models.py
git commit -m "Update manual VLM GUI to use shared preset file"
```

### Task 6: Run the focused regression pass and finish documentation

**Files:**
- Modify: `README.md`
- Verify: `scripts/pipeline/test_manual_vlm_models.py`
- Verify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- Verify: `scripts/pipeline/test_build_photo_pre_model_annotations.py`
- Verify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Update README examples to show `.vocatio` inheritance and the new filename**

```text
The code now reads model presets from `conf/vlm_models.yaml`.
`.vocatio` must provide exact `VLM_NAME` / `PREMODEL_NAME` values and may override only allowed model fields.
Workflow settings such as `VLM_WINDOW_RADIUS` and `PREMODEL_IMAGE_COLUMN` remain local `.vocatio` settings.
```

- [ ] **Step 2: Run the focused verification pass**

Run:

```bash
python3 scripts/pipeline/test_manual_vlm_models.py
python3 scripts/pipeline/test_probe_vlm_photo_boundaries.py
python3 scripts/pipeline/test_build_photo_pre_model_annotations.py
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
python3 -m py_compile scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/review_performance_proxy_gui.py
```

Expected:

- all four test files PASS
- `py_compile` exits successfully with no output

- [ ] **Step 3: Commit**

```bash
git add README.md scripts/pipeline/lib/manual_vlm_models.py scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/build_photo_pre_model_annotations.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_manual_vlm_models.py scripts/pipeline/test_probe_vlm_photo_boundaries.py scripts/pipeline/test_build_photo_pre_model_annotations.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Document VLM preset inheritance from .vocatio"
```

## Self-Review

- Spec coverage:
  - exact-name `VLM_NAME` / `PREMODEL_NAME` lookup: Tasks 2, 3, 4
  - no local-only model fallback: Tasks 2, 3, 4
  - provider incompatibility is an error: Task 2
  - workflow fields stay outside inheritance: Tasks 3 and 4
  - migrate `manual_vlm_models.yaml` references to `vlm_models.yaml`: Tasks 1, 5, 6
- Placeholder scan:
  - no `TODO` / `TBD`
  - every task has exact files, commands, and target behavior
- Type consistency:
  - shared helper names are `resolve_vlm_model_config(...)` and `resolve_premodel_model_config(...)`
  - YAML file path is consistently `conf/vlm_models.yaml`

