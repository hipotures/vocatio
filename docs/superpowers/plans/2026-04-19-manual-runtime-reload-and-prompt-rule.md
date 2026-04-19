# Manual Runtime Reload And Prompt Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make manual ML and manual VLM actions reload the shared probe module at click time, and strengthen the VLM prompt so background or lighting changes never justify either a boundary or a segment-type change.

**Architecture:** Keep the existing GUI wrapper over `probe_vlm_photo_boundaries.py`, but add one small reload seam in `review_performance_proxy_gui.py` that refreshes the imported probe module immediately before manual inference runs. Update only the forbidden-evidence wording in `build_user_prompt(...)`, then cover both changes with focused regression tests in the existing GUI and probe test files.

**Tech Stack:** Python 3, PySide6 GUI, `importlib.reload`, pytest/unittest script-style tests.

---

### Task 1: Add a safe probe-module reload seam in the GUI

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing GUI tests for reload-on-click**

Add tests near the existing manual action tests in `scripts/pipeline/test_review_gui_image_only_diagnostics.py`:

```python
    def test_run_manual_ml_prediction_reloads_probe_module_before_inference(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.manual_ml_prediction_state = {
            "status": "idle",
            "left_relative_path": "cam/a.jpg",
            "right_relative_path": "cam/b.jpg",
        }
        window.current_manual_ml_prediction_state = Mock(return_value=window.manual_ml_prediction_state)
        window.selected_photo_entries = Mock(return_value=[{"relative_path": "cam/a.jpg"}, {"relative_path": "cam/b.jpg"}])
        window.refresh_info_panel = Mock()
        window.statusBar = Mock(return_value=Mock(showMessage=Mock()))
        window.day_dir = Path("/day")
        window.workspace_dir = Path("/workspace")
        window.index_payload = {"day": "20260323"}
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.joined_rows = []

        with patch.object(review_gui, "reload_probe_vlm_boundary_module") as reload_mock, \
             patch.object(review_gui, "resolve_manual_prediction_state", return_value={"status": "running"}), \
             patch.object(review_gui, "compute_manual_ml_prediction_result", return_value={"status": "result", "result_text": "ok"}):
            window.run_manual_ml_prediction = review_gui.MainWindow.run_manual_ml_prediction.__get__(window, review_gui.MainWindow)
            window.run_manual_ml_prediction()

        reload_mock.assert_called_once_with()

    def test_run_manual_vlm_analyze_reloads_probe_module_before_analysis(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.manual_vlm_analyze_state = {
            "status": "idle",
            "left_relative_path": "cam/a.jpg",
            "right_relative_path": "cam/b.jpg",
        }
        window.current_manual_vlm_analyze_state = Mock(return_value=window.manual_vlm_analyze_state)
        window.selected_photo_entries = Mock(return_value=[{"relative_path": "cam/a.jpg"}, {"relative_path": "cam/b.jpg"}])
        window.refresh_info_panel = Mock()
        window.statusBar = Mock(return_value=Mock(showMessage=Mock()))
        window.day_dir = Path("/day")
        window.workspace_dir = Path("/workspace")
        window.index_payload = {"day": "20260323"}
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.joined_rows = []

        with patch.object(review_gui, "reload_probe_vlm_boundary_module") as reload_mock, \
             patch.object(review_gui, "resolve_manual_vlm_analyze_state", return_value={"status": "running"}), \
             patch.object(review_gui, "compute_manual_vlm_analyze_result", return_value={"status": "result", "result_text": "ok"}):
            window.run_manual_vlm_analyze = review_gui.MainWindow.run_manual_vlm_analyze.__get__(window, review_gui.MainWindow)
            window.run_manual_vlm_analyze()

        reload_mock.assert_called_once_with()
```

- [ ] **Step 2: Run the GUI diagnostics test file and verify the new tests fail**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- FAIL with missing `reload_probe_vlm_boundary_module` assertions or `AttributeError` because no reload seam exists yet.

- [ ] **Step 3: Implement the reload seam in the GUI module**

Add a dedicated helper in `scripts/pipeline/review_performance_proxy_gui.py` near the probe import:

```python
import importlib
```

```python
def reload_probe_vlm_boundary_module():
    global probe_vlm_boundary
    probe_vlm_boundary = importlib.reload(probe_vlm_boundary)
    return probe_vlm_boundary
```

Then call it at the top of both manual actions:

```python
    def run_manual_ml_prediction(self) -> None:
        current_state = self.current_manual_ml_prediction_state()
        if not current_state:
            return
        try:
            reload_probe_vlm_boundary_module()
        except Exception as exc:
            self.manual_ml_prediction_state = {
                **dict(current_state),
                "status": "error",
                "error": f"module reload failed: {exc}",
            }
            self.refresh_info_panel()
            return
```

```python
    def run_manual_vlm_analyze(self) -> None:
        current_state = self.current_manual_vlm_analyze_state()
        if not current_state:
            return
        try:
            reload_probe_vlm_boundary_module()
        except Exception as exc:
            self.manual_vlm_analyze_state = {
                **dict(current_state),
                "status": "error",
                "error": f"module reload failed: {exc}",
            }
            self.refresh_info_panel()
            return
```

Keep the rest of the existing logic unchanged.

- [ ] **Step 4: Run the GUI diagnostics test file and verify it passes**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- PASS
- existing manual ML/VLM action tests still green
- the two new reload tests green

- [ ] **Step 5: Commit the GUI reload seam**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Reload probe module for manual GUI actions"
```

### Task 2: Strengthen the prompt rule for background and lighting evidence

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] **Step 1: Write the failing prompt regression test**

Add a focused test near the prompt-builder tests in `scripts/pipeline/test_probe_vlm_photo_boundaries.py`:

```python
    def test_build_user_prompt_forbids_background_and_lighting_for_boundary_and_segment_change(self):
        prompt = probe.build_user_prompt(
            window_size=6,
            ml_hint_lines=[],
            response_schema_mode="on",
        )

        self.assertIn(
            "Background change and lighting change, whether alone or together, never justify a boundary.",
            prompt,
        )
        self.assertIn(
            "Background change and lighting change, whether alone or together, never justify a segment-type change.",
            prompt,
        )
```

- [ ] **Step 2: Run the probe test file and verify the new test fails**

Run:

```bash
uv run pytest scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:
- FAIL because the prompt still contains only the weaker `lighting change alone` and `background change alone` wording.

- [ ] **Step 3: Update only the prompt rule text**

In `scripts/pipeline/probe_vlm_photo_boundaries.py`, keep the rest of `build_user_prompt(...)` intact and replace the relevant forbidden-evidence block with explicit combined wording:

```python
        "Forbidden reasons for a boundary:\n"
        "- pose change\n"
        "- motion change\n"
        "- choreography phrase change\n"
        "- a new movement phrase inside the same act\n"
        "- framing change\n"
        "- background change alone\n"
        "- lighting change alone\n"
        "- background change and lighting change together\n"
```

and add an explicit rule later in the prompt:

```python
        "- Background change and lighting change, whether alone or together, never justify a boundary.\n"
        "- Background change and lighting change, whether alone or together, never justify a segment-type change.\n"
```

Do not change any other task wording in this task.

- [ ] **Step 4: Run the probe test file and verify it passes**

Run:

```bash
uv run pytest scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:
- PASS
- new prompt wording test green
- existing manual-window and manual-VLM tests still green

- [ ] **Step 5: Commit the prompt rule change**

```bash
git add scripts/pipeline/probe_vlm_photo_boundaries.py scripts/pipeline/test_probe_vlm_photo_boundaries.py
git commit -m "Strengthen VLM prompt boundary evidence rules"
```

### Task 3: Run end-to-end regression for the touched surfaces

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Run the probe regression**

Run:

```bash
uv run pytest scripts/pipeline/test_probe_vlm_photo_boundaries.py
```

Expected:
- PASS

- [ ] **Step 2: Run the GUI regression**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- PASS

- [ ] **Step 3: Verify the working tree only contains intentional changes**

Run:

```bash
git status --short
```

Expected:
- no unexpected modified files
- pre-existing unrelated untracked entries, if any, are unchanged

- [ ] **Step 4: Commit any remaining integration adjustments**

```bash
git add -u
git commit -m "Finish manual prompt reload experiment support"
```
