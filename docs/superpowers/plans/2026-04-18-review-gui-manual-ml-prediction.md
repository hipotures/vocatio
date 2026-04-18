# Review GUI Manual ML Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sectioned, copy-friendly info panel to the review GUI and a manual `ML Prediction` action for exactly two selected photos, without changing the existing index or review-state formats.

**Architecture:** Keep the existing GUI startup/data-loading flow intact and replace only the info-dock presentation layer. Build a small section widget model in the GUI, render current diagnostics into named sections, and add one synchronous manual ML prediction action that computes from two selected anchor photos using `.vocatio`/existing defaults. The prediction result lives only in memory and is shown inline in the info panel.

**Tech Stack:** Python 3, PySide6, existing `review_performance_proxy_gui.py` GUI logic, existing `probe_vlm_photo_boundaries.py` ML/VLM helper functions, `unittest`-style script tests.

---

## File Map

- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - Replace single-text info dock rendering with section widgets.
  - Add clipboard-copy section headers.
  - Add manual ML prediction action and temporary result state.
  - Resolve `.vocatio`/default parameters at runtime using existing config helpers.
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
  - Add section-model tests, copy-action tests, conditional manual-prediction section tests, and manual-gap prediction tests.
- Reference only: `scripts/pipeline/probe_vlm_photo_boundaries.py`
  - Reuse existing ML hint context loading, config defaults, and prediction helpers; no planned code changes here.
- Reference only: `scripts/pipeline/lib/workspace_dir.py`
  - Use existing `.vocatio` loading semantics; no planned code changes here.

### Task 1: Add failing tests for sectioned info panels and copyable headers

**Files:**
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add focused tests for a section-based info model, for example:

```python
def test_build_info_sections_for_set_returns_named_sections(self):
    diagnostics = {"available": False, "error": ""}
    display_set = {
        "display_name": "VLM0001",
        "original_performance_number": "VLM0001",
        "set_id": "vlm-set-0001",
        "base_set_id": "vlm-set-0001",
        "type_code": "D",
        "type_override_active": False,
        "timeline_status": "vlm_probe:1_hits",
        "photo_count": 5,
        "review_count": 0,
        "duration_seconds": 10,
        "max_internal_photo_gap_seconds": 3,
        "performance_start_local": "2026-03-23T10:00:00",
        "performance_end_local": "2026-03-23T10:00:10",
        "first_photo_local": "2026-03-23T10:00:00",
        "last_photo_local": "2026-03-23T10:00:10",
        "merged_manually": False,
        "photos": [],
    }
    sections = review_gui.build_image_only_set_info_sections(
        display_set,
        diagnostics,
        no_photos_confirmed=False,
        show_manual_ml_prediction=False,
        manual_prediction_state=None,
    )
    self.assertEqual([section["title"] for section in sections], ["Set summary"])
    self.assertIn("Type: D", sections[0]["body"])
```

```python
def test_build_copy_status_message_uses_section_title(self):
    self.assertEqual(
        review_gui.build_info_section_copy_status_message("ML hints"),
        "Copied ML hints",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL with missing section helpers or assertion mismatches.

- [ ] **Step 3: Write minimal implementation**

Add minimal section-model helpers in `scripts/pipeline/review_performance_proxy_gui.py`, shaped like:

```python
def build_info_section(title: str, description: str, body: str, *, key: str = "") -> Dict[str, str]:
    return {
        "key": key or title.strip().lower().replace(" ", "_"),
        "title": title,
        "description": description,
        "body": body,
    }


def build_info_section_copy_status_message(title: str) -> str:
    return f"Copied {str(title or '').strip()}"
```

And add minimal section builders:

```python
def build_image_only_set_info_sections(...):
    return [
        build_info_section(
            "Set summary",
            "Basic set metadata and review state.",
            build_image_only_set_info_text(...),
        )
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: the new section-model assertions pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add review GUI info section helpers"
```

### Task 2: Replace the info dock text label with copyable section widgets

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add tests for the section selection logic:

```python
def test_build_image_only_multi_photo_info_sections_omits_manual_ml_section_without_two_photos(self):
    diagnostics = {"available": False, "error": ""}
    photos = [{"filename": "a.jpg"}, {"filename": "b.jpg"}, {"filename": "c.jpg"}]
    sections = review_gui.build_image_only_multi_photo_info_sections(
        photos,
        diagnostics,
        show_manual_ml_prediction=False,
        manual_prediction_state=None,
    )
    self.assertEqual([section["title"] for section in sections], ["Selection summary"])
```

```python
def test_build_image_only_photo_info_sections_include_boundary_and_ml_sections(self):
    diagnostics = {"available": False, "error": ""}
    photo = {
        "filename": "b.jpg",
        "relative_path": "cam/b.jpg",
        "type_code": "D",
        "type_override_active": False,
        "assignment_status": "assigned",
        "assignment_reason": "",
        "seconds_to_nearest_boundary": "0.0",
        "stream_id": "p-main",
        "device": "A7R5",
        "adjusted_start_local": "2026-03-23T10:00:05",
        "proxy_exists": True,
    }
    sections = review_gui.build_image_only_photo_info_sections(photo, diagnostics)
    self.assertEqual(
        [section["title"] for section in sections],
        ["Photo summary", "Boundary diagnostics", "ML hints"],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL due to missing section builders or missing conditional section logic.

- [ ] **Step 3: Write minimal implementation**

In `scripts/pipeline/review_performance_proxy_gui.py`:

- replace `self.meta_label = QLabel("")` with a small read-only panel container, for example a `QScrollArea` + inner `QWidget` + `QVBoxLayout`,
- add a tiny section widget helper such as:

```python
def build_info_section_widget(
    self,
    title: str,
    description: str,
    body: str,
) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout()
    header_button = QPushButton(title)
    header_button.clicked.connect(lambda: self.copy_info_section_body(title, body))
    description_label = QLabel(description)
    body_view = QPlainTextEdit()
    body_view.setReadOnly(True)
    body_view.setPlainText(body)
    layout.addWidget(header_button)
    layout.addWidget(description_label)
    layout.addWidget(body_view)
    container.setLayout(layout)
    return container
```

- add:

```python
def copy_info_section_body(self, title: str, body: str) -> None:
    QApplication.clipboard().setText(body)
    self.statusBar().showMessage(build_info_section_copy_status_message(title))
```

- add a renderer:

```python
def render_info_sections(self, sections: Sequence[Mapping[str, str]]) -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/review_performance_proxy_gui.py --help
```

Expected:
- script tests PASS
- GUI help still renders normally

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: render copyable info sections in review GUI"
```

### Task 3: Add runtime manual ML prediction state and two-photo eligibility logic

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add tests for eligibility and ephemeral state:

```python
def test_should_show_manual_ml_prediction_requires_exactly_two_selected_photos(self):
    self.assertFalse(review_gui.should_show_manual_ml_prediction([]))
    self.assertFalse(review_gui.should_show_manual_ml_prediction([{"filename": "a.jpg"}]))
    self.assertTrue(
        review_gui.should_show_manual_ml_prediction(
            [{"filename": "a.jpg"}, {"filename": "b.jpg"}]
        )
    )
    self.assertFalse(
        review_gui.should_show_manual_ml_prediction(
            [{"filename": "a.jpg"}, {"filename": "b.jpg"}, {"filename": "c.jpg"}]
        )
    )
```

```python
def test_build_manual_ml_prediction_section_renders_busy_state(self):
    section = review_gui.build_manual_ml_prediction_section(
        manual_prediction_state={"status": "running", "result_text": "", "error_text": ""},
    )
    self.assertEqual(section["title"], "Manual ML Prediction")
    self.assertIn("Computing...", section["description"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL with missing helper/state functions.

- [ ] **Step 3: Write minimal implementation**

Add in `scripts/pipeline/review_performance_proxy_gui.py`:

```python
def should_show_manual_ml_prediction(selected_photos: Sequence[Mapping[str, Any]]) -> bool:
    return len(selected_photos) == 2
```

And add ephemeral window state fields in `MainWindow.__init__`:

```python
self.manual_ml_prediction_state = {
    "status": "idle",
    "result_text": "",
    "error_text": "",
    "selection_signature": "",
}
```

Then add a section helper:

```python
def build_manual_ml_prediction_section(manual_prediction_state: Mapping[str, str]) -> Dict[str, str]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: PASS for the new manual-section state tests.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add manual ML prediction section state"
```

### Task 4: Resolve manual-gap anchors and prediction parameters from existing config

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Reference: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Reference: `scripts/pipeline/lib/workspace_dir.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add tests for parameter resolution and anchor semantics:

```python
def test_resolve_manual_prediction_window_config_prefers_vocatio_values(self):
    config = {"VLM_WINDOW_SIZE": "7", "VLM_OVERLAP": "3"}
    resolved = review_gui.resolve_manual_prediction_window_config(config)
    self.assertEqual(resolved, {"window_size": 7, "overlap": 3})
```

```python
def test_resolve_manual_prediction_window_config_falls_back_to_existing_defaults(self):
    resolved = review_gui.resolve_manual_prediction_window_config({})
    self.assertEqual(resolved, {"window_size": review_gui.probe_vlm_boundary.DEFAULT_WINDOW_SIZE, "overlap": review_gui.probe_vlm_boundary.DEFAULT_OVERLAP})
```

```python
def test_build_manual_anchor_pair_sorts_selected_photos_and_ignores_interior_rows(self):
    joined_rows = [
        {"relative_path": "a", "start_epoch_ms": "1000"},
        {"relative_path": "b", "start_epoch_ms": "2000"},
        {"relative_path": "c", "start_epoch_ms": "3000"},
        {"relative_path": "d", "start_epoch_ms": "9000"},
    ]
    selected = [
        {"relative_path": "d", "source_path": "/d"},
        {"relative_path": "a", "source_path": "/a"},
    ]
    result = review_gui.resolve_manual_prediction_anchor_pair(selected, joined_rows)
    self.assertEqual(result["left_relative_path"], "a")
    self.assertEqual(result["right_relative_path"], "d")
    self.assertEqual(result["gap_seconds"], 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL due to missing config/anchor resolution helpers.

- [ ] **Step 3: Write minimal implementation**

In `scripts/pipeline/review_performance_proxy_gui.py`:

- load `.vocatio` through existing helper when needed,
- add:

```python
def resolve_manual_prediction_window_config(vocatio_config: Mapping[str, str]) -> Dict[str, int]:
    ...
```

- add:

```python
def resolve_manual_prediction_anchor_pair(
    selected_photos: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    ...
```

Use the selected photos only as left/right anchors. Sort by day order and compute the direct anchor gap in seconds. Ignore interior rows as part of the gap itself.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: PASS for config and anchor-resolution tests.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: resolve manual ML prediction anchors and config"
```

### Task 5: Run manual ML prediction from the GUI and render the result inline

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Reference: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add tests for the click path and result rendering:

```python
def test_run_manual_ml_prediction_updates_ephemeral_result_text(self):
    window = review_gui.MainWindow.__new__(review_gui.MainWindow)
    window.manual_ml_prediction_state = {"status": "idle", "result_text": "", "error_text": "", "selection_signature": ""}
    window.statusBar = lambda: MagicMock()
    window.selected_photo_entries = MagicMock(return_value=[{"relative_path": "a"}, {"relative_path": "d"}])
    window.run_manual_ml_prediction = review_gui.MainWindow.run_manual_ml_prediction.__get__(window, review_gui.MainWindow)
```

More concrete assertion target:

```python
def test_format_manual_ml_prediction_result_text(self):
    text = review_gui.format_manual_ml_prediction_result_text(
        {
            "boundary_prediction": True,
            "boundary_confidence": 0.84,
            "segment_type_prediction": "dance",
            "segment_type_confidence": 0.93,
            "gap_seconds": 128,
            "left_relative_path": "cam/a.jpg",
            "right_relative_path": "cam/d.jpg",
        }
    )
    self.assertIn("Boundary: cut (0.84)", text)
    self.assertIn("Right-side segment: dance (0.93)", text)
    self.assertIn("Gap seconds: 128", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected: FAIL because the manual prediction action/rendering path is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

In `scripts/pipeline/review_performance_proxy_gui.py`:

- add a section widget path that can optionally render a button,
- add:

```python
def format_manual_ml_prediction_result_text(result: Mapping[str, Any]) -> str:
    ...
```

- add `MainWindow.run_manual_ml_prediction()` that:
  - verifies exactly two selected photos,
  - sets manual state to running,
  - disables the button and updates the panel,
  - loads ML hint context from existing helpers,
  - builds the manual anchor gap and prediction request,
  - computes the result synchronously,
  - stores only the in-memory result text/error text,
  - re-renders the info panel.

Use the existing prediction helpers from `probe_vlm_photo_boundaries.py` rather than introducing a new scoring path.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/review_performance_proxy_gui.py --help
```

Expected:
- script tests PASS
- GUI help still works

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add manual ML prediction to review GUI"
```

## Self-Review

- Spec coverage:
  - sectioned info panel: Tasks 1-2
  - clickable copyable section headers with status-bar feedback: Tasks 1-2
  - `Manual ML Prediction` only for exactly two selected photos: Task 3
  - runtime-only prediction with no history/review-state writes: Tasks 3 and 5
  - `.vocatio` then existing defaults for `VLM_WINDOW_SIZE` / `VLM_OVERLAP`: Task 4
  - non-consecutive anchor photos with ignored interior photos: Task 4
  - disabled/busy prediction button and inline result: Task 5
  - copyable panel content replacing non-selectable single text block: Task 2
- Placeholder scan:
  - no `TODO` / `TBD`
  - every code-changing task includes concrete code or function signatures
- Type consistency:
  - `manual_ml_prediction_state`, `selected_photo_keys_by_set`, `resolve_manual_prediction_window_config`, `resolve_manual_prediction_anchor_pair`, and `format_manual_ml_prediction_result_text` are used consistently across tasks
