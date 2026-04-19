# Manual Background Workers And Selection Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make manual ML/VLM actions run in background workers without freezing the GUI, while simplifying the image-only selection summary and keeping finished results visible even after selection changes.

**Architecture:** Keep all changes inside `review_performance_proxy_gui.py` and its diagnostics test file. Introduce a small `QThreadPool`-backed worker and signal bridge for manual actions, a global in-flight lock for ML/VLM buttons, and a trimmed selection summary renderer. Preserve existing manual compute logic by moving reload, resolution, and compute into worker jobs.

**Tech Stack:** Python 3, PySide6 (`QThreadPool`, `QRunnable`, `QObject` signals), existing unittest-style GUI diagnostics script.

---

### Task 1: Simplify image-only selection summary rendering

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Add failing summary-render tests**

Add targeted tests asserting multi-photo summary:

```python
    def test_build_image_only_multi_photo_summary_body_uses_set_name_and_no_status_reason(self):
        body = review_gui.build_image_only_multi_photo_summary_body(
            [
                {
                    "adjusted_start_local": "2026-03-23T08:51:28.250",
                    "relative_path": "p-a7r5/20260323_085128_010_11935744.hif",
                    "display_name": "VLM0123",
                    "display_set_id": "vlm0123",
                    "assignment_status": "assigned",
                    "assignment_reason": "manual",
                },
                {
                    "adjusted_start_local": "2026-03-23T08:51:44.468",
                    "relative_path": "p-a7r5/20260323_085144_001_18194432.hif",
                    "display_name": "VLM0124",
                    "display_set_id": "vlm0124",
                    "assignment_status": "review",
                    "assignment_reason": "boundary",
                },
            ]
        )

        self.assertIn("Selected photos: 2", body)
        self.assertIn("| set=VLM0123", body)
        self.assertIn("| set=VLM0124", body)
        self.assertNotIn("First time:", body)
        self.assertNotIn("Last time:", body)
        self.assertNotIn("Selected photo rows", body)
        self.assertNotIn("status=", body)
        self.assertNotIn("reason=", body)
```

- [ ] **Step 2: Run GUI diagnostics and confirm failure**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- FAIL on old summary text.

- [ ] **Step 3: Implement trimmed summary body**

Update `build_image_only_multi_photo_summary_body(...)` to output:

```python
def build_image_only_multi_photo_summary_body(photos: Sequence[Mapping[str, Any]]) -> str:
    sorted_photos = sort_selected_photos(photos)
    lines = [f"Selected photos: {len(sorted_photos)}"]
    for photo in sorted_photos:
        display_name = str(photo.get("display_name", "") or photo.get("display_set_id", "") or "").strip()
        lines.append(
            " | ".join(
                [
                    format_value(photo.get("adjusted_start_local")),
                    format_value(photo.get("relative_path")),
                    f"set={format_value(display_name)}",
                ]
            )
        )
    return join_info_section_lines(lines)
```

- [ ] **Step 4: Re-run GUI diagnostics and confirm pass**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- PASS with new summary assertions green.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Simplify selection summary rendering"
```

### Task 2: Add background worker runtime for manual ML and VLM

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Add failing tests for async state and global lock**

Add tests covering:

```python
    def test_manual_actions_disable_both_buttons_when_ml_running(self):
        sections = review_gui.build_image_only_multi_photo_info_sections(
            photos=[{"relative_path": "a"}, {"relative_path": "b"}],
            diagnostics={},
            show_manual_ml_prediction=True,
            manual_prediction_state={"status": "running"},
            show_manual_vlm_analyze=True,
            manual_vlm_analyze_state={"status": "idle"},
        )
        ml_section = next(section for section in sections if section["key"] == "manual_ml_prediction")
        vlm_section = next(section for section in sections if section["key"] == "manual_vlm_analyze")

        self.assertEqual(ml_section["action_text"], "Run")
        self.assertFalse(ml_section["action_enabled"])
        self.assertTrue(ml_section["action_running"])
        self.assertEqual(vlm_section["action_text"], "Analyze")
        self.assertFalse(vlm_section["action_enabled"])
        self.assertFalse(vlm_section["action_running"])
```

and worker completion tests that prove:
- click handler sets `running` immediately
- heavy work is delegated to worker factory, not computed inline
- both buttons re-enable on result/error

- [ ] **Step 2: Run GUI diagnostics and confirm failure**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- FAIL because sections do not expose global-disable/running-indicator contract and handlers still compute inline.

- [ ] **Step 3: Add worker and signal bridge**

In `review_performance_proxy_gui.py`, add:

```python
class ManualActionSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, object)


class ManualActionWorker(QRunnable):
    def __init__(self, action_key: str, work_fn: Callable[[], Mapping[str, Any]], signals: ManualActionSignals):
        super().__init__()
        self.action_key = action_key
        self.work_fn = work_fn
        self.signals = signals

    def run(self) -> None:
        try:
            result = self.work_fn()
        except Exception as exc:
            self.signals.failed.emit(self.action_key, exc)
        else:
            self.signals.finished.emit(self.action_key, dict(result))
```

Add one global manual in-flight state on `MainWindow`, for example:

```python
self.manual_action_in_flight: Optional[str] = None
self.manual_action_signals = ManualActionSignals()
self.manual_action_signals.finished.connect(self.on_manual_action_finished)
self.manual_action_signals.failed.connect(self.on_manual_action_failed)
```

- [ ] **Step 4: Move manual ML/VLM heavy work behind worker entrypoints**

Refactor click handlers so they:
- bail if `manual_action_in_flight` is already set
- set `manual_action_in_flight` to `manual_ml` or `manual_vlm`
- set section state to `running` immediately
- call `refresh_current_info_dock()`
- queue a worker whose `work_fn` performs reload + resolution + compute

Pattern:

```python
def run_manual_ml_prediction(self) -> None:
    current_state = self.current_manual_ml_prediction_state()
    if not isinstance(current_state, Mapping) or self.manual_action_in_flight:
        return
    running_state = dict(current_state)
    running_state["status"] = "running"
    running_state["started_at"] = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    running_state.pop("error", None)
    running_state.pop("result_text", None)
    running_state.pop("resolution_error", None)
    self.manual_ml_prediction_state = running_state
    self.manual_action_in_flight = "manual_ml"
    self.refresh_current_info_dock()
    self.thread_pool.start(
        ManualActionWorker("manual_ml", self.build_manual_ml_work_fn(dict(current_state)), self.manual_action_signals)
    )
```

Inside `build_manual_ml_work_fn(...)` / `build_manual_vlm_work_fn(...)`, move:
- `reload_probe_vlm_boundary_module()`
- state resolution
- joined-row loading
- compute call

into the background thread.

- [ ] **Step 5: Update section rendering contract**

Change section builders so button state depends on the global in-flight action:

```python
section["action_text"] = "Run"
section["action_enabled"] = manual_action_in_flight is None
section["action_running"] = manual_action_in_flight == "manual_ml"
```

and analogously for VLM:

```python
section["action_text"] = "Analyze"
section["action_enabled"] = manual_action_in_flight is None
section["action_running"] = manual_action_in_flight == "manual_vlm"
```

Do not change the short base labels in running state.

- [ ] **Step 6: Render visible spinner without changing button label**

Extend `build_info_section_widget(...)` to honor `action_running` by adding a compact adjacent spinner label while keeping the button text unchanged:

```python
        if action_handler is not None:
            action_button = QPushButton(str(section.get("action_text", "") or "Run"))
            action_button.setEnabled(bool(section.get("action_enabled", True)))
            action_button.clicked.connect(action_handler)
            header_layout.addWidget(action_button, 0, Qt.AlignRight)
            if bool(section.get("action_running", False)):
                spinner_label = QLabel("⏳")
                spinner_label.setObjectName("infoSectionSpinner")
                header_layout.addWidget(spinner_label, 0, Qt.AlignRight)
```

- [ ] **Step 7: Re-enable buttons and keep results visible on completion**

Add main-thread slots:

```python
def on_manual_action_finished(self, action_key: str, result: object) -> None:
    self.manual_action_in_flight = None
    if action_key == "manual_ml":
        next_state = dict(self.manual_ml_prediction_state or {})
        next_state["status"] = "result"
        next_state.update(dict(result))
        self.manual_ml_prediction_state = next_state
    else:
        next_state = dict(self.manual_vlm_analyze_state or {})
        next_state["status"] = "result"
        next_state.update(dict(result))
        self.manual_vlm_analyze_state = next_state
    self.refresh_current_info_dock()
```

```python
def on_manual_action_failed(self, action_key: str, exc: object) -> None:
    self.manual_action_in_flight = None
    error_text = str(exc)
    if action_key == "manual_ml":
        next_state = dict(self.manual_ml_prediction_state or {})
        next_state["status"] = "error"
        next_state["error"] = error_text
        self.manual_ml_prediction_state = next_state
    else:
        next_state = dict(self.manual_vlm_analyze_state or {})
        next_state["status"] = "error"
        next_state["error"] = error_text
        if hasattr(exc, "debug_file_paths") and getattr(exc, "debug_file_paths"):
            next_state["debug_file_paths"] = [str(value) for value in getattr(exc, "debug_file_paths")]
        self.manual_vlm_analyze_state = next_state
    self.refresh_current_info_dock()
```

Do not clear finished results merely because current selection changed. Keep the last completed result state visible.

- [ ] **Step 8: Re-run GUI diagnostics and confirm pass**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- PASS
- manual summary, running-state, worker, and visibility tests green

- [ ] **Step 9: Commit**

```bash
git add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git commit -m "Run manual GUI analysis in background workers"
```

### Task 3: Final targeted regression

**Files:**
- Modify: none
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Run final GUI diagnostics**

Run:

```bash
env QT_QPA_PLATFORM=offscreen python3 scripts/pipeline/test_review_gui_image_only_diagnostics.py
```

Expected:
- PASS

- [ ] **Step 2: Verify working tree**

Run:

```bash
git status --short
```

Expected:
- no unexpected files beyond intentional changes

- [ ] **Step 3: Commit any final integration touch-ups**

```bash
git add -u
git commit -m "Finish manual background workers rollout"
```
