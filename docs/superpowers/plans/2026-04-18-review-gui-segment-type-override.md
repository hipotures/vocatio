# Review GUI Segment Type Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a keyboard-driven segment type override to the review GUI so the current set can cycle through `? / D / C / A / R / O`, persist that choice in `review_state.json`, and render overridden rows in italic.

**Architecture:** Extend the existing per-set review-state entry with a `segment_type_override` field, resolve an effective type for each display set/photo row during tree rebuild, and wire a new `Y` shortcut that cycles overrides and immediately rebuilds the tree. Reuse existing save/rebuild/status-bar patterns so the feature behaves like split/merge/no-photos actions instead of introducing a separate settings flow.

**Tech Stack:** Python 3, PySide6, existing JSON review-state persistence, `unittest`-style script tests.

---

### Task 1: Add failing tests for override state and keyboard semantics

**Files:**
- Modify: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`
- Create or Modify: `scripts/pipeline/test_export_selected_photos_json.py` only if a second lightweight GUI-state test is needed
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- cycling helper maps `"" -> dance -> ceremony -> audience -> rehearsal -> other -> ""`
- effective type code uses override when present
- set/photo info text includes `Type override: yes`
- italic styling helper is activated for overridden rows without requiring bold

Use concrete assertions like:

```python
def test_cycle_segment_type_override_wraps_back_to_empty(self):
    self.assertEqual(review_gui.next_segment_type_override(""), "dance")
    self.assertEqual(review_gui.next_segment_type_override("dance"), "ceremony")
    self.assertEqual(review_gui.next_segment_type_override("ceremony"), "audience")
    self.assertEqual(review_gui.next_segment_type_override("audience"), "rehearsal")
    self.assertEqual(review_gui.next_segment_type_override("rehearsal"), "other")
    self.assertEqual(review_gui.next_segment_type_override("other"), "")
```

```python
def test_build_image_only_set_info_text_reports_type_override(self):
    diagnostics = {"available": False, "error": ""}
    display_set = {
        "display_name": "VLM0001",
        "original_performance_number": "VLM0001",
        "set_id": "vlm-set-0001",
        "base_set_id": "vlm-set-0001",
        "duplicate_status": "normal",
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
        "type_code": "C",
        "type_override_active": True,
        "photos": [],
    }
    text = review_gui.build_image_only_set_info_text(display_set, diagnostics, no_photos_confirmed=False)
    self.assertIn("Type: C", text)
    self.assertIn("Type override: yes", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: FAIL with missing helper/function/assertion mismatch around override behavior.

- [ ] **Step 3: Write minimal implementation helpers**

Add the minimal helpers and info-text plumbing in `scripts/pipeline/review_performance_proxy_gui.py`:
- `SEGMENT_TYPE_OVERRIDE_CYCLE`
- `next_segment_type_override(current: str) -> str`
- `resolve_effective_segment_type(base_type: str, override_type: str) -> tuple[str, bool]`

Use code shaped like:

```python
SEGMENT_TYPE_OVERRIDE_CYCLE = ("", "dance", "ceremony", "audience", "rehearsal", "other")


def next_segment_type_override(current: str) -> str:
    normalized = str(current or "").strip().lower()
    try:
        index = SEGMENT_TYPE_OVERRIDE_CYCLE.index(normalized)
    except ValueError:
        index = 0
    return SEGMENT_TYPE_OVERRIDE_CYCLE[(index + 1) % len(SEGMENT_TYPE_OVERRIDE_CYCLE)]


def resolve_effective_segment_type(base_type: str, override_type: str) -> tuple[str, bool]:
    normalized_override = str(override_type or "").strip().lower()
    if normalized_override:
        return normalized_override, True
    return str(base_type or "").strip().lower(), False
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: PASS for the new override-focused assertions.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add review GUI type override helpers"
```

### Task 2: Persist override in review state and resolve effective type in the tree

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing test for persisted override resolution**

Add a test that builds a minimal `MainWindow`-like state or helper-level payload and verifies:
- `segment_type_override` from `review_state["performances"][set_id]` overrides base `segment_type`
- effective `type_code` is propagated to child photo rows

Use a helper-level assertion if possible, for example:

```python
def test_apply_segment_type_override_to_display_set(self):
    display_set = {"set_id": "vlm-set-0001", "segment_type": "dance", "type_code": "D", "photos": [{"filename": "a.hif"}]}
    entry = {"segment_type_override": "ceremony"}
    review_gui.apply_segment_type_override_to_display_set(display_set, entry)
    self.assertEqual(display_set["segment_type"], "ceremony")
    self.assertEqual(display_set["type_code"], "C")
    self.assertTrue(display_set["type_override_active"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: FAIL because override application is not implemented yet.

- [ ] **Step 3: Implement override persistence and effective-type propagation**

Modify `scripts/pipeline/review_performance_proxy_gui.py` to:
- add `segment_type_override` to the default review entry
- ensure `review_entry()` always initializes it
- add a helper like `apply_segment_type_override_to_display_set(display_set, entry)`
- call it during `rebuild_display_sets()` after base type derivation and before child rows are finalized
- propagate:
  - `segment_type`
  - `type_code`
  - `type_override_active`
  to both set rows and child photo rows

Use code shaped like:

```python
def apply_segment_type_override_to_display_set(display_set: Dict[str, Any], entry: Mapping[str, Any]) -> None:
    effective_segment_type, override_active = resolve_effective_segment_type(
        str(display_set.get("segment_type", "") or ""),
        str(entry.get("segment_type_override", "") or ""),
    )
    display_set["segment_type"] = effective_segment_type
    display_set["type_code"] = segment_type_to_code(effective_segment_type)
    display_set["type_override_active"] = override_active
    for photo in display_set.get("photos", []):
        photo["segment_type"] = effective_segment_type
        photo["type_code"] = display_set["type_code"]
        photo["type_override_active"] = override_active
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: PASS for override propagation tests.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: persist review GUI type overrides"
```

### Task 3: Add `Y` shortcut and status-bar feedback

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing test for cycling override state**

Add a test for the action-level helper that updates the current review entry and returns the right status message:

```python
def test_build_segment_type_override_status_message(self):
    self.assertEqual(
        review_gui.build_segment_type_override_status_message("VLM0007", "dance", override_active=True),
        "Type set to D for set VLM0007",
    )
    self.assertEqual(
        review_gui.build_segment_type_override_status_message("VLM0007", "", override_active=False),
        "Type reset for set VLM0007",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: FAIL because the status helper and action are missing.

- [ ] **Step 3: Implement `Y` shortcut and cycle action**

Modify `scripts/pipeline/review_performance_proxy_gui.py` to:
- add a `QAction` on `Y`
- connect it to `cycle_current_set_segment_type_override()`
- in that method:
  - get `current_top_level_item()`
  - update `review_entry(set_id)["segment_type_override"]`
  - set `updated_at`
  - mark state dirty
  - rebuild tree with `preferred_set_id`
  - flush state
  - show a status-bar message

Use code shaped like:

```python
type_override_action = QAction(self)
type_override_action.setShortcut(QKeySequence("Y"))
type_override_action.triggered.connect(self.cycle_current_set_segment_type_override)
self.addAction(type_override_action)
```

```python
def cycle_current_set_segment_type_override(self) -> None:
    item = self.current_top_level_item()
    if item is None:
        return
    display_set = item.data(0, Qt.UserRole)
    entry = self.review_entry(display_set["set_id"])
    next_override = next_segment_type_override(entry.get("segment_type_override", ""))
    entry["segment_type_override"] = next_override
    self.review_state["updated_at"] = self.current_timestamp()
    self.state_dirty = True
    self.rebuild_tree_after_state_change(preferred_set_id=display_set["set_id"])
    override_active = bool(next_override)
    message = build_segment_type_override_status_message(display_set["display_name"], next_override, override_active)
    if self.flush_review_state():
        self.statusBar().showMessage(message)
    else:
        self.statusBar().showMessage(f"{message} in memory, but save failed")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: PASS for shortcut/status helper coverage.

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: add review GUI type override shortcut"
```

### Task 4: Render override visually with italic rows and update info panel

**Files:**
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
- Test: `scripts/pipeline/test_review_gui_image_only_diagnostics.py`

- [ ] **Step 1: Write the failing test for info-panel override indicator**

Add tests that assert:
- set info text contains `Type override: yes/no`
- photo info text contains `Type override: yes/no`

Use assertions like:

```python
def test_build_image_only_photo_info_text_reports_type_override(self):
    diagnostics = {"available": False, "error": ""}
    photo = {
        "display_name": "VLM0001",
        "original_performance_number": "VLM0001",
        "base_set_id": "vlm-set-0001",
        "type_code": "A",
        "type_override_active": True,
        "relative_path": "cam/b.jpg",
        "filename": "b.jpg",
        "adjusted_start_local": "2026-03-23T10:00:05",
        "assignment_status": "assigned",
        "assignment_reason": "",
        "seconds_to_nearest_boundary": "0.000000",
        "stream_id": "p-main",
        "device": "A7R5",
        "proxy_exists": True,
    }
    text = review_gui.build_image_only_photo_info_text(photo, diagnostics)
    self.assertIn("Type: A", text)
    self.assertIn("Type override: yes", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
```

Expected: FAIL because override marker text and italic rendering are incomplete.

- [ ] **Step 3: Implement italic styling and info-panel labels**

Modify `scripts/pipeline/review_performance_proxy_gui.py` to:
- update `apply_review_font()` so `type_override_active` sets `font.setItalic(True)`
- keep existing viewed/unviewed logic intact
- do not introduce bold specifically for type override
- add `Type override: yes/no` lines to both set and photo info builders

Use code shaped like:

```python
font = QFont(QApplication.font())
font.setBold(not is_viewed)
font.setWeight(QFont.Normal if is_viewed else QFont.Bold)
font.setItalic(bool(display_set.get("type_override_active")))
```

And in info builders:

```python
f"Type override: {'yes' if display_set.get('type_override_active') else 'no'}",
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_review_gui_image_only_diagnostics.py -q
python3 /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/review_performance_proxy_gui.py --help
```

Expected:
- pytest PASS
- GUI help output still renders normally

- [ ] **Step 5: Commit**

```bash
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier add scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_review_gui_image_only_diagnostics.py
git -C /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier commit -m "feat: show review GUI type overrides in tree and info"
```

## Self-Review

- Spec coverage:
  - `Y` shortcut: Task 3
  - cycle `? / D / C / A / R / O`: Tasks 1 and 3
  - persisted override in `review_state`: Task 2
  - effective type in table/info: Tasks 2 and 4
  - italic, not bold, for override: Task 4
- Placeholder scan:
  - no `TODO` / `TBD`
  - each code-changing task contains concrete code blocks
- Type consistency:
  - `segment_type_override`, `type_override_active`, `next_segment_type_override`, and `resolve_effective_segment_type` are used consistently across tasks
