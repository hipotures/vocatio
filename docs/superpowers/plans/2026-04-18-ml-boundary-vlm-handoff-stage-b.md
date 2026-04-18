# ML Boundary VLM Handoff Stage B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw heuristic and pre-model prompt hints in `probe_vlm_photo_boundaries.py` with compact ML hints from the trained boundary and segment-type models.

**Architecture:** Reconstruct the same tabular feature row for each 5-frame VLM candidate window, load the trained AutoGluon predictors from the workspace ML corpus model directory, run local inference for the current window, and convert the two predictor outputs into short task-language hint lines. The prompt builder keeps the same image-first task instructions, but the old raw `gap_*` and `frame_*` structured hint blocks are removed and replaced with the ML hint block.

**Tech Stack:** Python 3, AutoGluon TabularPredictor, existing ML feature builders, existing VLM probe CLI/tests, rich console.

---

### Task 1: Add ML hint inference helpers

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] Build helpers that reconstruct a candidate-style row from the current 5-frame probe window using existing fields (`frame_01_relpath`, `frame_03_photo_id`, `frame_04_photo_id`, timestamps, `window_relative_paths`, etc.).
- [ ] Load descriptor annotations and heuristic pair rows for the current workspace using the same source files already available to the probe path.
- [ ] Reuse `build_candidate_feature_row(...)` to create the tabular feature vector instead of hand-encoding another schema.
- [ ] Add a helper that loads `boundary_model` and `segment_type_model` from a selected ML run directory and returns:
  - boundary prediction
  - boundary probability/confidence
  - segment type prediction
  - segment type probability/confidence
- [ ] Add tests that cover successful inference and missing-model fallback.

### Task 2: Add ML hint text rendering and prompt swap

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] Add a compact ML hint formatter that renders task-language lines such as:
  - `ML hint for this 5-frame window: likely cut after frame_03 (confidence 0.82).`
  - `ML hint for the likely segment after the boundary: ceremony (confidence 0.74).`
- [ ] Extend `build_user_prompt(...)` to accept ML hint lines.
- [ ] Remove direct prompt injection of:
  - `build_gap_hint_lines(...)`
  - `build_photo_pre_model_lines(...)`
- [ ] Add prompt instruction text explaining:
  - ML hint is advisory
  - ML aggregates heuristic gap signals and per-photo annotations
  - images take priority if they contradict the ML hint
- [ ] Add tests that assert the raw heuristic/pre-model blocks are gone and the ML hint block is present.

### Task 3: Wire model selection into probe CLI defaults

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Test: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] Add probe CLI/config support for selecting the ML run used for hint inference.
- [ ] Resolve a default ML model directory from workspace, using the existing ML corpus structure and a configurable run id.
- [ ] Ensure missing model artifacts degrade gracefully: no crash, prompt says ML hints are unavailable.
- [ ] Add tests for CLI parsing, `.vocatio` defaults, and missing-artifact fallback.

### Task 4: Verify end-to-end prompt generation behavior

**Files:**
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] Add a focused probe test that runs one candidate window with fake models and asserts:
  - images are still attached
  - ML hints are included
  - old raw hint blocks are absent
  - response parsing/output rows still work as before
- [ ] Run the focused test file and fix any regressions.

### Task 5: Final verification and commit

**Files:**
- Modify: `scripts/pipeline/probe_vlm_photo_boundaries.py`
- Modify: `scripts/pipeline/test_probe_vlm_photo_boundaries.py`

- [ ] Run:
  - `uv run pytest /home/xai/DEV/vocatio/.worktrees/ml-boundary-verifier/scripts/pipeline/test_probe_vlm_photo_boundaries.py -q`
- [ ] Run a syntax check if needed on the touched probe file.
- [ ] Commit with one Stage B-focused message after tests are green.
