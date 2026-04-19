## Goal

Stage B integrates the trained ML boundary verifier outputs into the main visual boundary VLM prompt.

This stage starts **only after** Stage A is implemented, the user has run real ML experiments, and the user confirms that the richer ML models are good enough to become the structured hint layer for the VLM.

The purpose of this document is to preserve the design intent before Stage A implementation fills the working context.

## Why The Work Is Split

The work is intentionally split into Stage A and Stage B to avoid debugging two moving parts at once.

Stage A changes the ML feature path and training artifacts:

- it expands the structured input available to ML
- it changes what the ML models can learn
- it requires real-day retraining and evaluation

Stage B changes the prompt contract for the visual boundary VLM:

- it removes existing raw heuristic and descriptor hints from the prompt
- it inserts compact ML hints instead
- it changes how the VLM interprets supporting context

If both stages are changed together, failures become hard to localize. Splitting them keeps the evaluation loop readable:

1. make ML stronger
2. test ML on real data
3. only then replace raw prompt hints with ML hints

## Preconditions

Stage B may begin only when all of the following are true:

- Stage A is complete
- the user has run real ML evaluation after Stage A
- the user has reviewed the new ML results
- the user explicitly approves moving from raw prompt hints to ML hints

Until then, `probe_vlm_photo_boundaries.py` must keep its current prompt behavior.

## Current Prompt Contract

Today the visual boundary VLM receives:

1. five attached images
2. heuristic gap hints derived from `photo_boundary_scores.csv`
3. per-photo pre-model annotations from `photo_pre_model_annotations`

These structured hints are verbose and heterogeneous. They push a large amount of low-level information into the prompt and make the decision contract noisy.

## Stage B Target Contract

After Stage B, the visual boundary VLM should receive:

1. five attached images
2. a short ML-generated hint block
3. the same core task instructions and response schema

The goal is to move feature integration out of the prompt and into ML.

In other words:

- ML aggregates structured, non-image signals
- VLM receives images plus a compact summary of what ML thinks
- VLM remains the final visual arbiter

## ML Hints To Expose

The prompt should receive the simplest useful ML outputs from both predictors.

Required hint fields:

- `ml_boundary_prediction`
- `ml_boundary_probability`
- `ml_segment_type_prediction`
- `ml_segment_type_probability`

These values should be translated into task-language text rather than raw internal field dumps.

Examples:

- `ML hint for this 5-frame window: likely no cut (confidence 0.88).`
- `ML hint for this 5-frame window: likely cut after frame_03 (confidence 0.82).`
- `ML hint for the likely segment after the boundary: ceremony (confidence 0.74).`

The prompt must never expose unlabeled numeric values such as `ml_boundary_prediction=1` without semantic explanation.

## Prompt Instruction Changes

Stage B must extend the prompt instructions to explain what the ML hint means.

The prompt should explicitly say:

- the ML hint comes from a separate tabular model that aggregates heuristic gap signals and per-photo annotations
- the ML hint is advisory, not authoritative
- if the images clearly contradict the ML hint, the model should trust the images first

The existing decision-priority principle should remain:

- images first
- supporting signals second

The ML hint section should therefore strengthen the prompt structure without turning the VLM into a blind follower of ML.

## Prompt Simplification

Stage B is not just an additive change. It is a replacement of verbose structured hints with compressed ML hints.

Once Stage B is active, the prompt should stop directly injecting:

- the current `build_gap_hint_lines(...)` block
- the current `build_photo_pre_model_lines(...)` block

Those source signals still matter, but only upstream inside ML.

This is the core design reason for the ML-to-VLM handoff.

## Inference Artifact Contract

Stage B needs a stable way to fetch ML predictions for each VLM candidate window at prompt-build time.

The inference artifact should support lookup by the same 5-frame identity used by the VLM boundary probe path.

Preferred matching basis:

- ordered `window_relative_paths`

This is preferable to introducing a separate ad hoc prompt-only key because:

- it already exists in the ML candidate dataset
- it already exists conceptually in the VLM window path
- it preserves alignment with Stage A feature joins

Stage B should therefore use a stable, deterministic per-window inference record keyed by the ordered 5-frame window identity.

## Rollout Plan

Stage B rollout should happen in one guarded switch:

1. keep the current prompt path untouched until ML quality is accepted
2. generate ML inference outputs for candidate windows
3. teach `probe_vlm_photo_boundaries.py` to read those inference outputs
4. swap raw prompt hints for compact ML hints
5. validate that prompt text is materially shorter and easier to interpret

There should be no partial hybrid mode where the prompt contains:

- raw heuristic hints
- raw pre-model lines
- and ML hints

all at once.

That would defeat the simplification goal.

## Validation Goals

When Stage B is eventually implemented, validation should focus on these questions:

1. does the prompt become meaningfully shorter and cleaner?
2. does the VLM still make correct boundary decisions on hard windows?
3. when ML and images disagree, does the VLM still prioritize the images?
4. does replacing raw hints with ML hints improve or degrade final review quality?

These questions are intentionally deferred until after Stage A testing.

## Boundary Between The Stages

Stage A ends at richer ML training and evaluation.

Stage B begins only when the user says the ML outputs are ready to become the hint layer for the VLM.

That boundary is intentional and should be preserved unless the user explicitly changes strategy later.
