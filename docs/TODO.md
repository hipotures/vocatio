# TODO

## Improve Polish Diarization + ASR Quality in the Pipeline

### Context
Current transcription quality for Polish recordings is inconsistent on noisy or low-clarity material. In practice, the biggest gains are usually not from switching to a larger model, but from tightening the full pipeline configuration: fixed language, deterministic aligner, VAD tuning, prompt biasing, and better handling of numerals.

The project should treat this as a combined ASR + alignment + diarization quality task, not only a single-model issue.

## Priority Plan

### 1) Make Polish language explicit in every transcription entrypoint
- Set `language="pl"` by default where WhisperX is called.
- Avoid per-file language auto-detection for routine Polish batches.
- Keep autodetect only as an explicit fallback mode.

Why:
- Better consistency.
- Lower runtime overhead.
- Fewer language drift errors on short/noisy segments.

### 2) Use an explicit Polish aligner model
- Pin aligner to: `jonatasgrosman/wav2vec2-large-xlsr-53-polish`.
- Do not rely on implicit aligner selection when deterministic output is required.

Why:
- Stable word-level timing behavior.
- Easier debugging and reproducibility across machines/runs.

### 3) Keep `condition_on_previous_text=False`
- Preserve this setting as default in WhisperX ASR options.

Why:
- Reduces cross-window looping and hallucinated continuation on pause-heavy speech.

### 4) Tune VAD before changing ASR model size
- Start with `pyannote` VAD for quality-first setup.
- Keep defaults first, then tune when segment boundaries are cut:
  - `vad_onset`: start near `0.50`
  - `vad_offset`: start near `0.363`
- Test `silero` as an alternative fallback profile.

Why:
- VAD quality has high impact on difficult recordings.
- Boundary errors often look like ASR errors but originate in segmentation.

### 5) Introduce numerals strategy as a first-class pipeline option
Add two operating modes:
- Text fidelity mode (final transcript with digits): `suppress_numerals=False`
- Timing fidelity mode (better alignment stability): `suppress_numerals=True`

Optional advanced path:
- Two-pass workflow:
  - pass A for timing (`suppress_numerals=True`)
  - pass B for text (`suppress_numerals=False`)
  - merge text from B with timing from A

Why:
- Digits often degrade alignment quality.
- Different downstream tasks require different tradeoffs.

### 6) Add domain prompts and hotwords for Polish proper nouns
- Support configurable `initial_prompt` and `hotwords` in pipeline scripts.
- Include recurring names/terms (cities, organizations, event-specific vocabulary).

Why:
- Better recall for proper nouns and abbreviations.
- More stable lexical choices in domain-specific material.

### 7) Keep quality-oriented compute defaults on GPU
- Prefer `compute_type="float16"` for quality runs.
- Treat `int8` as memory-saving fallback, not quality default.
- Treat `batch_size` primarily as throughput/VRAM tuning.

### 8) Improve audio pre-processing discipline
- Ensure uniform input conversion (16 kHz mono path where expected).
- Add optional light denoise pre-step for problematic sources.

Why:
- Input conditioning can yield larger gains than decoder flag changes.

## Recommended Baseline Profile (Polish)

```python
import whisperx

device = "cuda"
audio = whisperx.load_audio("audio.wav")

model = whisperx.load_model(
    "large-v3",
    device=device,
    compute_type="float16",
    language="pl",
    vad_method="pyannote",
    asr_options={
        "suppress_numerals": True,  # switch to False for final text with digits
        "condition_on_previous_text": False,
        "initial_prompt": (
            "This is Polish transcription. Keep Polish proper nouns, abbreviations, "
            "and inflection consistent."
        ),
        "hotwords": "Łódź,Białystok,PZU,PKO BP,KSeF,Orlen",
    },
    vad_options={
        "vad_onset": 0.50,
        "vad_offset": 0.363,
    },
)

result = model.transcribe(audio, batch_size=8)

model_a, metadata = whisperx.load_align_model(
    language_code="pl",
    device=device,
    model_name="jonatasgrosman/wav2vec2-large-xlsr-53-polish",
)

result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False,
)
```

## Implementation Tasks in This Repository

### Script-level updates
- `scripts/pipeline/transcribe_video_batch.py`
  - Add explicit preset profile for Polish (`--language pl` already exists, but enforce profile-level defaults for align/VAD/options).
  - Expose `initial_prompt` and `hotwords` as CLI options.
  - Add optional `--suppress-numerals` / `--no-suppress-numerals` toggles.

- `scripts/pipeline/transcribe_video_batch_api.py`
  - Mirror the same options and defaults as the batch script.
  - Keep behavior parity between local and API-based runs.

- `scripts/pipeline/test_diarize_clip.py`
  - Add ready-to-run quality presets for quick A/B checks.

### Evaluation and regression checks
- Create a fixed Polish validation subset with noisy, clean, and mixed-event clips.
- Compare profiles on:
  - WER/CER (if references exist)
  - boundary quality (start/end clipping)
  - proper noun accuracy
  - numeric stability
  - diarization turn quality (speaker boundary consistency)

## Acceptance Criteria
- Fewer clipped starts/ends of utterances in noisy material.
- Fewer hallucinated continuations across pauses.
- Improved proper noun accuracy in event-specific vocabulary.
- Stable word-level timings for subtitle/timeline use cases.
- Reproducible output between repeated runs with the same config.

## Short Version (Top 4 First Moves)
1. Force `language="pl"`.
2. Pin aligner `jonatasgrosman/wav2vec2-large-xlsr-53-polish`.
3. Use explicit numerals strategy (`suppress_numerals`) by output goal.
4. Tune VAD before changing model size.

## Cross-Pipeline Media Index

- Design and implement a single canonical media indexing file that can serve both the legacy audio-first pipeline and the new image-only pipeline.
- Keep media identity stable across both flows, with one shared contract for source paths, relative paths, timestamps, and logical ordering.
- Ensure the shared index can be rebuilt without breaking legacy artifacts already stored under `_workspace`.

## Review GUI and JSON Contract

- Replace the current free-form naming model in the review GUI with an explicit choice:
  - set number
  - custom name
- Add a dedicated type identifier field to every saved JSON artifact.
- Use one canonical type list everywhere:
  - `performance`
  - `ceremony`
  - `warmup`
- Do not infer type from filename or display name.
- Keep semantic type in a canonical field, because filenames and display names may be arbitrary and can otherwise mix up what a given JSON represents.

## ML Boundary Follow-Up

- Add a separate `left_segment_type` model for the ML boundary pipeline.
- Keep `boundary` and `right_segment_type` as separate predictors; do not try to force a fake multilabel setup in AutoGluon.
- Revisit how `left_segment_type` and `right_segment_type` should be exposed to the downstream VLM prompt after the extra model exists.
- For manual single-gap prediction flows, `VLM_OVERLAP` likely has no value; either remove it from that path or document the exact scenario where it still helps.
- Consider replacing manual-gap `overlap` with a more direct symmetric-context parameter, for example `gap_side_span`, where `1 -> window 2`, `2 -> window 4`, `3 -> window 6`, and the cut is always centered between `L` and `P`.

## Manual VLM Window Config Consistency

- Fix `Manual VLM analyze` so `VLM_WINDOW_RADIUS` is resolved from the current `.vocatio`, not from stale review-index payload metadata.
- Keep `VLM_WINDOW_SCHEMA` and `VLM_WINDOW_SCHEMA_SEED` on the same resolution model as `VLM_WINDOW_RADIUS`:
  - GUI dropdown may override schema for the current manual run.
  - `.vocatio` should remain the default source of truth.
  - Review-index payload should be fallback-only, not the primary runtime source.
- Current bug:
  - changing `VLM_WINDOW_RADIUS` in `.vocatio` may still leave manual GUI on the old window size from payload metadata
  - this makes the panel show a `Window config` inconsistent with the user’s current config expectations
  - in practice, `window_radius=3` still yields 6 images even after `.vocatio` was edited to another value if the loaded payload still carries the older radius
