# Repository Guidelines

## Project Structure & Module Organization

`vocatio` is currently a migration-stage Python repository. Top-level documentation starts in `README.md`. Operational code is under `scripts/pipeline/`, which contains standalone CLI scripts for media export, sync estimation, transcription, announcement extraction, timeline building, proxy generation, review tooling, and delivery prep. Runtime configuration lives in `conf/`, for example `conf/copy_reviewed_set_assets.default.yaml`.

Treat `scripts/pipeline/` as the active code surface. Keep new helpers close to the workflow they support, and prefer descriptive script names such as `build_*`, `extract_*`, or `generate_*`.

## Build, Test, and Development Commands

There is no packaged build system yet. Run scripts directly with Python 3:

```bash
python3 scripts/pipeline/export_event_media_csv.py --help
python3 scripts/pipeline/build_performance_timeline.py /data/20260323
python3 scripts/pipeline/copy_reviewed_set_assets.py /data/20260323 out 158 --config conf/copy_reviewed_set_assets.default.yaml
```

Useful validation commands:

```bash
python3 scripts/pipeline/test_segment_speech_music.py --help
python3 scripts/pipeline/test_diarize_clip.py --help
```

Several scripts depend on external tools or libraries such as `ffmpeg`, `ffprobe`, `exiftool`, `whisperx`, `PySide6`, `PyYAML`, and `rich`.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, functions and variables in `snake_case`, and explicit type hints where practical. Keep scripts CLI-first with a `parse_args()` and `main()` entrypoint. Prefer `pathlib.Path` over string paths and keep console output readable with `rich` when extending existing scripts.

## Python Script UX and Rich Progress

- Keep scripts self-contained and easy to run.
- Use English only for code, comments, CLI help, logs, and all user-facing text.
- Do not add code comments unless explicitly requested by the user.
- Use `rich.progress` for any script that processes more than one file.
- Use a left-justified, non-expanding progress layout (`expand=False`).
- Use this standard progress column set:
  - `SpinnerColumn()`
  - `TextColumn("[progress.description]{task.description}")`
  - `BarColumn(bar_width=40)`
  - `MofNCompleteColumn()`
  - `TaskProgressColumn()`
  - `TimeElapsedColumn()`
- Keep task descriptions at a fixed width (for example, `.ljust(25)`) so the progress bar does not shift horizontally.

## Testing Guidelines

This repo does not use a centralized test runner yet. Validation is script-based and focused on real workflow data. Name exploratory or verification scripts `test_*.py` inside `scripts/pipeline/`, keep them safe to run locally, and document required input files in the script defaults or help text. When changing pipeline behavior, run the nearest affected script with `--help` and at least one representative invocation.

## Commit & Pull Request Guidelines

Git history is brief, but existing subjects use short imperative summaries such as `Add initial event pipeline migration snapshot`. Keep commit titles concise, capitalized, and focused on one change. PRs should explain the workflow step affected, note any required local tools or sample data, and include screenshots only for GUI changes such as `review_performance_proxy_gui.py`.

## Migration Notes

This repository is a safe copy of logic still active in `scriptoza`. Preserve current behavior unless the change is explicitly part of the migration plan, and document any intentional divergence in the PR description.
