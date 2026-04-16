#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from build_ml_boundary_split_manifest import OUTPUT_HEADERS as SPLIT_MANIFEST_HEADERS
from build_ml_boundary_split_manifest import build_split_manifest_rows
from lib.ml_boundary_truth import VALID_SEGMENT_TYPES
from lib.pipeline_io import atomic_write_csv, atomic_write_json
from lib.workspace_dir import resolve_workspace_dir


console = Console()

CORPUS_CANDIDATES_FILENAME = "ml_boundary_candidates.corpus.csv"
DAY_METADATA_FILENAME = "ml_boundary_day_metadata.csv"
SPLIT_MANIFEST_FILENAME = "ml_boundary_splits.csv"
VALIDATION_REPORT_FILENAME = "ml_boundary_validation_report.json"
PIPELINE_SUMMARY_FILENAME = "ml_boundary_pipeline_summary.json"


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ML boundary verifier pipeline from reviewed GUI state to train/evaluate artifacts.",
    )
    parser.add_argument(
        "day_dirs",
        nargs="+",
        help="One or more day directories like /data/20260323",
    )
    parser.add_argument(
        "--corpus-workspace",
        help=(
            "Directory for merged corpus artifacts. "
            "Default: first DAY workspace/ml_boundary_corpus"
        ),
    )
    parser.add_argument(
        "--mode",
        default="tabular_only",
        choices=("tabular_only", "tabular_plus_thumbnail"),
        help="Training mode passed to train_ml_boundary_verifier.py",
    )
    parser.add_argument(
        "--model-run-id",
        default="run-001",
        help="Run directory name under ml_boundary_models/ and ml_boundary_eval/. Default: run-001",
    )
    parser.add_argument(
        "--required-heldout-classes",
        nargs="+",
        help=(
            "Required classes for validation/test split coverage. "
            "Default: all segment types present in merged corpus."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after dataset, split, and validation artifacts (skip train/evaluate).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rebuild candidate rows with --overwrite and replace train/eval artifacts.",
    )
    return parser.parse_args(argv)


def _script_path(script_name: str) -> Path:
    return Path(__file__).resolve().parent / script_name


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _resolve_days(day_values: Sequence[str]) -> list[Path]:
    day_dirs: list[Path] = []
    for value in day_values:
        day_dir = Path(value).expanduser().resolve()
        if not day_dir.is_dir():
            raise FileNotFoundError(f"Day directory does not exist: {day_dir}")
        day_dirs.append(day_dir)
    return day_dirs


def _prepare_single_day(day_dir: Path, *, restart: bool) -> Path:
    export_command = [
        sys.executable,
        str(_script_path("export_ml_boundary_reviewed_truth.py")),
        str(day_dir),
    ]
    console.print(f"Run Export reviewed truth: {' '.join(export_command)}")
    _run_command(export_command)

    build_command = [
        sys.executable,
        str(_script_path("build_ml_boundary_candidate_dataset.py")),
        str(day_dir),
    ]
    if restart:
        build_command.append("--overwrite")
    console.print(f"Run Build candidate dataset: {' '.join(build_command)}")
    _run_command(build_command)

    validate_command = [
        sys.executable,
        str(_script_path("validate_ml_boundary_dataset.py")),
        str(day_dir),
    ]
    console.print(f"Run Validate day dataset: {' '.join(validate_command)}")
    _run_command(validate_command)

    return resolve_workspace_dir(day_dir, None)


def _read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [header for header in CANDIDATE_ROW_HEADERS if header not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} has no candidate rows")
    return rows


def _merge_candidate_rows(candidate_csv_paths: Sequence[Path]) -> list[dict[str, str]]:
    merged_rows: list[dict[str, str]] = []
    with _progress() as progress:
        task_id = progress.add_task("Merge candidate rows".ljust(25), total=len(candidate_csv_paths))
        for path in candidate_csv_paths:
            merged_rows.extend(_read_candidate_rows(path))
            progress.advance(task_id)
    merged_rows.sort(
        key=lambda row: (
            str(row.get("day_id", "")),
            str(row.get("candidate_id", "")),
        )
    )
    return merged_rows


def _build_day_metadata_rows(merged_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    segment_types_by_day: dict[str, set[str]] = {}
    for row in merged_rows:
        day_id = str(row.get("day_id", "")).strip()
        segment_type = str(row.get("segment_type", "")).strip()
        if day_id == "" or segment_type == "":
            raise ValueError("merged candidate rows must include non-blank day_id and segment_type")
        if segment_type not in VALID_SEGMENT_TYPES:
            choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
            raise ValueError(
                f"merged candidate rows include unsupported segment_type {segment_type!r}; expected one of {choices}"
            )
        segment_types_by_day.setdefault(day_id, set()).add(segment_type)

    metadata_rows: list[dict[str, str]] = []
    for day_id in sorted(segment_types_by_day):
        year = day_id[:4] if len(day_id) >= 4 and day_id[:4].isdigit() else ""
        metadata_rows.append(
            {
                "day_id": day_id,
                "segment_types": json.dumps(
                    sorted(segment_types_by_day[day_id]),
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                "year": year,
                "camera": "",
                "domain_shift_hint": "",
            }
        )
    return metadata_rows


def _required_classes(
    metadata_rows: Sequence[dict[str, str]],
    explicit_required: Sequence[str] | None,
) -> list[str]:
    if explicit_required:
        values = []
        for value in explicit_required:
            normalized = value.strip()
            if normalized == "":
                raise ValueError("required held-out classes must not include blank values")
            if normalized not in VALID_SEGMENT_TYPES:
                choices = ", ".join(sorted(VALID_SEGMENT_TYPES))
                raise ValueError(f"required held-out classes must be one of: {choices}")
            values.append(normalized)
        return values

    discovered: set[str] = set()
    for row in metadata_rows:
        parsed = json.loads(row["segment_types"])
        if not isinstance(parsed, list):
            raise ValueError("segment_types must be a JSON array in day metadata rows")
        for value in parsed:
            if not isinstance(value, str):
                raise ValueError("segment_types values must be strings")
            discovered.add(value)
    return sorted(discovered)


def _run_training_and_evaluation(
    *,
    corpus_candidates_path: Path,
    split_manifest_path: Path,
    model_dir: Path,
    eval_dir: Path,
    mode: str,
    restart: bool,
) -> None:
    train_command = [
        "uv",
        "run",
        "--no-default-groups",
        "--group",
        "autogluon",
        "python3",
        str(_script_path("train_ml_boundary_verifier.py")),
        str(corpus_candidates_path),
        "--split-manifest-csv",
        str(split_manifest_path),
        "--mode",
        mode,
        "--output-dir",
        str(model_dir),
    ]
    if restart:
        train_command.append("--overwrite")
    console.print(f"Run Train verifier: {' '.join(train_command)}")
    _run_command(train_command)

    evaluate_command = [
        "uv",
        "run",
        "--no-default-groups",
        "--group",
        "autogluon",
        "python3",
        str(_script_path("evaluate_ml_boundary_verifier.py")),
        str(corpus_candidates_path),
        "--model-dir",
        str(model_dir),
        "--split-manifest-csv",
        str(split_manifest_path),
        "--output-dir",
        str(eval_dir),
    ]
    if restart:
        evaluate_command.append("--overwrite")
    console.print(f"Run Evaluate verifier: {' '.join(evaluate_command)}")
    _run_command(evaluate_command)


def _write_pipeline_summary(
    *,
    summary_path: Path,
    day_dirs: Sequence[Path],
    day_workspaces: Sequence[Path],
    corpus_candidates_path: Path,
    day_metadata_path: Path,
    split_manifest_path: Path | None,
    validation_report_path: Path | None,
    model_dir: Path | None,
    eval_dir: Path | None,
    required_heldout_classes: Sequence[str],
    mode: str,
    prepare_only: bool,
    note: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "day_dirs": [str(path) for path in day_dirs],
        "day_workspaces": [str(path) for path in day_workspaces],
        "corpus_candidates_csv": str(corpus_candidates_path),
        "day_metadata_csv": str(day_metadata_path),
        "required_heldout_classes": list(required_heldout_classes),
        "mode": mode,
        "prepare_only": prepare_only,
    }
    if split_manifest_path is not None:
        payload["split_manifest_csv"] = str(split_manifest_path)
    if validation_report_path is not None:
        payload["validation_report_json"] = str(validation_report_path)
    if model_dir is not None:
        payload["model_dir"] = str(model_dir)
    if eval_dir is not None:
        payload["eval_dir"] = str(eval_dir)
    if note:
        payload["note"] = note
    atomic_write_json(summary_path, payload)


def _minimum_day_count_note(day_count: int) -> str:
    return (
        "ML boundary training/evaluation requires at least three day_id values to build "
        f"train/validation/test splits without leakage; got {day_count}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    day_dirs = _resolve_days(args.day_dirs)
    day_workspaces: list[Path] = []
    for day_dir in day_dirs:
        workspace_dir = _prepare_single_day(day_dir, restart=args.restart)
        day_workspaces.append(workspace_dir)

    if args.corpus_workspace:
        corpus_workspace = Path(args.corpus_workspace).expanduser().resolve()
    else:
        corpus_workspace = (day_workspaces[0] / "ml_boundary_corpus").resolve()
    corpus_workspace.mkdir(parents=True, exist_ok=True)

    candidate_csv_paths = [workspace / "ml_boundary_candidates.csv" for workspace in day_workspaces]
    merged_rows = _merge_candidate_rows(candidate_csv_paths)
    corpus_candidates_path = corpus_workspace / CORPUS_CANDIDATES_FILENAME
    atomic_write_csv(corpus_candidates_path, CANDIDATE_ROW_HEADERS, merged_rows)
    console.print(f"Wrote merged candidate dataset: {corpus_candidates_path}")

    day_metadata_rows = _build_day_metadata_rows(merged_rows)
    day_metadata_headers = ["day_id", "segment_types", "year", "camera", "domain_shift_hint"]
    day_metadata_path = corpus_workspace / DAY_METADATA_FILENAME
    atomic_write_csv(day_metadata_path, day_metadata_headers, day_metadata_rows)
    console.print(f"Wrote day metadata: {day_metadata_path}")

    if len(day_metadata_rows) < 3:
        note = _minimum_day_count_note(len(day_metadata_rows))
        summary_path = corpus_workspace / PIPELINE_SUMMARY_FILENAME
        if args.prepare_only:
            _write_pipeline_summary(
                summary_path=summary_path,
                day_dirs=day_dirs,
                day_workspaces=day_workspaces,
                corpus_candidates_path=corpus_candidates_path,
                day_metadata_path=day_metadata_path,
                split_manifest_path=None,
                validation_report_path=None,
                model_dir=None,
                eval_dir=None,
                required_heldout_classes=[],
                mode=args.mode,
                prepare_only=True,
                note=note,
            )
            console.print(f"Stop after corpus preparation: {note}")
            console.print(f"Wrote pipeline summary: {summary_path}")
            return 0
        raise ValueError(f"{note} Pass at least three day directories or use --prepare-only.")

    required_heldout_classes = _required_classes(
        day_metadata_rows,
        explicit_required=args.required_heldout_classes,
    )
    split_rows = build_split_manifest_rows(
        day_metadata_rows,
        required_heldout_classes=required_heldout_classes,
    )
    split_manifest_path = corpus_workspace / SPLIT_MANIFEST_FILENAME
    atomic_write_csv(split_manifest_path, SPLIT_MANIFEST_HEADERS, split_rows)
    console.print(f"Wrote split manifest: {split_manifest_path}")

    validate_command = [
        sys.executable,
        str(_script_path("validate_ml_boundary_dataset.py")),
        str(corpus_candidates_path),
        "--split-manifest-csv",
        str(split_manifest_path),
        "--report-json",
        str(corpus_workspace / VALIDATION_REPORT_FILENAME),
    ]
    if required_heldout_classes:
        validate_command.append("--required-heldout-classes")
        validate_command.extend(required_heldout_classes)
    console.print(f"Run Validate corpus dataset: {' '.join(validate_command)}")
    _run_command(validate_command)

    model_dir: Path | None = None
    eval_dir: Path | None = None
    if not args.prepare_only:
        model_dir = corpus_workspace / "ml_boundary_models" / args.model_run_id
        eval_dir = corpus_workspace / "ml_boundary_eval" / args.model_run_id
        _run_training_and_evaluation(
            corpus_candidates_path=corpus_candidates_path,
            split_manifest_path=split_manifest_path,
            model_dir=model_dir,
            eval_dir=eval_dir,
            mode=args.mode,
            restart=args.restart,
        )

    summary_path = corpus_workspace / PIPELINE_SUMMARY_FILENAME
    _write_pipeline_summary(
        summary_path=summary_path,
        day_dirs=day_dirs,
        day_workspaces=day_workspaces,
        corpus_candidates_path=corpus_candidates_path,
        day_metadata_path=day_metadata_path,
        split_manifest_path=split_manifest_path,
        validation_report_path=corpus_workspace / VALIDATION_REPORT_FILENAME,
        model_dir=model_dir,
        eval_dir=eval_dir,
        required_heldout_classes=required_heldout_classes,
        mode=args.mode,
        prepare_only=args.prepare_only,
    )
    console.print(f"Wrote pipeline summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
